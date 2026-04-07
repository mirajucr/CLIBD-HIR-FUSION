#!/usr/bin/env python3
import argparse
import os
import time
from types import SimpleNamespace

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from model.dna_encoder import load_pre_trained_Modified_dnabert2, load_pre_trained_dnabert2
from model.loss_func import ClipLossNew, ClipLossNewTogether
from model.losses_HMLC import HMLC
from model.simple_clip import SimpleCLIP, unwrap
from utils.data import BIOSCANHierarchicalDataset, HierarchicalBatchSampler
from utils.utils import AverageMeter, ProgressMeter, TwoCropTransform

try:
    import wandb
except ImportError:
    wandb = None


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    is_distributed = world_size > 1 and "RANK" in os.environ
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class GatedFusion(nn.Module):
    def __init__(self, d_embed, hidden_mult=2, dropout=0.0):
        super().__init__()
        hidden = hidden_mult * d_embed
        self.gate = nn.Sequential(
            nn.Linear(2 * d_embed, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_embed),
            nn.Sigmoid(),
        )

    def forward(self, image_embedding, dna_embedding):
        gate = self.gate(torch.cat([image_embedding, dna_embedding], dim=-1))
        fused = gate * image_embedding + (1.0 - gate) * dna_embedding
        return F.normalize(fused, dim=-1)


def flatten_training_batch(batch, device):
    image_input_batch, hierarchy_labels, language_input_batch, dna_input_batch, label_batch = batch
    hierarchy_labels = hierarchy_labels.squeeze().to(device)
    language_input_batch = torch.cat(language_input_batch).to(device)
    dna_input_batch = tuple(item[0] for item in dna_input_batch)
    label_batch = torch.tensor(label_batch, dtype=torch.long, device=device)
    image_input_batch = torch.cat(
        [image_input_batch[0].squeeze(), image_input_batch[1].squeeze()],
        dim=0,
    ).to(device)
    return image_input_batch, hierarchy_labels, language_input_batch, dna_input_batch, label_batch


def build_dataloaders(preprocess, text_tokenizer, config):
    train_dataset = BIOSCANHierarchicalDataset(
        input_filename=config.dataset_config.csv_train_path,
        image_filename=config.dataset_config.image_hdf5_path,
        img_transforms=TwoCropTransform(preprocess),
        img_key="image_file",
        caption_key="taxonomy",
        dna_key="nucraw",
        text_tokenizer=text_tokenizer,
    )
    val_dataset = BIOSCANHierarchicalDataset(
        input_filename=config.dataset_config.csv_val_path,
        image_filename=config.dataset_config.image_hdf5_path,
        img_transforms=TwoCropTransform(preprocess),
        img_key="image_file",
        caption_key="taxonomy",
        dna_key="nucraw",
        text_tokenizer=text_tokenizer,
    )

    train_sampler = HierarchicalBatchSampler(
        batch_size=config.model_config.batch_size,
        drop_last=False,
        dataset=train_dataset,
    )
    val_sampler = HierarchicalBatchSampler(
        batch_size=config.model_config.batch_size,
        drop_last=False,
        dataset=val_dataset,
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=1,
            num_workers=config.dataset_config.num_workers,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=1,
            num_workers=config.dataset_config.num_workers,
            pin_memory=True,
        ),
    }
    return dataloaders, {"train": train_sampler, "val": val_sampler}


def build_model_and_tokenizers(config, device):
    if config.model_config.use_bioclip:
        open_clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip",
            pretrained=config.pre_trained_model_config.open_clip_bioclip,
        )
        text_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
        dna_encoder, dna_tokenizer = load_pre_trained_Modified_dnabert2(
            config.pre_trained_model_config.dnabert2
        )
    else:
        open_clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L/14",
            pretrained=config.pre_trained_model_config.open_clip_vitl14,
        )
        text_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        dna_encoder, dna_tokenizer = load_pre_trained_dnabert2(config.pre_trained_model_config.dnabert2)

    if config.model_config.fix_text_encoder:
        for param in open_clip_model.token_embedding.parameters():
            param.requires_grad = False
        open_clip_model.positional_embedding.requires_grad = False
        for param in open_clip_model.transformer.parameters():
            param.requires_grad = False
        for param in open_clip_model.ln_final.parameters():
            param.requires_grad = False
        if open_clip_model.text_projection is not None:
            if isinstance(open_clip_model.text_projection, nn.Linear):
                for param in open_clip_model.text_projection.parameters():
                    param.requires_grad = False
            else:
                open_clip_model.text_projection.requires_grad = False

    model = SimpleCLIP(
        dna_encoder=dna_encoder,
        dna_tokenizer=dna_tokenizer,
        device=device,
        open_clip_model=open_clip_model,
    ).to(device)
    return model, preprocess, text_tokenizer


def attach_fusion_head(model, sample_batch, device, hidden_mult, dropout):
    image_input_batch, _, language_input_batch, dna_input_batch, _ = flatten_training_batch(sample_batch, device)
    with torch.no_grad():
        image_output, _, _ = model(image_input_batch, language_input_batch, dna_input_batch)
        embedding_dim = int(image_output.shape[-1])
    model.fusion_head = GatedFusion(embedding_dim, hidden_mult=hidden_mult, dropout=dropout).to(device)


def compute_losses(batch, model, criterion_clip, criterion_pair, criterion_hmlc, device, config, detach_dna_for_fusion):
    image_input_batch, hierarchy_labels, language_input_batch, dna_input_batch, label_batch = flatten_training_batch(
        batch, device
    )
    image_output, dna_output, language_output = model(image_input_batch, language_input_batch, dna_input_batch)
    image_no_aug = image_output[: int(image_input_batch.size(0) / 2)]

    if config.model_config.use_gather:
        clip_loss, image_text_loss, dna_text_loss = criterion_clip(
            image_no_aug,
            dna_output,
            language_output,
            label_batch,
            weights=config.model_config.alpha,
        )
    else:
        image_text_loss = (
            criterion_clip(image_no_aug, language_output, label_batch, device)
            + criterion_clip(language_output, image_no_aug, label_batch, device)
        ) / 2
        dna_text_loss = (
            criterion_clip(dna_output, language_output, label_batch, device)
            + criterion_clip(language_output, dna_output, label_batch, device)
        ) / 2
        image_dna_loss = (
            criterion_clip(image_no_aug, dna_output.detach(), label_batch, device)
            + criterion_clip(dna_output.detach(), image_no_aug, label_batch, device)
        ) / 2
        clip_loss = image_text_loss + dna_text_loss + image_dna_loss

    batch_size = hierarchy_labels.shape[0]
    first_view, second_view = torch.split(image_output, [batch_size, batch_size], dim=0)
    features = torch.cat([first_view.unsqueeze(1), second_view.unsqueeze(1)], dim=1)
    hierarchy_loss = criterion_hmlc(features, hierarchy_labels)

    fusion_head = unwrap(model).fusion_head
    fused_output = fusion_head(image_no_aug, dna_output.detach() if detach_dna_for_fusion else dna_output)
    fused_text_loss = criterion_pair(fused_output, language_output, label_batch, device)

    gamma = getattr(config.model_config, "gamma", 1.0)
    beta = getattr(config.model_config, "beta", 1.0)
    total_loss = clip_loss + gamma * fused_text_loss + (1 - beta) * hierarchy_loss

    return {
        "total_loss": total_loss,
        "image_text_loss": image_text_loss,
        "dna_text_loss": dna_text_loss,
        "fused_text_loss": fused_text_loss,
        "hierarchy_loss": hierarchy_loss,
    }


def run_epoch(split, epoch, dataloader, model, optimizer, scheduler, criterion_clip, criterion_pair, criterion_hmlc, device, config, use_wandb):
    is_train = split == "train"
    model.train(is_train)

    batch_time = AverageMeter(f"{split}_batch_time", ":6.3f")
    data_time = AverageMeter(f"{split}_data_time", ":6.3f")
    total_losses = AverageMeter(f"{split}_total_loss", ":.4f")
    image_text_losses = AverageMeter(f"{split}_img2text_loss", ":.4f")
    dna_text_losses = AverageMeter(f"{split}_dna2text_loss", ":.4f")
    fused_text_losses = AverageMeter(f"{split}_imgdna2text_loss", ":.4f")
    hierarchy_losses = AverageMeter(f"{split}_hierarchy_loss", ":.4f")

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, total_losses, image_text_losses, dna_text_losses, fused_text_losses, hierarchy_losses],
        prefix=f"{split.capitalize()} Epoch: [{epoch}]",
    )

    end = time.time()
    context_manager = torch.enable_grad() if is_train else torch.no_grad()
    with context_manager:
        for step, batch in enumerate(dataloader):
            data_time.update(time.time() - end)
            losses = compute_losses(
                batch,
                model,
                criterion_clip,
                criterion_pair,
                criterion_hmlc,
                device,
                config,
                detach_dna_for_fusion=is_train,
            )

            if is_train:
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = batch[0][0].shape[0] * 2
            total_losses.update(losses["total_loss"].item(), batch_size)
            image_text_losses.update(losses["image_text_loss"].item(), batch_size)
            dna_text_losses.update(losses["dna_text_loss"].item(), batch_size)
            fused_text_losses.update(losses["fused_text_loss"].item(), batch_size)
            hierarchy_losses.update(losses["hierarchy_loss"].item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.model_config.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                total_losses.all_reduce()
                image_text_losses.all_reduce()
                dna_text_losses.all_reduce()
                fused_text_losses.all_reduce()
                hierarchy_losses.all_reduce()

                if (not dist.is_initialized() or dist.get_rank() == 0) and use_wandb:
                    progress.display(step + 1)
                    wandb.log(
                        {
                            f"{split}_total_loss": total_losses.avg,
                            f"{split}_img_text_loss": image_text_losses.avg,
                            f"{split}_dna_text_loss": dna_text_losses.avg,
                            f"{split}_fused_text_loss": fused_text_losses.avg,
                            f"{split}_hierarchy_loss": hierarchy_losses.avg,
                            "epoch": epoch,
                            "step": step,
                        }
                    )

                batch_time.reset()
                data_time.reset()
                total_losses.reset()
                image_text_losses.reset()
                dna_text_losses.reset()
                fused_text_losses.reset()
                hierarchy_losses.reset()

    return total_losses.avg


def save_checkpoint(model, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
    torch.save(unwrap(model).state_dict(), checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Training entrypoint for the BIOSCAN-CLIP paper-ready release.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = to_namespace(raw_config)

    device, rank, world_size = setup_distributed()
    use_wandb = bool(getattr(config.wandb_config, "wandb", False)) and wandb is not None and rank == 0
    if use_wandb:
        os.environ["WANDB_API_KEY"] = getattr(config.wandb_config, "api_key", "")
        wandb.init(project=config.wandb_config.wandb_project_name, config=raw_config)

    model, preprocess, text_tokenizer = build_model_and_tokenizers(config, device)
    dataloaders, samplers = build_dataloaders(preprocess, text_tokenizer, config)
    attach_fusion_head(
        model,
        next(iter(dataloaders["train"])),
        device,
        hidden_mult=getattr(config.model_config.fusion, "hidden_mult", 2),
        dropout=getattr(config.model_config.fusion, "dropout", 0.0),
    )

    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = optim.AdamW(model.parameters(), lr=config.model_config.lr_config.lr)
    total_steps = len(dataloaders["train"]) * config.model_config.epochs
    scheduler = None
    if getattr(config.model_config, "lr_scheduler", "") == "one_cycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.model_config.lr_config.max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    if getattr(config.model_config, "use_gather", False):
        criterion_clip = ClipLossNewTogether(
            local_loss=False,
            gather_with_grad=False,
            rank=rank,
            world_size=world_size,
            criterion=ClipLossNew(temperature=0.07),
        )
        criterion_pair = ClipLossNew(temperature=0.07)
    else:
        criterion_clip = ClipLossNew(temperature=0.07)
        criterion_pair = criterion_clip
    criterion_hmlc = HMLC(loss_type="hmce")

    for epoch in range(config.model_config.epochs):
        samplers["train"].set_epoch(epoch)
        samplers["val"].set_epoch(epoch)

        run_epoch(
            "val",
            epoch,
            dataloaders["val"],
            model,
            optimizer,
            scheduler,
            criterion_clip,
            criterion_pair,
            criterion_hmlc,
            device,
            config,
            use_wandb,
        )
        run_epoch(
            "train",
            epoch,
            dataloaders["train"],
            model,
            optimizer,
            scheduler,
            criterion_clip,
            criterion_pair,
            criterion_hmlc,
            device,
            config,
            use_wandb,
        )

        if rank == 0:
            save_checkpoint(model, config.model_config.save_path, epoch)

    if use_wandb:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
