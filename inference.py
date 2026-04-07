#!/usr/bin/env python3
import argparse
import os

import numpy as np
import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model.dna_encoder import load_pre_trained_dnabert2, load_pre_trained_modified_dnabert2
from model.simple_clip import SimpleCLIP, unwrap
from utils.data import CsvDataset, CsvDatasetText


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
        init_process_group(backend=backend)

    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        destroy_process_group()


NUCS = np.array(list("ACGT"))


def _rng(seed):
    return np.random.RandomState(seed)


def substitute(seq_arr, p_sub, rs):
    if p_sub <= 0:
        return seq_arr
    mask = rs.rand(len(seq_arr)) < p_sub
    if not mask.any():
        return seq_arr
    out = seq_arr.copy()
    for idx in np.flatnonzero(mask):
        base = out[idx]
        if base not in "ACGT":
            continue
        alts = [b for b in "ACGT" if b != base]
        out[idx] = alts[rs.randint(0, 3)]
    return out


def mask_ambiguous(seq_arr, p_mask, rs):
    if p_mask <= 0:
        return seq_arr
    mask = rs.rand(len(seq_arr)) < p_mask
    if not mask.any():
        return seq_arr
    out = seq_arr.copy()
    out[mask] = "N"
    return out


def apply_indels(seq_arr, p_ins, p_del, rs):
    if p_ins <= 0 and p_del <= 0:
        return seq_arr
    out = []
    for ch in seq_arr:
        if p_del > 0 and rs.rand() < p_del:
            continue
        out.append(ch)
        if p_ins > 0 and rs.rand() < p_ins:
            out.append(rs.choice(NUCS))
    return np.array(out, dtype="<U1") if out else np.array(["N"], dtype="<U1")


def contiguous_dropout(seq_arr, run_frac, rs):
    if run_frac <= 0:
        return seq_arr
    run_len = max(1, int(round(run_frac * len(seq_arr))))
    run_len = min(run_len, len(seq_arr))
    start = 0 if len(seq_arr) == run_len else rs.randint(0, len(seq_arr) - run_len + 1)
    out = seq_arr.copy()
    out[start : start + run_len] = "N"
    return out


def truncate_tail(seq_arr, frac):
    if frac <= 0:
        return seq_arr
    keep = max(1, int(round((1.0 - frac) * len(seq_arr))))
    return seq_arr[:keep]


def corrupt_one_sequence(seq_str, rs, p_sub, p_ins, p_del, p_mask, truncate_frac, run_frac):
    if not isinstance(seq_str, str):
        return seq_str
    seq_str = "".join([c for c in seq_str.upper() if c in {"A", "C", "G", "T", "N"}])
    if not seq_str:
        return "N"
    arr = np.array(list(seq_str), dtype="<U1")
    arr = substitute(arr, p_sub, rs)
    arr = mask_ambiguous(arr, p_mask, rs)
    arr = apply_indels(arr, p_ins, p_del, rs)
    arr = contiguous_dropout(arr, run_frac, rs)
    arr = truncate_tail(arr, truncate_frac)
    if len(arr) == 0:
        arr = np.array(["N"], dtype="<U1")
    return "".join(arr.tolist())


def maybe_corrupt_batch(dna_batch, args, step, rank):
    if not args.noise:
        return dna_batch
    seed = (args.noise_seed + 1009 * (rank + 1) + 7919 * (step + 1)) & 0x7FFFFFFF
    rs = _rng(seed)
    return [
        corrupt_one_sequence(
            seq,
            rs,
            args.p_sub,
            args.p_ins,
            args.p_del,
            args.p_mask,
            args.truncate_frac,
            args.run_frac,
        )
        for seq in dna_batch
    ]


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

    def forward(self, z_img, z_dna):
        gate = self.gate(torch.cat([z_img, z_dna], dim=-1))
        return gate * z_img + (1.0 - gate) * z_dna


def load_model(args, device):
    if args.backbone == "bioclip":
        dna_encoder, dna_tokenizer = load_pre_trained_modified_dnabert2(args.dnabert2_ckpt)
        open_clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip",
            pretrained=args.openclip_ckpt,
        )
        text_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    else:
        dna_encoder, dna_tokenizer = load_pre_trained_dnabert2(args.dnabert2_ckpt)
        open_clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L/14",
            pretrained=args.openclip_ckpt,
        )
        text_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = SimpleCLIP(
        dna_encoder=dna_encoder,
        dna_tokenizer=dna_tokenizer,
        device=device,
        open_clip_model=open_clip_model,
    ).to(device)

    state_dict = torch.load(args.full_model_ckpt, map_location="cpu")
    fusion_keys = [k for k in state_dict if k.startswith("fusion_head.")]
    if fusion_keys:
        out_dim = state_dict["fusion_head.gate.3.bias"].shape[0]
        hidden_dim = state_dict["fusion_head.gate.3.weight"].shape[1]
        hidden_mult = max(1, hidden_dim // out_dim)
        unwrap(model).fusion_head = GatedFusion(out_dim, hidden_mult=hidden_mult).to(device)

    model.load_state_dict(state_dict, strict=True)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model, preprocess, text_tokenizer


def accuracy(logits, captions_label, caption_set):
    logits_np = logits.detach().cpu().numpy()
    top_5_idx = np.argsort(logits_np, axis=-1)[..., -5:][..., ::-1]
    top1_correct = 0
    top5_correct = 0
    for i, label in enumerate(list(captions_label)):
        candidates = [caption_set[idx] for idx in top_5_idx[i]]
        if candidates[0] == label:
            top1_correct += 1
        if label in candidates:
            top5_correct += 1
    return top1_correct, top5_correct, len(captions_label)


def reduce_tensor(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for the BIOSCAN-CLIP paper release.")
    parser.add_argument("--infer-mode", default="image", choices=["image", "dna", "fused", "bestof"])
    parser.add_argument("--backbone", default="bioclip", choices=["bioclip", "vitl14"])
    parser.add_argument("--dnabert2-ckpt", required=True, help="Local path or model identifier for DNABERT-2.")
    parser.add_argument("--openclip-ckpt", required=True, help="OpenCLIP weights path or identifier.")
    parser.add_argument("--full-model-ckpt", required=True, help="Path to the trained BIOSCAN-CLIP checkpoint.")
    parser.add_argument("--test-csv", required=True, help="CSV file with taxonomy, image ids, and DNA sequences.")
    parser.add_argument("--test-hdf5", required=True, help="HDF5 file containing BIOSCAN images.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--noise", action="store_true", help="Enable DNA corruption at inference.")
    parser.add_argument("--p-sub", type=float, default=0.01)
    parser.add_argument("--p-ins", type=float, default=0.002)
    parser.add_argument("--p-del", type=float, default=0.002)
    parser.add_argument("--p-mask", type=float, default=0.003)
    parser.add_argument("--truncate-frac", type=float, default=0.10)
    parser.add_argument("--run-frac", type=float, default=0.05)
    parser.add_argument("--noise-seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    device, rank, _ = setup_distributed()
    rank0 = rank == 0

    model, preprocess, text_tokenizer = load_model(args, device)
    unwrap(model).eval()

    test_dataset = CsvDataset(
        input_filename=args.test_csv,
        image_filename=args.test_hdf5,
        img_transforms=preprocess,
        img_key="image_file",
        caption_key="taxonomy",
        dna_key="nucraw",
        text_tokenizer=text_tokenizer,
    )
    sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    caption_dataset = CsvDatasetText(
        input_filename=args.test_csv,
        caption_key="taxonomy",
        text_tokenizer=text_tokenizer,
    )
    caption_loader = DataLoader(
        caption_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        caption_embeddings = []
        for batch_caption in caption_loader:
            batch_caption = batch_caption.to(device)
            caption_embeddings.append(unwrap(model).open_clip_model.encode_text(batch_caption).cpu())
        caption_embeddings = torch.cat(caption_embeddings, dim=0).to(device)
        caption_embeddings = F.normalize(caption_embeddings, dim=-1)

    caption_set = caption_dataset.unique_captions
    totals = torch.zeros(3, device=device)
    logit_scale = 1.0 / args.temperature

    with torch.no_grad():
        iterator = tqdm(dataloader, disable=not rank0)
        for step, batch in enumerate(iterator):
            image_input, language_input, captions_label, dna_input = batch
            image_input = image_input.to(device, non_blocking=True)
            language_input = language_input.to(device, non_blocking=True)
            dna_input = maybe_corrupt_batch(dna_input, args, step, rank)

            z_img, z_dna, _ = model(image_input, language_input, dna_input)
            z_img = F.normalize(z_img, dim=-1)
            z_dna = F.normalize(z_dna, dim=-1)

            if args.infer_mode == "image":
                logits = logit_scale * (z_img @ caption_embeddings.T)
            elif args.infer_mode == "dna":
                logits = logit_scale * (z_dna @ caption_embeddings.T)
            elif args.infer_mode == "fused":
                if not hasattr(unwrap(model), "fusion_head"):
                    raise RuntimeError("Checkpoint does not contain a fusion head.")
                z_fused = unwrap(model).fusion_head(z_img, z_dna)
                logits = logit_scale * (z_fused @ caption_embeddings.T)
            else:
                logits_img = logit_scale * (z_img @ caption_embeddings.T)
                logits_dna = logit_scale * (z_dna @ caption_embeddings.T)
                choices = [logits_img, logits_dna]
                best_scores = [logits_img.max(dim=1).values, logits_dna.max(dim=1).values]
                if hasattr(unwrap(model), "fusion_head"):
                    z_fused = unwrap(model).fusion_head(z_img, z_dna)
                    logits_fused = logit_scale * (z_fused @ caption_embeddings.T)
                    choices.append(logits_fused)
                    best_scores.append(logits_fused.max(dim=1).values)
                stacked_scores = torch.stack(best_scores, dim=1)
                picked = stacked_scores.argmax(dim=1)
                logits = torch.empty_like(logits_img)
                for idx, choice_logits in enumerate(choices):
                    mask = (picked == idx).unsqueeze(1).expand_as(choice_logits)
                    logits[mask] = choice_logits[mask]

            top1_correct, top5_correct, batch_size = accuracy(logits, captions_label, caption_set)
            totals += torch.tensor([top1_correct, top5_correct, batch_size], device=device, dtype=torch.float32)
            if rank0:
                iterator.set_description(
                    f"{args.infer_mode} top1={totals[0] / totals[2]:.2%} top5={totals[1] / totals[2]:.2%}"
                )

    totals = reduce_tensor(totals)
    if rank0:
        print(f"Final Top-1: {(totals[0] / totals[2]).item():.2%}")
        print(f"Final Top-5: {(totals[1] / totals[2]).item():.2%}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
