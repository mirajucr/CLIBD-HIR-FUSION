import torch
import torch.nn as nn

try:
    import torch.distributed.nn
    import torch.distributed as dist

    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False
    dist = None


def gather_features(
    features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    if world_size == 1:
        return features
    if not HAS_DISTRIBUTED:
        raise RuntimeError("torch.distributed is required for feature gathering.")

    if gather_with_grad:
        return torch.cat(torch.distributed.nn.all_gather(features), dim=0)

    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_features, features)
    if not local_loss:
        gathered_features[rank] = features
    return torch.cat(gathered_features, dim=0)


class ClipLossNew(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, contrast_feature, anchor_feature, labels, device):
        batch_size = contrast_feature.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels does not match number of features.")

        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class ClipLossNewTogether(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        criterion=None,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.criterion = criterion or ClipLossNew(temperature=0.07)

    def forward(self, image_features, dna_features, text_features, labels, weights=1.0):
        device = image_features.device

        all_image_features = gather_features(
            image_features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
        all_dna_features = gather_features(
            dna_features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
        all_text_features = gather_features(
            text_features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
        all_labels = gather_features(
            labels.unsqueeze(-1).float(),
            local_loss=self.local_loss,
            gather_with_grad=False,
            rank=self.rank,
            world_size=self.world_size,
        ).squeeze(-1).long()

        img_text_loss = (
            self.criterion(all_image_features, all_text_features, all_labels, device)
            + self.criterion(all_text_features, all_image_features, all_labels, device)
        ) / 2
        dna_text_loss = (
            self.criterion(all_dna_features, all_text_features, all_labels, device)
            + self.criterion(all_text_features, all_dna_features, all_labels, device)
        ) / 2
        img_dna_loss = (
            self.criterion(all_image_features, all_dna_features, all_labels, device)
            + self.criterion(all_dna_features, all_image_features, all_labels, device)
        ) / 2

        total_loss = img_text_loss + dna_text_loss + weights * img_dna_loss
        return total_loss, img_text_loss, dna_text_loss
