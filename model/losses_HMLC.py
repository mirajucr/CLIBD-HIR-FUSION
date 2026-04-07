from __future__ import print_function

import torch
import torch.nn as nn


def unique_row_by_col_index(x, column_index):
    non_zero_mask = x[:, column_index] != 0
    non_zero_original_indices = torch.where(non_zero_mask)[0]
    filtered_tensor = x[non_zero_mask]

    unique_values, inverse_indices = torch.unique(
        filtered_tensor[:, column_index], return_inverse=True
    )
    perm = torch.arange(
        inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device
    )
    inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
    reverse = inverse_indices.new_empty(unique_values.size(0)).scatter_(0, inverse_indices, perm)

    flip_non_zero_mask = ~non_zero_mask
    flip_non_zero_mask[non_zero_original_indices[reverse]] = True
    filtered_original_tensor = x[flip_non_zero_mask]
    reverse_indices = torch.nonzero(flip_non_zero_mask).squeeze()
    return filtered_original_tensor, reverse_indices


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("`features` must be [batch, views, dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Pass either labels or mask, not both.")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class HMLC(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, layer_penalty=None, loss_type="hmce"):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.layer_penalty = layer_penalty or self.pow_2
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def compare_rows_custom(self, row1, row2, index1, index2, target_col):
        if index1 == index2:
            return True
        val1 = row1[target_col]
        val2 = row2[target_col]
        if val1 == 0 or val2 == 0:
            return False
        return val1 == val2

    def gen_mask_labels(self, layer_labels, target_col, device="cpu"):
        mask_labels = []
        for i in range(layer_labels.shape[0]):
            mask = torch.tensor(
                [
                    self.compare_rows_custom(layer_labels[i], layer_labels[j], i, j, target_col)
                    for j in range(layer_labels.shape[0])
                ]
            )
            mask_labels.append(mask)
        return torch.stack(mask_labels).type(torch.uint8).to(device)

    def forward(self, features, labels):
        device = features.device
        mask = torch.ones(labels.shape, device=device)
        cumulative_loss = torch.tensor(0.0, device=device)
        max_loss_lower_layer = torch.tensor(float("-inf"), device=device)
        layers_with_loss = torch.tensor(0.0, device=device)

        for layer_offset in range(1, labels.shape[1]):
            target_col = labels.shape[1] - layer_offset - 1
            if (labels[:, target_col] == 0).all(dim=0):
                continue

            mask[:, labels.shape[1] - layer_offset :] = 0
            layer_labels = labels * mask
            mask_labels = self.gen_mask_labels(layer_labels, target_col, device)
            layer_loss = self.sup_con_loss(features, mask=mask_labels)

            if self.loss_type == "hmc":
                cumulative_loss += self.layer_penalty(torch.tensor(1 / layer_offset).float()) * layer_loss
            elif self.loss_type == "hce":
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == "hmce":
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(1 / layer_offset).float()) * layer_loss
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

            layers_with_loss += 1
            _, unique_indices = unique_row_by_col_index(layer_labels, column_index=target_col)
            max_loss_lower_layer = torch.max(max_loss_lower_layer, layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]

        return cumulative_loss / layers_with_loss
