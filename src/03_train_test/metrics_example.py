#!/usr/bin/env python3

def ssim(pred, target, c1=0.01 ** 2, c2=0.03 ** 2):
    if pred.shape != target.shape:
        raise ValueError("Predicted and target tensors must have the same shape.")
    if pred.dim() != 4:
        raise ValueError("Expected input shape (N, C, H, W).")

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    mu_x = pred_flat.mean(dim=1)
    mu_y = target_flat.mean(dim=1)
    var_x = pred_flat.var(dim=1, unbiased=False)
    var_y = target_flat.var(dim=1, unbiased=False)
    cov = ((pred_flat - mu_x[:, None]) * (target_flat - mu_y[:, None])).mean(dim=1)

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return numerator / denominator
