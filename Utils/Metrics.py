import torch

def spearman_correlation(x, y):
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    return torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1].item()

def ratio_metric(gt, pred):
    """
    Measures ratio consistency only for non-zero ground truth distances.
    Returns CV of the ratios (std/mean) for these cases.
    """
    # Only consider cases where GT is not zero
    nonzero_mask = gt > 0
    if not nonzero_mask.any():  # If no non-zero cases
        raise ValueError("All ground truth distances are zero!")

    valid_gt = gt[nonzero_mask]
    valid_pred = pred[nonzero_mask]

    # Calculate ratios
    ratios = valid_pred / valid_gt
    return ratios.std() / ratios.mean()

def pearson_correlation(x, y):
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()