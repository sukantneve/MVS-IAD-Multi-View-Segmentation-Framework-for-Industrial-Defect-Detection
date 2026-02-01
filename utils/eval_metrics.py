from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import matplotlib.pyplot as plt
import numpy as np

def compute_aupro(gt, pred, thresholds=np.linspace(0, 1, 21)):
    """
    Compute AUPRO (Area Under Per-Region Overlap) for a single class.

    Args:
        gt (ndarray): Binary ground truth mask (H, W)
        pred (ndarray): Predicted score map (H, W)
        thresholds (ndarray): List of thresholds to evaluate PRO

    Returns:
        float: AUPRO score
    """
    gt = (gt >= 0.5).astype(np.uint8)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)  # normalize

    labeled = label(gt)
    regions = regionprops(labeled)

    if len(regions) == 0:
        return np.nan  # no ground truth regions to evaluate

    pro_curve = []

    for t in thresholds:
        bin_pred = (pred >= t).astype(np.uint8)
        ious = []

        for region in regions:
            mask = (labeled == region.label).astype(np.uint8)
            intersection = np.logical_and(mask, bin_pred).sum()
            union = mask.sum()
            pro = intersection / union if union > 0 else 0
            ious.append(pro)

        pro_curve.append(np.mean(ious))

    aupro = np.trapz(pro_curve, thresholds)
    return aupro




def compute_segmentation_metrics(preds, gts, threshold=0.5):
    preds_np = preds.squeeze(0).detach().cpu().numpy()  # (C, H, W)
    gts_np = gts.squeeze(0).detach().cpu().numpy()

    n_classes = preds_np.shape[0]

    auroc_list = []
    f1_list = []
    ap_list = []
    aupro_list = []

    for c in range(n_classes):
        # if c == 0: continue

        pred_c = preds_np[c].flatten()
        gt_c = gts_np[c].flatten()
        gt_bin = (gt_c >= 0.5).astype(np.uint8)

        if len(np.unique(gt_bin)) < 2:
            auroc_list.append(np.nan)
            f1_list.append(np.nan)
            ap_list.append(np.nan)
            aupro_list.append(np.nan)
            continue

        try:
            auroc = roc_auc_score(gt_bin, pred_c)
        except ValueError:
            auroc = np.nan

        pred_bin = (pred_c >= threshold).astype(np.uint8)
        f1 = f1_score(gt_bin, pred_bin)
        ap = average_precision_score(gt_bin, pred_c)
        aupro = compute_aupro(gt_bin.reshape(preds_np[c].shape), preds_np[c])

        auroc_list.append(auroc)
        f1_list.append(f1)
        ap_list.append(ap)
        aupro_list.append(aupro)

    return {
        'auroc': auroc_list,
        'f1': f1_list,
        'ap': ap_list,
        'aupro': aupro_list
    }




def average_per_class_metric(metric_list):
    """
    Compute average score per class (per index) across multiple samples,
    skipping NaNs.

    Args:
        metric_list (List[List[float]]): e.g. [[nan, 0.8, nan], [0.5, 0.9, 0.3], ...]

    Returns:
        avg_scores (List[float]): Mean score per class/index
        counts (List[int]): Number of valid values used per class
    """
    arr = np.array(metric_list, dtype=np.float32)  # shape: [N_samples, N_classes]
    valid_mask = ~np.isnan(arr)
    sums = np.nansum(arr, axis=0)
    counts = np.sum(valid_mask, axis=0)
    avg_scores = (sums / np.maximum(counts, 1)).tolist()  # avoid divide-by-zero
    return avg_scores, counts.tolist()





def plot_metric_per_class(metric_values, counts, metric_name="AUROC", class_names=None, weighted_value=None, save_path=None):
    """
    Plots a bar chart for the metric per class with optional weighted average.

    Args:
        metric_values (list): Metric scores per class (NaN supported).
        counts (list): Number of valid samples used per class.
        metric_name (str): "AUROC", "F1", "AP", or "AUPRO"
        class_names (list): Optional list of class names.
        weighted_value (float): Optional weighted average to plot as a dashed line.
        save_path (str): Optional file path to save the plot.
    """
    metric_values = np.array(metric_values)
    num_classes = len(metric_values)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Handle NaNs for plotting
    plot_vals = np.nan_to_num(metric_values, nan=0)
    valid_flags = ~np.isnan(metric_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, plot_vals, color='skyblue', edgecolor='black')

    # Overlay NaN bars in red
    for idx, valid in enumerate(valid_flags):
        if not valid:
            bars[idx].set_color('lightcoral')
            ax.text(idx, 0.02, "NaN", ha='center', va='bottom', fontsize=8, color='black')

    # Annotate counts
    for idx, (val, count) in enumerate(zip(plot_vals, counts)):
        ax.text(idx, val + 0.01, f"n={count}", ha='center', va='bottom', fontsize=8)

    # Weighted average line
    if weighted_value is not None:
        ax.axhline(weighted_value, color='orange', linestyle='--', linewidth=2, label=f"Weighted {metric_name}: {weighted_value:.4f}")
        ax.legend()

    ax.set_title(f"{metric_name} per Class", fontsize=14)
    ax.set_ylabel(metric_name)
    ax.set_ylim([0, 1.05])
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()



def plot_all_metrics_per_class(metrics_dict, class_names=None, save_path=None):
    """
    Plots AUROC, F1, AP, and AUPRO in a single figure with subplots.

    Args:
        metrics_dict (dict): {
            'AUROC': (avg_values, counts, weighted),
            'F1': (avg_values, counts, weighted),
            'AP': (avg_values, counts, weighted),
            'AUPRO': (avg_values, counts, weighted)
        }
        class_names (list): Optional list of class labels.
        save_path (str): Path to save the figure.
    """
    metric_names = list(metrics_dict.keys())
    num_metrics = len(metric_names)
    num_classes = len(metrics_dict[metric_names[0]][0])

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 2.5 * num_metrics), sharex=True)

    for idx, metric in enumerate(metric_names):
        ax = axes[idx] if num_metrics > 1 else axes

        values, counts, weighted = metrics_dict[metric]
        values = np.array(values)
        counts = np.array(counts)
        valid = ~np.isnan(values)
        plot_vals = np.nan_to_num(values, nan=0)

        bars = ax.bar(class_names, plot_vals, color='steelblue', edgecolor='black')
        for i, (val, count, valid_flag) in enumerate(zip(plot_vals, counts, valid)):
            if valid_flag:
                ax.text(i, val + 0.04, f"{val:.3f}", ha='center', va='bottom', fontsize=8,
                        color='black')  # metric score
            else:
                ax.text(i, 0.02, "NaN", ha='center', va='bottom', fontsize=8, color='black')

            # ax.text(i, val + 0.01, f"n={count}", ha='center', va='bottom', fontsize=7, color='gray')  # sample count

        ax.axhline(weighted, color='orange', linestyle='--', linewidth=2, label=f"Weighted {metric}: {weighted:.4f}")
        ax.set_ylim([0, 1.05])
        ax.set_ylabel(metric, fontsize=12)
        ax.legend()

    axes[-1].set_xticks(np.arange(num_classes))
    axes[-1].set_xticklabels(class_names, rotation=45)
    fig.suptitle("Evaluation Metrics per Class", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics plot to {save_path}")
    else:
        plt.show()