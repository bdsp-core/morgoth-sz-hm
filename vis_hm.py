import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pooch.tests.test_core import FakeSleep
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve
from sklearn.metrics import auc as auc_value
from pathlib import Path
from tqdm import tqdm
import shutil

# def calculate_positive_counts_at_thresholds(csv_dir, continuous_test_data_list,thresholds, score_column='score', positive_threshold_func=None, step=1):
#     """
#     Compute the average number of positives in CSV files at different thresholds
#
#     Parameters:
#     csv_dir: directory containing 20 CSV files
#     thresholds: list of thresholds
#     score_column: score column name
#     positive_threshold_func: function to determine positive, default is score >= threshold
#     """
#
#     if positive_threshold_func is None:
#         positive_threshold_func = lambda scores, thresh: scores >= thresh
#
#     # Get all CSV files
#     csv_files = [
#         file_path for file_path in glob.glob(os.path.join(csv_dir, "*.csv"))
#         if os.path.splitext(os.path.basename(file_path))[0] in continuous_test_data_list
#     ]
#
#     print(f"Found {len(csv_files)} CSV files")
#
#     positive_counts = []
#
#     for threshold in thresholds:
#         print(threshold)
#         file_positive_counts = []
#
#         # Iterate over all CSV files
#         for csv_file in csv_files:
#             try:
#                 df = pd.read_csv(csv_file)
#                 if score_column in df.columns:
#                     scores = df[score_column].values
#                     if step!=1:
#                         scores = scores[::step]
#                     # Count positives in this file
#                     positive_mask = positive_threshold_func(scores, threshold)
#                     positive_count = positive_mask.sum()
#                     data_length_hour=((len(scores) - 1)* step +10)/3600
#                     positive_count_per_hour = positive_count / data_length_hour
#                     file_positive_counts.append(positive_count_per_hour)
#                 else:
#                     print(f"Warning: column '{score_column}' not found in {csv_file}")
#             except Exception as e:
#                 print(f"Error reading file {csv_file}: {e}")
#
#         # Compute average
#         avg_positive = np.mean(file_positive_counts) if file_positive_counts else 0
#         positive_counts.append(avg_positive)
#
#     return positive_counts



def calculate_positive_counts_at_thresholds_vectorized(
        csv_dir,
        continuous_test_data_list,
        thresholds,
        positive_threshold_func=None,
        score_column='score',
        step=1
):
    """
    Compute the average number of positives across all CSV files at different thresholds.
    Uses one-time loading + numpy vectorization for speed.

    Parameters:
    csv_dir: directory containing CSV files
    continuous_test_data_list: list of file basenames to match (without .csv)
    thresholds: list of thresholds (numpy array)
    positive_threshold_func: function to determine positives (scores, thresholds) -> mask
    score_column: score column name
    step: stride for sparse sampling

    Returns:
    avg_positive_per_threshold: average positive count at each threshold (normalized per hour)
    """

    # Find all matching CSV files
    csv_files = [
        file_path for file_path in glob.glob(os.path.join(csv_dir, "*.csv"))
        if os.path.splitext(os.path.basename(file_path))[0] in continuous_test_data_list
    ]
    print(f"Found {len(csv_files)} CSV files")

    all_positive_counts = []  # Store positive counts per file across all thresholds

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)
        if score_column in df.columns:
            scores = df[score_column].values
            if step != 1:
                scores = scores[::step]
            data_length_hour = ((len(scores) - 1) * step + 10) / 3600

            # Vectorized computation
            scores = scores[:, np.newaxis]  # (N,1)

            if positive_threshold_func is not None:
                positive_mask = positive_threshold_func(scores, thresholds)
            else:
                positive_mask = scores >= thresholds  # (N,M)

            positive_count_per_threshold = positive_mask.sum(axis=0) / data_length_hour  # (M,)
            all_positive_counts.append(positive_count_per_threshold)
        else:
            print(f"Warning: column '{score_column}' not found in {csv_file}")


    if len(all_positive_counts) == 0:
        print("No valid data files.")
        return np.zeros_like(thresholds)

    all_positive_counts = np.array(all_positive_counts)  # shape = (F, M)
    avg_positive_per_threshold = np.mean(all_positive_counts, axis=0)  # (M,)

    return avg_positive_per_threshold



def extract_sensitivity_at_thresholds(sensitivity_csv, thresholds, prediction_column='prediction',
                                      true_label_column='true_label'):
    """
    Compute sensitivity values at different thresholds from a CSV file containing predictions and true labels.

    Parameters:
    sensitivity_csv: path to CSV file containing predictions and true labels
    thresholds: list of thresholds
    prediction_column: prediction column name
    true_label_column: true label column name
    """

    df = pd.read_csv(sensitivity_csv)

    y_true = df[true_label_column].values
    y_pred_scores = df[prediction_column].values

    sensitivities = []

    for threshold in thresholds:
        # Generate predicted labels based on threshold
        y_pred = (y_pred_scores >= threshold).astype(int)

        # Compute True Positives and actual Positives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)

        # Compute sensitivity = TP / (TP + FN) = TP / actual_positive
        if actual_positive > 0:
            sensitivity = tp / actual_positive
        else:
            sensitivity = 0.0

        sensitivities.append(sensitivity)

    return sensitivities


def process_single_model_group(base_continuous_dir, continuous_test_data_list,base_discrete_file_prefix, group_id,
                               model_indices, thresholds, score_column='class_1_prob',
                               prediction_column='class_1_prob', true_label_column='true',
                               positive_threshold_func=None,
                               post=False,
                               step=10):
    """
    Process a single model group and return a list of results.

    Parameters:
    group_id: group ID for renumbering (0, 1, 2)
    """

    all_results = []
    if post:
        model_indices=range(1)

    for idx, model_idx in enumerate(model_indices):
        # New model ID: group ID + original index in decimal

        new_model_id = f"{group_id}-{model_idx:02d}"

        if post:
            continuous_dir=base_continuous_dir
            discrete_file=base_discrete_file_prefix

        else:
            continuous_dir = f"{base_continuous_dir}{model_idx}"
            discrete_file = f"{base_discrete_file_prefix}{model_idx}.csv"

        print(f"Processing model {new_model_id} (original: group {group_id}, model {model_idx})...")

        # Build file paths


        # Check if files exist
        if not os.path.exists(continuous_dir):
            print(f"Warning: directory not found {continuous_dir}")
            continue
        if not os.path.exists(discrete_file):
            print(f"Warning: file not found {discrete_file}")
            continue

        # Compute positive counts
        positive_counts = calculate_positive_counts_at_thresholds_vectorized(
                                csv_dir=continuous_dir,
                                continuous_test_data_list=continuous_test_data_list,
                                thresholds=thresholds,
                                score_column=score_column,
                                positive_threshold_func=positive_threshold_func,
                                step=step
                            )

        print(f"x-axis fp per hour done for round {idx}")

        # Compute sensitivities
        sensitivities= extract_sensitivity_at_thresholds(
            discrete_file, thresholds, prediction_column, true_label_column
        )

        print(f"y-axis sensitivities done for round {idx}")

        # Compute AUC
        max_positive_count = max(positive_counts) if positive_counts.size > 0 else 1
        normalized_positive_counts = [count / max_positive_count for count in positive_counts]

        sorted_indices = np.argsort(normalized_positive_counts)
        sorted_x = np.array(normalized_positive_counts)[sorted_indices]
        sorted_y = np.array(sensitivities)[sorted_indices]

        mroc_auc = auc_value(sorted_x, sorted_y)

        # Compute ROC-AUC
        df = pd.read_csv(discrete_file)
        y_true_original = df[true_label_column].values
        y_scores = df[prediction_column].values
        y_true_binary = (y_true_original == 1).astype(int)
        roc_auc = roc_auc_score(y_true_binary, y_scores)

        # Save results
        all_results.append({
            'model_id': new_model_id,
            'group_id': group_id,
            'original_model_idx': model_idx,
            'positive_counts': positive_counts,
            'sensitivities': sensitivities,
            'mroc_auc': mroc_auc,
            'roc_auc': roc_auc,
            'max_positive_count': max_positive_count
        })

        print(f"Model {new_model_id}: mROC AUC = {mroc_auc:.4f}, ROC AUC = {roc_auc:.4f}")


    return all_results


def plot_combined_mroc_curves(all_combined_results, figure_path,group_colors):
    """
    Plot mROC curves for all combined model groups.
    """
    plt.figure(figsize=(16, 12))

    # Define color scheme for each group

    for result in all_combined_results:
        group_id = result['group_id']
        model_id = result['model_id']
        positive_counts = result['positive_counts']
        sensitivities = result['sensitivities']
        mroc_auc = result['mroc_auc']

        # Select color based on group ID and set transparency
        base_color = group_colors[group_id]
        alpha = 0.7

        plt.plot(positive_counts, sensitivities,
                 color=base_color, linewidth=2, alpha=alpha,
                 label=f'{model_id} (AUC={mroc_auc:.3f})')

    plt.xlabel('#FP/hour', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.title('Modified ROC Curves - All Model Groups Combined', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print('[****] plot_combined_mroc_curves done!')
    # plt.show()


def plot_combined_roc_curves(all_combined_results, figure_path,group_colors):
    """
    Plot ROC curves for all combined model groups.
    """
    plt.figure(figsize=(16, 12))

    for result in all_combined_results:
        group_id = result['group_id']
        model_id = result['model_id']
        roc_auc = result['roc_auc']

        # We need to recompute the ROC curve data here
        # For simplicity, we can save ROC curve data in the earlier function
        base_color = group_colors[group_id]
        alpha = 0.7

        # Note: actual fpr, tpr data is needed here; will be added below
        # plt.plot(fpr, tpr, color=base_color, linewidth=2, alpha=alpha,
        #          label=f'{group_names[group_id]}-{model_id} (AUC={roc_auc:.3f})')

    # Draw diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves - All Model Groups Combined', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    # plt.show()


def plot_group_auc_comparison(group_names,all_combined_results, figure_path):
    """
    Plot AUC comparison across groups.
    """
    # Separate data by group
    n_groups = len(group_names)
    groups_data = {i: [] for i in range(n_groups)}
    for result in all_combined_results:
        groups_data[result['group_id']].append(result)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # mROC AUC comparison
    x_positions = []
    mroc_aucs = []
    roc_aucs = []
    labels = []
    colors = []

    group_colors = ['blue', 'red', 'green']

    for group_id in range(3):
        group_results = groups_data[group_id]
        for result in group_results:
            x_positions.append(len(x_positions))
            mroc_aucs.append(result['mroc_auc'])
            roc_aucs.append(result['roc_auc'])
            labels.append(result['model_id'])
            colors.append(group_colors[group_id])

    # mROC AUC bar chart
    bars1 = ax1.bar(x_positions, mroc_aucs, alpha=0.7, color=colors)
    ax1.set_xlabel('Model ID', fontsize=12)
    ax1.set_ylabel('mROC AUC', fontsize=12)
    ax1.set_title('mROC AUC Comparison Across All Groups', fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add group separator lines
    group_boundaries = []
    current_pos = 0
    for group_id in range(3):
        group_size = len(groups_data[group_id])
        if group_id > 0:
            ax1.axvline(x=current_pos - 0.5, color='black', linestyle='--', alpha=0.5)
        current_pos += group_size
        group_boundaries.append(current_pos)

    # ROC AUC bar chart
    bars2 = ax2.bar(x_positions, roc_aucs, alpha=0.7, color=colors)
    ax2.set_xlabel('Model ID', fontsize=12)
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('ROC AUC Comparison Across All Groups', fontsize=14)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add group separator lines
    current_pos = 0
    for group_id in range(3):
        group_size = len(groups_data[group_id])
        if group_id > 0:
            ax2.axvline(x=current_pos - 0.5, color='black', linestyle='--', alpha=0.5)
        current_pos += group_size

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=group_colors[i], alpha=0.7, label=group_names[i])
                       for i in range(3)]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print('[****] plot_group_auc_comparison done!')
    # plt.show()


def plot_group_statistics(all_combined_results, figure_path,group_names):
    """
    Plot statistics comparison across groups.
    """
    # Compute statistics by group
    n_groups = len(group_names)
    groups_stats = {}

    for group_id in range(n_groups):
        group_results = [r for r in all_combined_results if r['group_id'] == group_id]
        if group_results:
            mroc_aucs = [r['mroc_auc'] for r in group_results]
            roc_aucs = [r['roc_auc'] for r in group_results]

            groups_stats[group_id] = {
                'name': group_names[group_id],
                'count': len(group_results),
                'mroc_mean': np.mean(mroc_aucs),
                'mroc_std': np.std(mroc_aucs),
                'mroc_max': np.max(mroc_aucs),
                'mroc_min': np.min(mroc_aucs),
                'roc_mean': np.mean(roc_aucs),
                'roc_std': np.std(roc_aucs),
                'roc_max': np.max(roc_aucs),
                'roc_min': np.min(roc_aucs),
            }

    # Draw statistics comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    group_ids = list(groups_stats.keys())
    group_labels = [groups_stats[gid]['name'] for gid in group_ids]
    colors = ['blue', 'red', 'green']

    # mROC AUC mean comparison
    mroc_means = [groups_stats[gid]['mroc_mean'] for gid in group_ids]
    mroc_stds = [groups_stats[gid]['mroc_std'] for gid in group_ids]

    ax1.bar(group_labels, mroc_means, yerr=mroc_stds, alpha=0.7, color=colors, capsize=5)
    ax1.set_ylabel('mROC AUC')
    ax1.set_title('mROC AUC Mean ± Std by Group')
    ax1.grid(True, alpha=0.3, axis='y')

    # ROC AUC mean comparison
    roc_means = [groups_stats[gid]['roc_mean'] for gid in group_ids]
    roc_stds = [groups_stats[gid]['roc_std'] for gid in group_ids]

    ax2.bar(group_labels, roc_means, yerr=roc_stds, alpha=0.7, color=colors, capsize=5)
    ax2.set_ylabel('ROC AUC')
    ax2.set_title('ROC AUC Mean ± Std by Group')
    ax2.grid(True, alpha=0.3, axis='y')

    # mROC AUC range comparison
    mroc_mins = [groups_stats[gid]['mroc_min'] for gid in group_ids]
    mroc_maxs = [groups_stats[gid]['mroc_max'] for gid in group_ids]
    mroc_ranges = np.array(mroc_maxs) - np.array(mroc_mins)

    ax3.bar(group_labels, mroc_ranges, bottom=mroc_mins, alpha=0.7, color=colors)
    ax3.set_ylabel('mROC AUC')
    ax3.set_title('mROC AUC Range by Group')
    ax3.grid(True, alpha=0.3, axis='y')

    # ROC AUC range comparison
    roc_mins = [groups_stats[gid]['roc_min'] for gid in group_ids]
    roc_maxs = [groups_stats[gid]['roc_max'] for gid in group_ids]
    roc_ranges = np.array(roc_maxs) - np.array(roc_mins)

    ax4.bar(group_labels, roc_ranges, bottom=roc_mins, alpha=0.7, color=colors)
    ax4.set_ylabel('ROC AUC')
    ax4.set_title('ROC AUC Range by Group')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print('[****] plot_group_statistics done!')

    return groups_stats



def plot_combined_auc_trend(all_combined_results, all_eROC_results, figure_path, group_colors):
    """
    Plot continuous trend line chart for mROC AUC, ROC AUC, and eROC AUC.
    Order: 0-00, ..., 0-19, 1-00, ..., 1-19, 2-00, ..., 2-19
    """
    n_groups = len(group_colors)
    def sort_key(result):
        return (result['group_id'], result['original_model_idx'])

    sorted_results = sorted(all_combined_results, key=sort_key)
    sorted_eroc_results = sorted(all_eROC_results, key=sort_key)

    # Extract data
    model_ids = [r['model_id'] for r in sorted_results]
    mroc_aucs = [r['mroc_auc'] for r in sorted_results]
    roc_aucs = [r['roc_auc'] for r in sorted_results]
    eroc_aucs = [r['eroc_auc'] for r in sorted_eroc_results]
    x_positions = list(range(len(sorted_results)))

    plt.figure(figsize=(4*n_groups, 8))

    # mROC AUC curve
    plt.plot(x_positions, mroc_aucs,
             marker='o', linestyle='-', linewidth=2.5, markersize=6,
             color='blue', alpha=0.8, markerfacecolor='lightblue', markeredgecolor='darkblue',
             label='mROC AUC')

    # ROC AUC curve
    plt.plot(x_positions, roc_aucs,
             marker='s', linestyle='-', linewidth=2.5, markersize=6,
             color='red', alpha=0.8, markerfacecolor='lightcoral', markeredgecolor='darkred',
             label='ROC AUC')

    # eROC AUC curve
    plt.plot(x_positions, eroc_aucs,
             marker='^', linestyle='-', linewidth=2.5, markersize=6,
             color='green', alpha=0.8, markerfacecolor='lightgreen', markeredgecolor='darkgreen',
             label='eROC AUC')

    # Mean lines
    mean_mroc = np.mean(mroc_aucs)
    mean_roc = np.mean(roc_aucs)
    mean_eroc = np.mean(eroc_aucs)

    plt.axhline(y=mean_mroc, color='blue', linestyle='--', alpha=0.6, linewidth=2,
                label=f'Mean mROC AUC = {mean_mroc:.3f}')
    plt.axhline(y=mean_roc, color='red', linestyle='--', alpha=0.6, linewidth=2,
                label=f'Mean ROC AUC = {mean_roc:.3f}')
    plt.axhline(y=mean_eroc, color='green', linestyle='--', alpha=0.6, linewidth=2,
                label=f'Mean eROC AUC = {mean_eroc:.3f}')

    # Background color separated by group
    current_pos = 0
    for group_id in range(n_groups):
        group_size = len([r for r in sorted_results if r['group_id'] == group_id])
        if group_size > 0:
            plt.axvspan(current_pos, current_pos + group_size - 1,
                        alpha=0.08, color=group_colors[group_id], label=f'Round {group_id} Region')
        if group_id > 0:
            plt.axvline(x=current_pos - 0.5, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        current_pos += group_size

    # Annotate best points
    mroc_max_idx = np.argmax(mroc_aucs)
    roc_max_idx = np.argmax(roc_aucs)
    eroc_max_idx = np.argmax(eroc_aucs)

    plt.annotate(f'Best mROC\n{model_ids[mroc_max_idx]}\n{mroc_aucs[mroc_max_idx]:.3f}',
                 (x_positions[mroc_max_idx], mroc_aucs[mroc_max_idx]),
                 textcoords="offset points", xytext=(0, 0), ha='center',
                 fontsize=12, color='blue', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.6))

    plt.annotate(f'Best ROC\n{model_ids[roc_max_idx]}\n{roc_aucs[roc_max_idx]:.3f}',
                 (x_positions[roc_max_idx], roc_aucs[roc_max_idx]),
                 textcoords="offset points", xytext=(0, 0), ha='center',
                 fontsize=12, color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.6))

    plt.annotate(f'Best eROC\n{model_ids[eroc_max_idx]}\n{eroc_aucs[eroc_max_idx]:.3f}',
                 (x_positions[eroc_max_idx], eroc_aucs[eroc_max_idx]),
                 textcoords="offset points", xytext=(0, 0), ha='center',
                 fontsize=12, color='green', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.6))

    # Figure properties
    plt.xlabel('Model Sequence (Round-Checkpoint)', fontsize=18)
    plt.ylabel('AUC Value', fontsize=18)
    plt.title('AUC Trend: mROC, ROC, and eROC Across All Models', fontsize=24,fontweight='bold')
    plt.grid(True, alpha=0.3)

    # x-axis tick settings
    if len(model_ids) > 30:
        tick_positions = x_positions[::5]
        tick_labels = [model_ids[i] for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=14)
    else:
        plt.xticks(x_positions, model_ids, rotation=90, fontsize=14)

    plt.yticks(fontsize=12)
    # Legend
    plt.legend(loc='lower right', fontsize=18, frameon=False)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # Return statistics

    trend_stats = {
        'total_models': len(sorted_results),
        'best_mroc_model': model_ids[mroc_max_idx],
        'best_mroc_value': mroc_aucs[mroc_max_idx],
        'worst_mroc_model': model_ids[np.argmin(mroc_aucs)],
        'worst_mroc_value': min(mroc_aucs),
        'best_roc_model': model_ids[roc_max_idx],
        'best_roc_value': roc_aucs[roc_max_idx],
        'worst_roc_model': model_ids[np.argmin(roc_aucs)],
        'worst_roc_value': min(roc_aucs),
        'best_eroc_model': model_ids[eroc_max_idx],
        'best_eroc_value': eroc_aucs[eroc_max_idx],
        'worst_eroc_model': model_ids[np.argmin(eroc_aucs)],
        'worst_eroc_value': min(eroc_aucs),
        'mean_mroc': mean_mroc,
        'mean_roc': mean_roc,
        'mean_eroc': mean_eroc,
        'mroc_range': max(mroc_aucs) - min(mroc_aucs),
        'roc_range': max(roc_aucs) - min(roc_aucs),
        'eroc_range': max(eroc_aucs) - min(eroc_aucs),
    }

    print('[****] plot_combined_auc_trend done!')
    return trend_stats


def save_combined_results_to_csv(all_combined_results, csv_path, thresholds):
    """
    Save combined results to a CSV file.
    """
    # Save detailed results
    detailed_data = {'threshold': thresholds}

    for result in all_combined_results:
        model_id = result['model_id']
        detailed_data[f'{model_id}_positive_count'] = result['positive_counts']
        detailed_data[f'{model_id}_sensitivity'] = result['sensitivities']

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(csv_path.replace('.csv', '_detailed.csv'), index=False)

    # Save summary results
    summary_data = []
    for result in all_combined_results:
        summary_data.append({
            'model_id': result['model_id'],
            'group_id': result['group_id'],
            'original_model_idx': result['original_model_idx'],
            'mroc_auc': result['mroc_auc'],
            'roc_auc': result['roc_auc'],
            'max_positive_count': result['max_positive_count']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_path.replace('.csv', '_summary.csv'), index=False)

    print(f"Detailed results saved to: {csv_path.replace('.csv', '_detailed.csv')}")
    print(f"Summary results saved to: {csv_path.replace('.csv', '_summary.csv')}")

    return detailed_df, summary_df


def plot_best_models_mroc_curves(all_combined_results, figure_path,group_names,group_colors,group_markers,group_linestyles):
    """
    Plot mROC curves for the best model from each group (by mROC AUC).
    Best model curves from all groups are drawn on the same figure.

    Parameters:
    all_combined_results: list of results for all models
    figure_path: path to save the figure
    """

    # Separate data by group
    n_groups=len(group_names)
    groups_data = {i: [] for i in range(n_groups)}
    for result in all_combined_results:
        groups_data[result['group_id']].append(result)


    plt.figure(figsize=(14, 10))

    best_models_info = []

    # Find the best model for each group and plot its curve
    for group_id in range(len(group_names)):
        group_results = groups_data[group_id]

        if not group_results:
            print(f"Warning: group {group_id} ({group_names[group_id]}) has no data")
            continue

        # Find the model with the highest mROC AUC in this group
        best_model = max(group_results, key=lambda x: x['mroc_auc'])

        # Extract best model data
        model_id = best_model['model_id']
        positive_counts = best_model['positive_counts']
        sensitivities = best_model['sensitivities']
        mroc_auc = best_model['mroc_auc']
        roc_auc = best_model['roc_auc']

        # Plot mROC curve for this group's best model
        plt.plot(positive_counts, sensitivities,
                 color=group_colors[group_id],
                 linewidth=3,
                 alpha=0.9,
                 marker=group_markers[group_id],
                 markersize=6,
                 linestyle=group_linestyles[group_id],
                 markerfacecolor='white',
                 markeredgecolor=group_colors[group_id],
                 markeredgewidth=2,
                 label=f'Round {group_id} Best: {model_id} (mAUC={mroc_auc:.3f})')

        # Save best model info
        best_models_info.append({
            'group_name': group_names[group_id],
            'model_id': model_id,
            'mroc_auc': mroc_auc,
            'roc_auc': roc_auc,
            'original_model_idx': best_model['original_model_idx']
        })

        print(f"Best model in group {group_names[group_id]}: {model_id} (mROC AUC: {mroc_auc:.4f}, ROC AUC: {roc_auc:.4f})")

    # Set chart properties
    plt.xlabel('#FP/hour', fontsize=14, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=14, fontweight='bold')
    plt.title('Best Model mROC Curves from Each Group', fontsize=16, fontweight='bold', pad=20)

    # Set grid
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Set legend
    plt.legend(loc='lower right', fontsize=12, frameon=True,
               fancybox=True, shadow=True, framealpha=0.9)

    # Set axis ranges and ticks
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    # Beautify axes
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add statistics text box
    stats_text = "Best Models Summary:\n"
    for info in best_models_info:
        stats_text += f"{info['group_name']}: {info['model_id']} (mAUC={info['mroc_auc']:.3f})\n"

    # Add statistics info in the top-right corner
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save figure
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()

    return best_models_info


def plot_best_models_comparison(all_combined_results, figure_path,group_names):
    """
    Plot a detailed comparison of the best model from each group.
    Includes mROC curves, AUC comparison bar charts, etc.
    """

    # Separate data by group and find the best model
    best_models = []

    for group_id in range(3):
        group_results = [r for r in all_combined_results if r['group_id'] == group_id]
        if group_results:
            best_model = max(group_results, key=lambda x: x['mroc_auc'])
            best_models.append(best_model)

    if len(best_models) != 3:
        print("Warning: best model not found for some groups")
        return None

    # Create subplots
    fig = plt.figure(figsize=(18, 12))

    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Main plot: mROC curves
    ax_main = fig.add_subplot(gs[0, :])

    group_colors = ['blue', 'red', 'green']
    group_markers = ['o', 's', '^']
    group_linestyles = ['-', '--', '-.']

    for i, best_model in enumerate(best_models):
        positive_counts = best_model['positive_counts']
        sensitivities = best_model['sensitivities']
        mroc_auc = best_model['mroc_auc']
        model_id = best_model['model_id']

        ax_main.plot(positive_counts, sensitivities,
                     color=group_colors[i],
                     linewidth=3,
                     alpha=0.9,
                     marker=group_markers[i],
                     markersize=8,
                     linestyle=group_linestyles[i],
                     markerfacecolor='white',
                     markeredgecolor=group_colors[i],
                     markeredgewidth=2,
                     label=f'Round {i}: {model_id} (mAUC={mroc_auc:.3f})')

    ax_main.set_xlabel('#FP/hour', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Sensitivity', fontsize=14, fontweight='bold')
    ax_main.set_title('Best Model mROC Curves Comparison', fontsize=16, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(fontsize=12)
    ax_main.set_xlim(left=0)
    ax_main.set_ylim(0, 1.05)

    # Subplot 1: mROC AUC comparison
    ax1 = fig.add_subplot(gs[1, 0])
    mroc_aucs = [model['mroc_auc'] for model in best_models]
    bars1 = ax1.bar(group_names, mroc_aucs, color=group_colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('mROC AUC', fontweight='bold')
    ax1.set_ylim(0.5, 1)
    ax1.set_title('mROC AUC Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate values on bar chart
    for i, (bar, auc) in enumerate(zip(bars1, mroc_aucs)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 2: ROC AUC comparison
    ax2 = fig.add_subplot(gs[1, 1])
    roc_aucs = [model['roc_auc'] for model in best_models]
    bars2 = ax2.bar(group_names, roc_aucs, color=group_colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('ROC AUC', fontweight='bold')
    ax2.set_ylim(0.5, 1)
    ax2.set_title('ROC AUC Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate values on bar chart
    for i, (bar, auc) in enumerate(zip(bars2, roc_aucs)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 3: model info table
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')

    # Create table data
    table_data = []
    for i, model in enumerate(best_models):
        table_data.append([
            model['model_id'],
            f"{model['mroc_auc']:.3f}",
            f"{model['roc_auc']:.3f}"
        ])

    table = ax3.table(cellText=table_data,
                      colLabels=['Model ID', 'mROC AUC', 'ROC AUC'],
                      cellLoc='center',
                      loc='center',
                      colColours=['lightgray'] * 4)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax3.set_title('Best Models Summary', fontweight='bold', pad=20)

    # Save figure
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()

    return best_models


def add_best_models_plotting(group_names,smp_names,all_combined_results,smp_combined_results, ROC_results_prefix,all_eROC_results,smp_eROC_results,eROC_results_prefix,all_eROC2_results,
        smp_eROC2_results,output_dir,group_colors,group_markers,group_linestyles):
    """
    Add best-model plotting to the main program.
    """
    print(f"\n{'=' * 50}")
    print("Plotting best model comparison for each group...")
    print(f"{'=' * 50}")

    # Plot best model mROC curves
    best_models_info = plot_best_models_mroc_curves(
        all_combined_results,
        f"{output_dir}/best_models_mroc_curves.png",
        group_names, group_colors, group_markers, group_linestyles

    )

    # Plot detailed comparison
    best_models_details = plot_best_models_comparison_zoomin_all2(
        group_names=group_names,
        smp_names=smp_names,
        group_colors=group_colors,
        group_markers=group_markers,
        group_linestyles=group_linestyles,
        ROC_results_prefix=ROC_results_prefix,
        all_combined_results=all_combined_results,
        smp_combined_results=smp_combined_results,
        all_eROC_results=all_eROC_results,
        smp_eROC_results=smp_eROC_results,
        all_eROC2_results=all_eROC2_results,
        smp_eROC2_results=smp_eROC2_results,
        eROC_results_prefix=eROC_results_prefix,
        figure_path=f"{output_dir}/best_models_detailed_comparison.png"
    )

    # Print best model info
    print(f"\nBest model summary per group:")
    print("-" * 80)
    print(f"{'Group':<10} {'Model ID':<10} {'mROC AUC':<12} {'ROC AUC':<12} {'Orig idx':<10}")
    print("-" * 80)

    for info in best_models_info:
        print(
            f"{info['group_name']:<10} {info['model_id']:<10} {info['mroc_auc']:<12.4f} {info['roc_auc']:<12.4f} {info['original_model_idx']:<10}")

    print('[****] add add_best_models_plotting done!')

    return best_models_info, best_models_details





def plot_best_models_comparison_zoomin_all2(group_names, smp_names, group_colors,
                                            group_markers, group_linestyles, ROC_results_prefix, all_combined_results, smp_combined_results,
                                            all_eROC_results, smp_eROC_results, all_eROC2_results, smp_eROC2_results,
                                            eROC_results_prefix, figure_path):
    """
    Plot detailed comparison of the best model from each group:
    Row 1: full mROC curve, zoomed mROC curve (0-10 FP/h), mROC AUC bar chart, ROC AUC curve
    Row 2: PR AUC curve, ROC AUC bar chart, eROC1 AUC curve, ePR1 AUC curve
    Row 3: eROC1 AUC bar chart, eROC2 AUC curve, ePR2 AUC curve, eROC2 AUC bar chart
    """

    n_groups = len(group_names)
    print(f'Total {n_groups} model results')

    # Get best models
    best_models = []
    for group_id in range(n_groups):
        group_results = [r for r in all_combined_results if r['group_id'] == group_id]
        if group_results:
            best_model = max(group_results, key=lambda x: x['mroc_auc'])
            best_models.append(best_model)

    # Get eROC1 AUC for best models
    best_eroc_aucs = []
    for model in best_models:
        match = next((r for r in all_eROC_results if r['model_id'] == model['model_id']), None)
        best_eroc_aucs.append(match['eroc_auc'] if match else None)

    # Get eROC2 AUC for best models
    best_eroc2_aucs = []
    for model in best_models:
        match = next((r for r in all_eROC2_results if r['model_id'] == model['model_id']), None)
        best_eroc2_aucs.append(match['eroc_auc'] if match else None)

    if len(best_models) != n_groups:
        print("Warning: best model not found for some groups")
        return None

    # Prepare colors and markers for SMP results
    smp_colors = group_colors[:2]+[group_colors[-1]]
    round_colors= group_colors[2:-1]
    smp_linestyles = group_linestyles[-3:]

    # Configure figure layout: 3 rows x 4 columns
    fig = plt.figure(figsize=(40, 22.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.25)

    # Column 1: mROC-related
    ax_mroc_full = fig.add_subplot(gs[0, 0])
    ax_mroc_zoom = fig.add_subplot(gs[1, 0])
    ax_mroc_bar = fig.add_subplot(gs[2, 0])

    # Column 2: ROC-related
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_pr = fig.add_subplot(gs[1, 1])
    ax_roc_bar = fig.add_subplot(gs[2, 1])

    # Column 3: eROC1-related
    ax_eroc1 = fig.add_subplot(gs[0, 2])
    ax_epr1 = fig.add_subplot(gs[1, 2])
    ax_eroc1_bar = fig.add_subplot(gs[2, 2])

    # Column 4: eROC2-related
    ax_eroc2 = fig.add_subplot(gs[0, 3])
    ax_epr2 = fig.add_subplot(gs[1, 3])
    ax_eroc2_bar = fig.add_subplot(gs[2, 3])

    # Prepare bar chart data (in order: first two SMP + best_models + last SMP)
    plot_names = []
    plot_mroc_aucs = []
    plot_roc_aucs = []
    plot_eroc_aucs = []
    plot_eroc2_aucs = []
    plot_colors = []

    for item in smp_combined_results:
        item['group_id'] = int(item['group_id'])
    smp_combined_results.sort(key=lambda x: x['group_id'])

    # Add first two SMP models
    for i in range(min(2, len(smp_names))):
        if i < len(smp_combined_results):
            plot_names.append(smp_names[i])
            plot_mroc_aucs.append(smp_combined_results[i]['mroc_auc'])
            plot_roc_aucs.append(smp_combined_results[i]['roc_auc'])

            smp_eroc_match = next((r for r in smp_eROC_results if r['model_id'] == smp_combined_results[i]['model_id']),
                                  None)
            plot_eroc_aucs.append(smp_eroc_match['eroc_auc'] if smp_eroc_match else 0)

            # All SMPs have eROC2 data, retrieved from mp_eROC2_results
            smp_eroc2_match = next(
                (r for r in smp_eROC2_results if r['model_id'] == smp_combined_results[i]['model_id']), None)
            plot_eroc2_aucs.append(smp_eroc2_match['eroc_auc'] if smp_eroc2_match else 0)

            plot_colors.append(smp_colors[i])

    # Add best_models
    for i, model in enumerate(best_models):
        plot_names.append(group_names[i])
        plot_mroc_aucs.append(model['mroc_auc'])
        plot_roc_aucs.append(model['roc_auc'])
        plot_eroc_aucs.append(best_eroc_aucs[i])
        plot_eroc2_aucs.append(best_eroc2_aucs[i])
        plot_colors.append(round_colors[i])

    # Add last SMP model
    if len(smp_names) > 2 and len(smp_combined_results) > 2:
        last_idx = len(smp_names) - 1
        plot_names.append(smp_names[last_idx])
        plot_mroc_aucs.append(smp_combined_results[last_idx]['mroc_auc'])
        plot_roc_aucs.append(smp_combined_results[last_idx]['roc_auc'])

        smp_eroc_match = next(
            (r for r in smp_eROC_results if r['model_id'] == smp_combined_results[last_idx]['model_id']), None)
        plot_eroc_aucs.append(smp_eroc_match['eroc_auc'] if smp_eroc_match else 0)

        # The last SMP also has eROC2 data, retrieved from mp_eROC2_results
        smp_eroc2_match = next(
            (r for r in smp_eROC2_results if r['model_id'] == smp_combined_results[last_idx]['model_id']), None)
        plot_eroc2_aucs.append(smp_eroc2_match['eroc_auc'] if smp_eroc2_match else 0)

        plot_colors.append(smp_colors[last_idx])

    # Plot best models
    for i, model in enumerate(best_models):
        pos_counts = model['positive_counts']
        pos_counts_zoom = np.array(pos_counts)
        sens = model['sensitivities']
        model_id = model['model_id']
        mroc_auc = model['mroc_auc']

        # Full mROC plot
        # ax_mroc_full.plot(pos_counts, sens, label=f'Round {i}: {model_id} (AUC={mroc_auc:.3f})',
        #                   color=round_colors[i], linewidth=3, linestyle=group_linestyles[i])


        # Zoomed mROC plot
        # ax_mroc_zoom.plot(pos_counts_zoom, sens, label=f'Round {i}: {model_id} (AUC={mroc_auc:.3f})',
        #                   color=round_colors[i], linewidth=3, linestyle=group_linestyles[i])


        # Convert to 12-hour FP ----------------------------------------
        x_full = np.asarray(pos_counts, dtype=float) * 12
        x_zoom = np.asarray(pos_counts_zoom, dtype=float) * 12

        # Plot (ensure sens dimension matches x)
        ax_mroc_full.plot(
            x_full, sens,
            label=f'Round {i}: {model_id} (AUC={mroc_auc:.3f})',
            color=round_colors[i], linewidth=3, linestyle=group_linestyles[i]
        )

        ax_mroc_zoom.plot(
            x_zoom, sens,  # or sens_zoom (same length as x_zoom)
            label=f'Round {i}: {model_id} (AUC={mroc_auc:.3f})',
            color=round_colors[i], linewidth=3, linestyle=group_linestyles[i]
        )
        # Convert to 12-hour FP ----------------------------------------


        # Find y values corresponding to x=1 and x=5
        pos_counts = np.array(model['positive_counts'])  # Convert to numpy array
        sens = np.array(model['sensitivities'])
        for x_val in [1,5]:
            # Find the index closest to the target x value
            idx = np.argmin(np.abs(pos_counts - x_val))
            if idx < len(sens):
                y_val = sens[idx]

                # Add vertical line
                ax_mroc_zoom.axvline(x=x_val, color='gray', linestyle='--', alpha=0.7, linewidth=1)

                # # Add point marker and value annotation
                # ax_mroc_zoom.plot(x_val, y_val, 'o', color=round_colors[i], markersize=6)
                # ax_mroc_zoom.text(x_val, y_val + 0.03, f'{y_val:.3f}',
                #                   ha='center', va='bottom', fontsize=12,
                #                   color=round_colors[i], fontweight='bold')

        # ROC and PR (based on ROC_results_prefix)
        roc_file = f'{ROC_results_prefix}{group_names[i]}/pred_discrete_checkpoint{model["original_model_idx"]}.csv'
        try:
            df_roc = pd.read_csv(roc_file)
            y_true_roc = df_roc['true'].values
            y_true_roc = (y_true_roc == 1).astype(int)
            y_pred_roc = df_roc['class_1_prob'].values

            # ROC
            fpr_roc, tpr_roc, _ = roc_curve(y_true_roc, y_pred_roc)
            roc_auc = auc_value(fpr_roc, tpr_roc)
            ax_roc.plot(fpr_roc, tpr_roc,
                        label=f'Round {i}: {model_id} (AUC={roc_auc:.3f})',
                        color=round_colors[i],
                        linestyle=group_linestyles[i],
                        linewidth=3)

            # PR
            precision_roc, recall_roc, _ = precision_recall_curve(y_true_roc, y_pred_roc)
            pr_auc = auc_value(recall_roc, precision_roc)
            ax_pr.plot(recall_roc, precision_roc,
                       label=f'Round {i}: {model_id} (AUC={pr_auc:.3f})',
                       color=round_colors[i],
                       linestyle=group_linestyles[i],
                       linewidth=3)
        except:
            print(f"Warning: cannot read ROC file: {roc_file}")

        # eROC1 and ePR1
        eroc_file = f'{eROC_results_prefix}{group_names[i]}/pred_EEGlevel_checkpoint{model["original_model_idx"]}.csv'
        try:
            df = pd.read_csv(eroc_file)
            y_true = df['true_label'].values
            y_pred = df['pred_label'].values

            # eROC1
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ax_eroc1.plot(fpr, tpr,
                          label=f'Round {i}: {model_id} (AUC={best_eroc_aucs[i]:.3f})',
                          color=round_colors[i],
                          linestyle=group_linestyles[i],
                          linewidth=3)

            # ePR1
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc_value(recall, precision)
            ax_epr1.plot(recall, precision,
                         label=f'Round {i}: {model_id} (AUC={pr_auc:.3f})',
                         color=round_colors[i],
                         linestyle=group_linestyles[i],
                         linewidth=3)
        except:
            print(f"Warning: cannot read eROC1 file: {eroc_file}")

        # eROC2 and ePR2
        eroc2_file = f'{eROC_results_prefix}{group_names[i]}/pred_morgothEEGlevel_checkpoint{model["original_model_idx"]}.csv'
        try:
            df2 = pd.read_csv(eroc2_file)
            y_true2 = df2['true_label'].values
            y_pred2 = df2['pred_label'].values

            # eROC2
            fpr2, tpr2, _ = roc_curve(y_true2, y_pred2)
            ax_eroc2.plot(fpr2, tpr2,
                          label=f'Round {i}: {model_id} (AUC={best_eroc2_aucs[i]:.3f})',
                          color=round_colors[i],
                          linestyle=group_linestyles[i],
                          linewidth=3)

            # ePR2
            precision2, recall2, _ = precision_recall_curve(y_true2, y_pred2)
            pr_auc2 = auc_value(recall2, precision2)
            ax_epr2.plot(recall2, precision2,
                         label=f'Round {i}: {model_id} (AUC={pr_auc2:.3f})',
                         color=round_colors[i],
                         linestyle=group_linestyles[i],
                         linewidth=3)
        except:
            print(f"Warning: cannot read eROC2 file: {eroc2_file}")


    # Plot SMP results
    for i, smp_result in enumerate(smp_combined_results):
        pos_counts = smp_result['positive_counts']
        pos_counts_zoom = np.array(pos_counts)
        sens = smp_result['sensitivities']
        model_id = smp_result['model_id']
        mroc_auc = smp_result['mroc_auc']

        # Full mROC plot
        # ax_mroc_full.plot(pos_counts, sens, label=f'{smp_names[i]} (AUC={mroc_auc:.3f})',
        #                   color=smp_colors[i], linewidth=3, linestyle=smp_linestyles[i])



        # Zoomed mROC plot
        # ax_mroc_zoom.plot(pos_counts_zoom, sens, label=f'{smp_names[i]} (AUC={mroc_auc:.3f})',
        #                   color=smp_colors[i], linewidth=3, linestyle=smp_linestyles[i])


        # Convert to 12-hour FP ----------------------------------------
        # Full mROC plot
        x_full = np.asarray(pos_counts, dtype=float) * 12
        assert len(x_full) == len(sens), f"Dimension mismatch: {x_full.shape} vs {sens.shape}"
        ax_mroc_full.plot(
            x_full,
            sens,
            label=f'{smp_names[i]} (AUC={mroc_auc:.3f})',
            color=smp_colors[i],
            linewidth=3,
            linestyle=smp_linestyles[i]
        )

        # Zoomed mROC plot
        x_zoom = np.asarray(pos_counts_zoom, dtype=float) * 12
        assert len(x_zoom) == len(sens), f"Dimension mismatch: {x_zoom.shape} vs {sens.shape}"
        ax_mroc_zoom.plot(
            x_zoom,
            sens,
            label=f'{smp_names[i]} (AUC={mroc_auc:.3f})',
            color=smp_colors[i],
            linewidth=3,
            linestyle=smp_linestyles[i]
        )
        # Convert to 12-hour FP ----------------------------------------

        pos_counts = np.array(smp_result['positive_counts'])  # Convert to numpy array
        sens = np.array(smp_result['sensitivities'])

        def plot_points_on_curve(
                ax,
                pos_counts,
                sens,
                x_vals,
                *,
                annotate=True,
                markersize=6,
                y_offset=0.03,
                color=None,
                fmt='o',
                left=None,  # Left boundary value for extrapolation; defaults to leftmost sens
                right=None  # Right boundary value for extrapolation; defaults to rightmost sens
        ):
            """
            Annotate given x_vals as points on the curve y = sens(x=pos_counts).
            - Exact match: directly use (x_val, corresponding sens)
            - No exact match: linear interpolation (with boundary handling)
            - Automatically handles: duplicate pos_counts (average sens for same x), non-ascending order

            Parameters
            ----------
            ax : matplotlib Axes
            pos_counts : array-like
            sens : array-like
            x_vals : iterable of numbers
            annotate : bool, whether to annotate values above the point
            markersize : int
            y_offset : float, vertical offset for annotation text
            color : matplotlib color, or None to use default or outer loop color
            fmt : point style, default 'o'
            left, right : float or None, boundary values for np.interp

            Returns
            -------
            pts : list[(x, y)]  coordinates of plotted points
            """

            pos_counts = np.asarray(pos_counts, dtype=float)
            sens = np.asarray(sens, dtype=float)

            # 1) Handle duplicate pos_counts: average sens for same x to avoid interpolation ambiguity
            uniq_x, inv = np.unique(pos_counts, return_inverse=True)
            sens_mean = np.zeros_like(uniq_x, dtype=float)
            counts = np.zeros_like(uniq_x, dtype=int)
            for i, idx in enumerate(inv):
                sens_mean[idx] += sens[i]
                counts[idx] += 1
            sens_mean = sens_mean / np.maximum(counts, 1)

            # 2) Ensure x is ascending (np.unique already sorts ascending)
            xs = uniq_x
            ys = sens_mean

            # 3) Default boundary handling: use boundary y values when not specified
            if left is None:
                left = ys[0]
            if right is None:
                right = ys[-1]

            # 4) Build a lookup dict for exact matches
            exact_map = {float(x): float(y) for x, y in zip(xs, ys)}

            pts = []
            for x_val in x_vals:
                x_val = float(x_val)
                if x_val in exact_map:  # Exact match
                    y_val = exact_map[x_val]
                else:  # Linear interpolation (with boundaries)
                    y_val = float(np.interp(x_val, xs, ys, left=left, right=right))

                ax.plot(x_val, y_val, fmt, color=color, markersize=markersize)
                if annotate:
                    ax.text(
                        x_val, y_val + y_offset, f'{y_val:.3f}',
                        ha='center', va='bottom', fontsize=12,
                        color=color, fontweight='bold'
                    )
                pts.append((x_val, y_val))

            return pts

        # for x_val in [1, 5]:
        #
        #     ############ Use nearest approach ##########
        #     idx = np.argmin(np.abs(pos_counts - x_val))
        #     if idx < len(sens):
        #         y_val = sens[idx]
        #         ax_mroc_zoom.plot(x_val, y_val, 'o', color=smp_colors[i], markersize=6)
        #         ax_mroc_zoom.text(x_val, y_val + 0.03, f'{y_val:.3f}',
        #                           ha='center', va='bottom', fontsize=12,
        #                           color=smp_colors[i], fontweight='bold')
        #     ############ Use nearest approach ##########
        #
        #     # ############ Interpolation computation ##########
        #     # if x_val in pos_counts:
        #     #     idx = np.where(pos_counts == x_val)[0][0]
        #     #     y_val = sens[idx]
        #     # else:
        #     #     # Interpolation
        #     #     y_val = np.interp(x_val, pos_counts, sens)
        #     #
        #     # ax_mroc_zoom.plot(x_val, y_val, 'o', color=smp_colors[i], markersize=6)
        #     # ax_mroc_zoom.text(
        #     #     x_val, y_val + 0.03, f'{y_val:.3f}',
        #     #     ha='center', va='bottom', fontsize=12,
        #     #     color=smp_colors[i], fontweight='bold'
        #     # )
        #     # ############ Interpolation computation ##########
        #
        #     ############ Mark only existing values ##########
        #     # if x_val in pos_counts:
        #     #     idx = np.where(pos_counts == x_val)[0][0]
        #     #     y_val = sens[idx]
        #     #     ax_mroc_zoom.plot(x_val, y_val, 'o', color=smp_colors[i], markersize=6)
        #     #     ax_mroc_zoom.text(
        #     #         x_val, y_val + 0.03, f'{y_val:.3f}',
        #     #         ha='center', va='bottom', fontsize=12,
        #     #         color=smp_colors[i], fontweight='bold'
        #     #     )
        #     # else:
        #     #     # If this value does not exist, can skip or print a reminder
        #     #     print(f"Warning: pos_counts does not contain {x_val}")
        #     ############ Mark only existing values ##########

        # plot_points_on_curve(
        #     ax_mroc_zoom,
        #     pos_counts,
        #     sens,
        #     x_vals=[1, 5],
        #     color=smp_colors[i],
        #     annotate=True
        # )

        # SMP ROC and PR (based on ROC_results_prefix)
        smp_roc_file = f'{ROC_results_prefix}{smp_names[i]}/pred_discrete.csv'
        try:
            df_smp_roc = pd.read_csv(smp_roc_file)
            y_true_smp_roc = df_smp_roc['true'].values
            y_true_smp_roc = (y_true_smp_roc == 1).astype(int)
            y_pred_smp_roc = df_smp_roc['class_1_prob'].values

            # SMP ROC
            fpr_smp_roc, tpr_smp_roc, _ = roc_curve(y_true_smp_roc, y_pred_smp_roc)
            smp_roc_auc = auc_value(fpr_smp_roc, tpr_smp_roc)
            ax_roc.plot(fpr_smp_roc, tpr_smp_roc,
                        label=f'{smp_names[i]} (AUC={smp_roc_auc:.3f})',
                        color=smp_colors[i],
                        linestyle=smp_linestyles[i],
                        linewidth=3)

            # SMP PR
            precision_smp_roc, recall_smp_roc, _ = precision_recall_curve(y_true_smp_roc, y_pred_smp_roc)
            smp_pr_auc = auc_value(recall_smp_roc, precision_smp_roc)
            ax_pr.plot(recall_smp_roc, precision_smp_roc,
                       label=f'{smp_names[i]} (AUC={smp_pr_auc:.3f})',
                       color=smp_colors[i],
                       linestyle=smp_linestyles[i],
                       linewidth=3)

        except:
            print(f"Warning: cannot read SMP ROC file: {smp_roc_file}")

        # SMP eROC1 and ePR1
        smp_eroc_match = next((r for r in smp_eROC_results if r['model_id'] == model_id), None)
        if smp_eroc_match:
            eroc_auc = smp_eroc_match['eroc_auc']
            eroc_file = f'{eROC_results_prefix}{smp_names[i]}/pred_EEGlevel.csv'
            try:
                df = pd.read_csv(eroc_file)
                y_true = df['true_label'].values
                y_pred = df['pred_label'].values

                # eROC1
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                ax_eroc1.plot(fpr, tpr,
                              label=f'{smp_names[i]} (AUC={eroc_auc:.3f})',
                              color=smp_colors[i],
                              linestyle=smp_linestyles[i],
                              linewidth=3)

                # ePR1
                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                pr_auc = auc_value(recall, precision)
                ax_epr1.plot(recall, precision,
                             label=f'{smp_names[i]} (AUC={pr_auc:.3f})',
                             color=smp_colors[i],
                             linestyle=smp_linestyles[i],
                             linewidth=3)
            except:
                print(f"Warning: failed to read {eroc_file}")

        # SMP eROC2 and ePR2 - all three models have eROC2 data
        smp_eroc2_match = next((r for r in smp_eROC2_results if r['model_id'] == model_id), None)
        if smp_eroc2_match:
            eroc2_auc = smp_eroc2_match['eroc_auc']
            eroc2_file = f'{eROC_results_prefix}{smp_names[i]}/pred_morgothEEGlevel.csv'
            try:
                df2 = pd.read_csv(eroc2_file)
                y_true2 = df2['true_label'].values
                y_pred2 = df2['pred_label'].values

                # eROC2
                fpr2, tpr2, _ = roc_curve(y_true2, y_pred2)
                ax_eroc2.plot(fpr2, tpr2,
                              label=f'{smp_names[i]} (AUC={eroc2_auc:.3f})',
                              color=smp_colors[i],
                              linestyle=smp_linestyles[i],
                              linewidth=3)

                # ePR2
                precision2, recall2, _ = precision_recall_curve(y_true2, y_pred2)
                pr_auc2 = auc_value(recall2, precision2)
                ax_epr2.plot(recall2, precision2,
                             label=f'{smp_names[i]} (AUC={pr_auc2:.3f})',
                             color=smp_colors[i],
                             linestyle=smp_linestyles[i],
                             linewidth=3)
            except:
                print(f"Warning: failed to read SMP eROC2 file: {eroc2_file}")

    # Set figure properties
    # Column 1: mROC-related
    ax_mroc_full.set_title("mROC Curves", fontweight='bold', fontsize=24)
    ax_mroc_full.set_xlabel("#FP/12 hour", fontsize=20)
    ax_mroc_full.set_ylabel("Sensitivity", fontsize=20)
    ax_mroc_full.set_xlim(left=0)
    ax_mroc_full.set_ylim(0, 1.05)
    ax_mroc_full.grid(True, alpha=0.3)
    ax_mroc_full.legend(fontsize=16, frameon=False)
    ax_mroc_full.spines['top'].set_visible(False)
    ax_mroc_full.spines['right'].set_visible(False)
    ax_mroc_full.tick_params(labelsize=18)

    ax_mroc_zoom.set_title("mROC Curves (Zoomed to 0–10 FP/12h)", fontweight='bold', fontsize=24)
    ax_mroc_zoom.set_xlabel("#FP/12 hour", fontsize=20)
    ax_mroc_zoom.set_ylabel("Sensitivity", fontsize=20)
    ax_mroc_zoom.set_xlim(0, 11)
    ax_mroc_zoom.set_ylim(-0.05, 1.05)
    ax_mroc_zoom.grid(True, alpha=0.3)
    ax_mroc_zoom.spines['top'].set_visible(False)
    ax_mroc_zoom.spines['right'].set_visible(False)
    ax_mroc_zoom.tick_params(labelsize=18)

    # Column 2: ROC and PR
    ax_roc.set_title("ROC AUC", fontweight='bold', fontsize=24)
    ax_roc.set_xlabel('False Positive Rate', fontsize=20)
    ax_roc.set_ylabel('True Positive Rate', fontsize=20)
    ax_roc.legend(fontsize=16, frameon=False)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.tick_params(labelsize=18)

    ax_pr.set_title("PR AUC", fontweight='bold', fontsize=24)
    ax_pr.set_xlabel('Recall', fontsize=20)
    ax_pr.set_ylabel('Precision', fontsize=20)
    ax_pr.legend(fontsize=16, frameon=False)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.tick_params(labelsize=18)

    # Column 3: eROC1-related
    ax_eroc1.set_title("eROC1 AUC (Max probability in EEG)", fontweight='bold', fontsize=24)
    ax_eroc1.set_xlabel('False Positive Rate', fontsize=20)
    ax_eroc1.set_ylabel('True Positive Rate', fontsize=20)
    ax_eroc1.legend(fontsize=16, frameon=False)
    ax_eroc1.grid(True, alpha=0.3)
    ax_eroc1.spines['top'].set_visible(False)
    ax_eroc1.spines['right'].set_visible(False)
    ax_eroc1.tick_params(labelsize=18)

    ax_epr1.set_title("ePR1 AUC (Max probability in EEG)", fontweight='bold', fontsize=24)
    ax_epr1.set_xlabel('Recall', fontsize=20)
    ax_epr1.set_ylabel('Precision', fontsize=20)
    ax_epr1.legend(fontsize=16, frameon=False, loc='lower left')
    ax_epr1.grid(True, alpha=0.3)
    ax_epr1.spines['top'].set_visible(False)
    ax_epr1.spines['right'].set_visible(False)
    ax_epr1.tick_params(labelsize=18)

    # Column 4: eROC2-related
    ax_eroc2.set_title("eROC2 AUC (Morgoth1 EEG-level results)", fontweight='bold', fontsize=24)
    ax_eroc2.set_xlabel('False Positive Rate', fontsize=20)
    ax_eroc2.set_ylabel('True Positive Rate', fontsize=20)
    ax_eroc2.legend(fontsize=16, frameon=False)
    ax_eroc2.grid(True, alpha=0.3)
    ax_eroc2.spines['top'].set_visible(False)
    ax_eroc2.spines['right'].set_visible(False)
    ax_eroc2.tick_params(labelsize=18)

    ax_epr2.set_title("ePR2 AUC (Morgoth1 EEG-level results)", fontweight='bold', fontsize=24)
    ax_epr2.set_xlabel('Recall', fontsize=20)
    ax_epr2.set_ylabel('Precision', fontsize=20)
    ax_epr2.legend(fontsize=16, frameon=False)
    ax_epr2.grid(True, alpha=0.3)
    ax_epr2.spines['top'].set_visible(False)
    ax_epr2.spines['right'].set_visible(False)
    ax_epr2.tick_params(labelsize=18)

    # Draw bar charts
    new_labels = ['sparcnet', 'morgoth1'] + [f'round {i}' for i in range(len(round_colors))] + ['post']

    # mROC AUC bar chart
    bars_mroc = ax_mroc_bar.bar(plot_names, plot_mroc_aucs, color=plot_colors)
    ax_mroc_bar.set_title("mROC AUC", fontweight='bold', fontsize=24)
    ax_mroc_bar.set_ylim(0.6, 1.05)
    ax_mroc_bar.tick_params(axis='x', labelsize=18, rotation=90)
    ax_mroc_bar.tick_params(axis='y', labelsize=18)
    for bar, auc in zip(bars_mroc, plot_mroc_aucs):
        ax_mroc_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax_mroc_bar.set_xticklabels(new_labels)

    # ROC AUC bar chart
    bars_roc = ax_roc_bar.bar(plot_names, plot_roc_aucs, color=plot_colors)
    ax_roc_bar.set_title("ROC AUC", fontweight='bold', fontsize=24)
    ax_roc_bar.set_ylim(0.6, 1.05)
    ax_roc_bar.tick_params(axis='x', labelsize=18, rotation=90)
    ax_roc_bar.tick_params(axis='y', labelsize=18)
    for bar, auc in zip(bars_roc, plot_roc_aucs):
        ax_roc_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax_roc_bar.set_xticklabels(new_labels)

    # eROC1 AUC bar chart
    bars_eroc1 = ax_eroc1_bar.bar(plot_names, plot_eroc_aucs, color=plot_colors)
    ax_eroc1_bar.set_title("eROC1 AUC", fontweight='bold', fontsize=24)
    ax_eroc1_bar.set_ylim(0.6, 1.05)
    ax_eroc1_bar.tick_params(axis='x', labelsize=18, rotation=90)
    ax_eroc1_bar.tick_params(axis='y', labelsize=18)
    for bar, auc in zip(bars_eroc1, plot_eroc_aucs):
        ax_eroc1_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax_eroc1_bar.set_xticklabels(new_labels)

    # eROC2 AUC bar chart
    bars_eroc2 = ax_eroc2_bar.bar(plot_names, plot_eroc2_aucs, color=plot_colors)
    ax_eroc2_bar.set_title("eROC2 AUC", fontweight='bold', fontsize=24)
    ax_eroc2_bar.set_ylim(0.6, 1.05)
    ax_eroc2_bar.tick_params(axis='x', labelsize=18, rotation=90)
    ax_eroc2_bar.tick_params(axis='y', labelsize=18)
    for bar, auc in zip(bars_eroc2, plot_eroc2_aucs):
        ax_eroc2_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax_eroc2_bar.set_xticklabels(new_labels)

    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    return best_models

def load_combined_results_from_csv(summary_csv_path, detailed_csv_path):
    """
    Restore all_combined_results structure from saved summary and detailed CSV files.
    """
    # Read CSV files
    summary_df = pd.read_csv(summary_csv_path)
    detailed_df = pd.read_csv(detailed_csv_path)

    # Get shared thresholds
    thresholds = detailed_df['threshold'].tolist()

    # Extract all model_ids (from column names)
    model_ids = set(col.split('_')[0] for col in detailed_df.columns if '_positive_count' in col)

    all_combined_results = []
    for model_id in model_ids:
        # Get sensitivity / positive count columns for this model
        positive_counts = detailed_df[f'{model_id}_positive_count'].tolist()
        sensitivities = detailed_df[f'{model_id}_sensitivity'].tolist()

        # Get other fields from summary
        summary_row = summary_df[summary_df['model_id'] == model_id].iloc[0]

        # Build the recovered dictionary
        result = {
            'model_id': model_id,
            'group_id': int(summary_row['group_id']),
            'original_model_idx': int(summary_row['original_model_idx']),
            'mroc_auc': float(summary_row['mroc_auc']),
            'roc_auc': float(summary_row['roc_auc']),
            'max_positive_count': float(summary_row['max_positive_count']),
            'positive_counts': positive_counts,
            'sensitivities': sensitivities
        }

        all_combined_results.append(result)
    return all_combined_results, thresholds

# if one seizure occur, then seizure
def  make_EEGlevel_results(base_continuous_dir,base_discrete_file_prefix,continuous_test_data_list, n_models,output_dir,rewrite=True,post=False,step=10):
    if post:
        n_models=1
    else:
        os.makedirs(output_dir, exist_ok=True)

    for model_idx in range(n_models):
        # Save results
        if post:
            output_path = output_dir
        else:
            output_path = os.path.join(output_dir, f"pred_EEGlevel_checkpoint{model_idx}.csv")

        if os.path.exists(output_path) and rewrite==False:
            continue

        EEG_results_df = pd.DataFrame(columns=['file_name', 'true_label', 'pred_label'])

        if post:
            continuous_dir = base_continuous_dir
            discrete_file = base_discrete_file_prefix

        else:
            continuous_dir = f"{base_continuous_dir}{model_idx}/"
            discrete_file = f"{base_discrete_file_prefix}{model_idx}.csv"

        # 1. Read discrete file and update rows where true_label == 1
        discrete_df = pd.read_csv(discrete_file)
        for _, row in discrete_df.iterrows():
            if row['true'] == 1:
                file_name = row['data'].split('_')[0]
                pred_label = row['class_1_prob']

                existing_idx = EEG_results_df.index[EEG_results_df['file_name'] == file_name].tolist()

                if existing_idx:
                    # Entry already exists for this file_name; compare and update
                    idx = existing_idx[0]
                    if pred_label > EEG_results_df.at[idx, 'pred_label']:
                        EEG_results_df.at[idx, 'pred_label'] = pred_label
                        EEG_results_df.at[idx, 'true_label'] = 1
                else:
                    # Does not exist; add directly
                    EEG_results_df = pd.concat([EEG_results_df, pd.DataFrame([{
                        'file_name': file_name,
                        'true_label': 1,
                        'pred_label': pred_label
                    }])], ignore_index=True)

        # 2. Read all CSV files under continuous_dir
        csv_files = [
            file_path for file_path in glob.glob(os.path.join(continuous_dir, "*.csv"))
            if os.path.splitext(os.path.basename(file_path))[0] in continuous_test_data_list
        ]

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            file_name = os.path.splitext(os.path.basename(csv_file))[0]
            file_name = file_name.split('_')[0]
            if file_name in EEG_results_df['file_name'].values:
                continue

            class_1_probs = df['class_1_prob'].values
            class_1_probs = class_1_probs[::step]
            sorted_probs = sorted(class_1_probs, reverse=True)

            ################Modified####################################
            # pred_label = sorted_probs[0]
            if len(sorted_probs) >= 11:
                pred_label = sorted_probs[10]  # 11th largest (0-indexed)
            else:
                pred_label = sorted_probs[-1]  # Fewer than 11 values, take the smallest
            ################Modified####################################

            EEG_results_df = pd.concat([EEG_results_df, pd.DataFrame([{
                'file_name': file_name,
                'true_label': 0,
                'pred_label': pred_label
            }])], ignore_index=True)


        EEG_results_df.to_csv(output_path, index=False)
        print(f"Saved EEG-level results to {output_path}")


# based on morgoth eeg_level results
def  make_EEGlevel_results2(model_EEGlevel_file_prefix,base_discrete_file_prefix,continuous_test_data_list, n_models,output_dir,post=False):
    if post:
        n_models=1
    else:
        os.makedirs(output_dir, exist_ok=True)

    for model_idx in range(n_models):
        EEG_results_df = pd.DataFrame(columns=['file_name', 'true_label', 'pred_label'])

        if post:
            continuous_file = model_EEGlevel_file_prefix
            discrete_file = base_discrete_file_prefix

        else:
            continuous_file= f"{model_EEGlevel_file_prefix}{model_idx}/pred_EEG_level_SEIZURE.csv"
            discrete_file = f"{base_discrete_file_prefix}{model_idx}.csv"

        # 1. Read discrete file and update rows where true_label == 1
        discrete_df = pd.read_csv(discrete_file)
        for _, row in discrete_df.iterrows():
            if row['true'] == 1:
                file_name = row['data'].split('_')[0]
                pred_label = row['class_1_prob']

                existing_idx = EEG_results_df.index[EEG_results_df['file_name'] == file_name].tolist()

                if existing_idx:
                    # Entry already exists for this file_name; compare and update
                    idx = existing_idx[0]
                    if pred_label > EEG_results_df.at[idx, 'pred_label']:
                        EEG_results_df.at[idx, 'pred_label'] = pred_label
                        EEG_results_df.at[idx, 'true_label'] = 1
                else:
                    # Does not exist; add directly
                    EEG_results_df = pd.concat([EEG_results_df, pd.DataFrame([{
                        'file_name': file_name,
                        'true_label': 1,
                        'pred_label': pred_label
                    }])], ignore_index=True)

        # 2. Read all CSV files under continuous_dir
        continuous_df = pd.read_csv(continuous_file)
        for _, row in continuous_df.iterrows():
            file_name = row['file_name']
            file_name=file_name.replace('_sparcnet', '')
            patient_id = file_name.split('_')[0]

            if os.path.splitext(file_name)[0] in continuous_test_data_list:
                if patient_id in EEG_results_df['file_name'].values:
                    continue

                pred_label=row['probability']
                EEG_results_df = pd.concat([EEG_results_df, pd.DataFrame([{
                    'file_name': patient_id,
                    'true_label': 0,
                    'pred_label': pred_label
                }])], ignore_index=True)

        # Save results
        if post:
            output_path=output_dir
        else:
            output_path = os.path.join(output_dir, f"pred_morgothEEGlevel_checkpoint{model_idx}.csv")
        EEG_results_df.to_csv(output_path, index=False)
        print(f"Saved morgoth EEG-level results to {output_path}")


def eROC_for_each_model(base_EEGlevel_file_prefix, group_id, n_models,post=False,eroc=1):
    results = []
    if post:
        n_models=1
    for model_idx in range(n_models):
        if post:
            file_path=base_EEGlevel_file_prefix
        else:
            file_path = f"{base_EEGlevel_file_prefix}{model_idx}.csv"


        df = pd.read_csv(file_path)

        if 'true_label' not in df.columns or 'pred_label' not in df.columns:
            print(f"Missing columns in {file_path}, skipping.")
            continue

        y_true = df['true_label'].values
        y_pred = df['pred_label'].values

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = None  # e.g., if only one class in y_true

        new_model_id = f"{group_id}-{model_idx:02d}"

        results.append({
            'model_id': new_model_id,
            'group_id': group_id,
            'original_model_idx': model_idx,
            'eroc_auc': auc,
        })

    return results


def process_csv_files(last_model_continuous_result_dir, post_continuous_dir):
    """
    Process CSV files with post-processing smoothing.

    Args:
        last_model_continuous_result_dir: Input directory path
        post_continuous_dir: Output directory path
    """
    os.makedirs(post_continuous_dir, exist_ok=True)
    input_path = Path(last_model_continuous_result_dir)
    csv_files = list(input_path.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            print(f"Processing file: {csv_file.name}")
            df = pd.read_csv(csv_file)

            if 'class_1_prob' not in df.columns or 'pred_class' not in df.columns:
                print(f"Warning: {csv_file.name} is missing required columns")
                continue

            original_class_1_prob = df['class_1_prob'].values
            original_pred_class = df['pred_class'].values.copy()
            original_pred_class[original_pred_class != 1] = 0

            processed_class_1_prob = post_processing(
                y=original_class_1_prob,
                y_binary=original_pred_class
            )

            changed_rows = ~np.isclose(original_class_1_prob, processed_class_1_prob)

            df['class_1_prob'] = processed_class_1_prob

            for idx in np.where(changed_rows)[0]:
                if processed_class_1_prob[idx] > 0.5:
                    df.at[idx, 'pred_class'] = 1
                else:
                    df.at[idx, 'pred_class'] = 0
                if 'class_0_prob' in df.columns:
                    df.at[idx, 'class_0_prob'] = 1 - processed_class_1_prob[idx]
                # Other columns set to 0 except selected
                for col in df.columns:
                    if col not in ['class_0_prob', 'class_1_prob', 'pred_class']:
                        df.at[idx, col] = 0

            output_file = Path(post_continuous_dir) / csv_file.name
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue

    print("All files processed!")

def post_processing(y, y_binary, th1=2, th2=20):
    """
    Post-processing function for label smoothing with linear interpolation filling.

    Args:
        y: Original values (1D numpy array)
        y_binary: Binary mask (1D numpy array)
        th1: Threshold for merging gaps (number of samples)
        th2: Threshold for minimum duration to keep (number of samples)

    Returns:
        y: Smoothed labels with interpolation and short segment removal
    """
    # Find rising and falling edges
    diff_y = np.diff(np.concatenate([[0], y_binary]))
    idx_r = np.where(diff_y == 1)[0]
    diff_y_end = np.diff(np.concatenate([y_binary, [0]]))
    idx_f = np.where(diff_y_end == -1)[0]

    if len(idx_r) == 0 or len(idx_f) == 0:
        return y

    # Merge close segments using interpolation
    for i in range(len(idx_f) - 1):
        gap = idx_r[i + 1] - idx_f[i] - 1
        if gap <= th1:
            y_binary[idx_f[i]:idx_r[i + 1] + 1] = 1
            left_val = y[idx_f[i]]
            right_val = y[idx_r[i + 1]]
            interp_len = idx_r[i + 1] - idx_f[i] + 1
            y[idx_f[i]:idx_r[i + 1] + 1] = np.linspace(left_val, right_val, interp_len)

    # Recompute rising / falling edges after merging
    diff_y = np.diff(np.concatenate([[0], y_binary]))
    idx_r = np.where(diff_y == 1)[0]
    diff_y_end = np.diff(np.concatenate([y_binary, [0]]))
    idx_f = np.where(diff_y_end == -1)[0]

    # Remove short segments
    for i in range(len(idx_r)):
        dur = idx_f[i] - idx_r[i] + 1
        if dur <= th2:
            y_binary[idx_r[i]:idx_f[i] + 1] = 0
            y[idx_r[i]:idx_f[i] + 1] = 0

    return y

def sparcnet_res(continuous_results_dir,new_continuous_results_dir,discrete_results_dir,discrete_results_template,discrete_results_path):

    os.makedirs(new_continuous_results_dir, exist_ok=True)
    for file in tqdm(os.listdir(continuous_results_dir),desc='make pred_continuousnonsz'):
        if file.endswith('.csv'):
            file_path = os.path.join(continuous_results_dir, file)
            df = pd.read_csv(file_path, header=None)  # No column names

            # Assign column names to the 6 columns
            df.columns = ['class_0_prob', 'class_1_prob', 'class_2_prob', 'class_3_prob', 'class_4_prob',
                          'class_5_prob']

            # Add pred_class column: index of the class with highest probability
            df['pred_class'] = df.iloc[:, :6].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

            # Save file
            file_name = file.replace('_sparcnet.csv', '.csv')
            new_file_path= os.path.join(new_continuous_results_dir, file_name)
            df.to_csv(new_file_path, index=False)

    discrete_results_df = pd.read_csv(discrete_results_template)
    for file in os.listdir(discrete_results_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(discrete_results_dir, file)
            df = pd.read_csv(file_path, header=None)

            file_name = file.replace('_sparcnet.csv', '.mat')

            # Find matching row indices
            mask = discrete_results_df['data'] == file_name

            # Update probability columns
            prob_columns = ['class_0_prob', 'class_1_prob', 'class_2_prob', 'class_3_prob', 'class_4_prob',
                            'class_5_prob']
            discrete_results_df.loc[mask, prob_columns] = df.values

            # Update pred_class column
            discrete_results_df.loc[mask, 'pred_class'] = discrete_results_df.loc[mask, prob_columns].idxmax(
                axis=1).str.extract(r'(\d+)').astype(int)

    discrete_results_df.to_csv(discrete_results_path, index=False)



if __name__ == "__main__":
    # Define path names
    path_names = ["IIIC1", "IIIC1_1", "IIIC2", "IIIC3", "IIIC4", "IIIC5", "IIIC6", "IIIC7", "IIIC8","IIIC9", "IIIC10","IIIC11"]

    n_groups = len(path_names)

    cmap = plt.get_cmap('viridis')
    group_colors = [cmap(i / (n_groups - 1)) for i in range(n_groups)]
    group_colors_background = [cmap(i / (n_groups - 1)) for i in range(n_groups)]

    group_markers = ['o']*n_groups
    group_linestyles = ['-']*n_groups

    ############################## Modify as needed! ##############################
    rewrite = True #### Whether to overwrite performance metric plots
    rewrite_eeglevelresults=False  ##### Whether to overwrite EEG-level results
    ############################## Modify as needed! ##############################

    # Set paths
    base_continuous_dirs = ["/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC1_1/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC2/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC3/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC4/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC5/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC6/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC7/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC8/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC9/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC10/pred_continuousnonsz_checkpoint",
                            "/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_continuousnonsz_checkpoint",
                            ]

    base_discrete_file_prefixes = ["/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC1_1/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC2/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC3/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC4/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC5/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC6/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC7/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC8/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC9/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC10/pred_discrete_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_discrete_checkpoint",
                                   ]

    base_EEGlevel_file_prefixes=["/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_EEGlevel_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC1_1/pred_EEGlevel_checkpoint",
                                   "/data/seizure_hm/test_set/pred/pred_hm_IIIC2/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC3/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC4/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC5/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC6/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC7/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC8/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC9/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC10/pred_EEGlevel_checkpoint",
                                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_EEGlevel_checkpoint",
                                 ]

    base_morgothEEGlevel_file_prefixes = ["/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC1_1/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC2/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC3/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC4/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC5/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC6/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC7/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC8/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC9/pred_morgothEEGlevel_checkpoint",
                                    "/data/seizure_hm/test_set/pred/pred_hm_IIIC10/pred_morgothEEGlevel_checkpoint",
                                        "/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_morgothEEGlevel_checkpoint",
                                          ]

    continuous_test_data_list=pd.read_csv('/data/seizure_hm/test_set/continuous_szfree_list_20250929.csv')['file_name'].tolist()


    # ##### Create eROC data: eROC is computed using the SPaRCNet 10-minute EEG test set (containing seizures) combined with a 10-minute seizure-free test set (likely seizure-free) ########
    # for group_id, (base_continuous_dir, base_discrete_file_prefix, path_name) in tqdm(enumerate(
    #             zip(base_continuous_dirs, base_discrete_file_prefixes, path_names)),desc='create EEG level 1 data', total=len(base_continuous_dirs)):
    #     EEGlevel_output_dir = os.path.dirname(base_discrete_file_prefix)
    #     make_EEGlevel_results(base_continuous_dir=base_continuous_dir,
    #                           base_discrete_file_prefix=base_discrete_file_prefix,
    #                           continuous_test_data_list=continuous_test_data_list,
    #                           n_models=20,
    #                           post=False,
    #                           output_dir=EEGlevel_output_dir,
    #                           rewrite=rewrite_eeglevelresults,
    #                           step=10)
    # ######################################################################################################
    # ##### Create eROC data 2: eROC is based on morgoth eeg level model########
    # for group_id, (base_EEGlevel_file_prefix, base_discrete_file_prefix, path_name) in tqdm(enumerate(
    #             zip(base_EEGlevel_file_prefixes, base_discrete_file_prefixes, path_names)),desc='create EEG level 2 data'):
    #     EEGlevel_output_dir = os.path.dirname(base_discrete_file_prefix)
    #     make_EEGlevel_results2(model_EEGlevel_file_prefix=base_EEGlevel_file_prefix,
    #                           base_discrete_file_prefix=base_discrete_file_prefix,
    #                           continuous_test_data_list=continuous_test_data_list,
    #                           n_models=20,
    #                           post=False,
    #                           output_dir=EEGlevel_output_dir,
    #                           )


    ######################################################################################################

    # ###### Morgoth1 ###################
    morgoth1_output_dir = "/data/seizure_hm/test_set/pred/pred_hm_morgoth1"
    os.makedirs(morgoth1_output_dir, exist_ok=True)

    first_model_discrete_result_path = "/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_discrete_checkpoint0.csv"
    first_model_continuous_result_dir="/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_continuousnonsz_checkpoint0"
    first_model_EEGlevel_result_dir = "/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_EEGlevel_checkpoint0.csv"
    first_model_morgothEEGlevel_result_dir = "/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_morgothEEGlevel_checkpoint0.csv"

    morgoth1_discrete_path=os.path.join(morgoth1_output_dir,'pred_discrete.csv')
    morgoth1_continuous_dir =os.path.join(morgoth1_output_dir,'pred_continuousnonsz')
    morgoth1_EEGlevel_path=os.path.join(morgoth1_output_dir,'pred_EEGlevel.csv')
    morgoth1_morgothEEGlevel_path = os.path.join(morgoth1_output_dir, 'pred_morgothEEGlevel.csv')

    # print('Processing Morgoth 1 results......')
    # os.makedirs(morgoth1_continuous_dir, exist_ok=True)
    #
    # shutil.copy2(first_model_discrete_result_path, morgoth1_discrete_path)
    # shutil.copytree(first_model_continuous_result_dir, morgoth1_continuous_dir, dirs_exist_ok=True)
    # shutil.copy2(first_model_EEGlevel_result_dir, morgoth1_EEGlevel_path)
    # shutil.copy2(first_model_morgothEEGlevel_result_dir, morgoth1_morgothEEGlevel_path)
    # ######Morgoth1###################


    ###### Sparcnet ###################

    sparcnet_output_dir='/data/seizure_hm/test_set/pred/pred_hm_sparcnet'
    os.makedirs(sparcnet_output_dir, exist_ok=True)
    sparcnet_discrete_path=os.path.join(sparcnet_output_dir,'pred_discrete.csv')
    sparcnet_continuous_dir =os.path.join(sparcnet_output_dir,'pred_continuousnonsz')
    sparcnet_EEGlevel_path=os.path.join(sparcnet_output_dir,'pred_EEGlevel.csv')
    sparcnet_morgothEEGlevel_path= os.path.join(sparcnet_output_dir, 'pred_morgothEEGlevel.csv')

    # print('Processing Sparcnet results......')
    #
    # sparcnet_res(continuous_results_dir='/data/seizure_hm/test_set/pred/sparcnet_res/continuous_data',
    #              new_continuous_results_dir=sparcnet_continuous_dir,
    #              discrete_results_dir='/data/seizure_hm/test_set/pred/sparcnet_res/sparcnet_test_mat', discrete_results_template='/data/seizure_hm/test_set/pred/pred_hm_IIIC1/pred_discrete_checkpoint3.csv',
    #              discrete_results_path='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_discrete.csv',
    #              )
    #
    #
    # make_EEGlevel_results(base_continuous_dir='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_continuousnonsz',
    #                       base_discrete_file_prefix='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_discrete.csv',
    #                       continuous_test_data_list=continuous_test_data_list,
    #                       n_models=-1,
    #                       post=True,
    #                       output_dir='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_EEGlevel.csv',
    #                       step=10)
    #
    # make_EEGlevel_results2(
    #     model_EEGlevel_file_prefix='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_EEGlevel/pred_EEG_level_SEIZURE.csv',
    #     base_discrete_file_prefix='/data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_discrete.csv',
    #     continuous_test_data_list=continuous_test_data_list,
    #     n_models=-1,
    #     post=True,
    #     output_dir=sparcnet_morgothEEGlevel_path,
    #     )
    ###### Sparcnet ###################


    ###### post processing ###################
    post_output_dir = "/data/seizure_hm/test_set/pred/pred_hm_post"
    os.makedirs(post_output_dir, exist_ok=True)

    last_model_discrete_result_path = "/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_discrete_checkpoint19.csv"
    last_model_continuous_result_dir="/data/seizure_hm/test_set/pred/pred_hm_IIIC11/pred_continuousnonsz_checkpoint19"

    post_discrete_path=os.path.join(post_output_dir,'pred_discrete.csv')
    post_continuous_dir =os.path.join(post_output_dir,'pred_continuousnonsz')
    post_EEGlevel_path=os.path.join(post_output_dir,'pred_EEGlevel.csv')
    post_morgothEEGlevel_path = os.path.join(post_output_dir, 'pred_morgothEEGlevel.csv')

    # print('Processing post processing results......')
    # os.makedirs(post_continuous_dir, exist_ok=True)
    # shutil.copy2(last_model_discrete_result_path, post_discrete_path)
    #
    # process_csv_files(last_model_continuous_result_dir=last_model_continuous_result_dir,
    #                   post_continuous_dir=post_continuous_dir,
    #                   )
    #
    # make_EEGlevel_results(base_continuous_dir=post_continuous_dir,
    #                       base_discrete_file_prefix=post_discrete_path,
    #                       continuous_test_data_list=continuous_test_data_list,
    #                       n_models=-1,
    #                       post=True,
    #                       output_dir=post_EEGlevel_path,
    #                       step=10)
    #
    # make_EEGlevel_results2(model_EEGlevel_file_prefix='/data/seizure_hm/test_set/pred/pred_hm_post/pred_EEGlevel/pred_EEG_level_SEIZURE.csv',
    #                       base_discrete_file_prefix='/data/seizure_hm/test_set/pred/pred_hm_post/pred_discrete.csv',
    #                       continuous_test_data_list=continuous_test_data_list,
    #                       n_models=-1,
    #                       post=True,
    #                       output_dir=post_morgothEEGlevel_path,
    #                       )
    #####post_processing###################




    #eROC
    print('Calculate eROC 1 (all_eROC_results)......')
    all_eROC_results=[]
    for group_id, (base_EEGlevel_file_prefix, path_name) in enumerate(
            zip(base_EEGlevel_file_prefixes, path_names)):

        results=eROC_for_each_model(base_EEGlevel_file_prefix=base_EEGlevel_file_prefix,
                                    group_id=group_id,
                                    n_models=20)

        all_eROC_results.extend(results)

        if results:
            eroc_aucs = [r['eroc_auc'] for r in results]
            print(f"\n{path_name} group statistics:")
            print(f"  Models processed: {len(results)}")
            print(f"  eROC AUC: mean={np.mean(eroc_aucs):.4f}, std={np.std(eroc_aucs):.4f}")

    # smp eROC
    print('Calculate eROC 1 (smp_eROC_results)......')
    smp_eROC_results=[]
    # sparcnet
    results = eROC_for_each_model(base_EEGlevel_file_prefix=sparcnet_EEGlevel_path,
                                  group_id=-3,
                                  n_models=1,
                                  post=True)
    smp_eROC_results.extend(results)

    # morgoth1
    results = eROC_for_each_model(base_EEGlevel_file_prefix=morgoth1_EEGlevel_path,
                                  group_id=-2,
                                  n_models=1,
                                  post=True)
    smp_eROC_results.extend(results)

    # post
    results = eROC_for_each_model(base_EEGlevel_file_prefix=post_EEGlevel_path,
                                  group_id=-1,
                                  n_models=1,
                                  post=True
                                  )
    smp_eROC_results.extend(results)


    # eROC2
    print('Calculate eROC 1 (all_eROC2_results)......')
    all_eROC2_results = []
    for group_id, (base_morgothEEGlevel_file_prefix, path_name) in enumerate(
            zip(base_morgothEEGlevel_file_prefixes, path_names)):

        results = eROC_for_each_model(base_EEGlevel_file_prefix=base_morgothEEGlevel_file_prefix,
                                      group_id=group_id,
                                      n_models=20)

        all_eROC2_results.extend(results)

        if results:
            eroc_aucs = [r['eroc_auc'] for r in results]
            print(f"\n{path_name} group statistics:")
            print(f"  Models processed: {len(results)}")
            print(f"  eROC2 AUC: mean={np.mean(eroc_aucs):.4f}, std={np.std(eroc_aucs):.4f}")


    # smp eROC2
    print('Calculate eROC 1 (smp_eROC2_results)......')
    smp_eROC2_results = []

    # sparcnet
    results = eROC_for_each_model(base_EEGlevel_file_prefix=sparcnet_morgothEEGlevel_path,
                                  group_id=-3,
                                  n_models=1,
                                  post=True)
    smp_eROC2_results.extend(results)

    # morgoth1
    results = eROC_for_each_model(base_EEGlevel_file_prefix=morgoth1_morgothEEGlevel_path,
                                  group_id=-2,
                                  n_models=1,
                                  post=True)
    smp_eROC2_results.extend(results)

    # post
    results = eROC_for_each_model(base_EEGlevel_file_prefix=post_morgothEEGlevel_path,
                                  group_id=-1,
                                  n_models=1,
                                  post=True
                                  )
    smp_eROC2_results.extend(results)


    # mROC and ROC
    # Set output path for combined results
    output_dir = "/data/seizure_hm/test_set/pred/vis"
    os.makedirs(output_dir, exist_ok=True)

    summary_csv_path=os.path.join(output_dir, "combined_results_summary.csv")
    detailed_csv_path=os.path.join(output_dir, "combined_results_detailed.csv")

    smp_summary_csv_path = os.path.join(output_dir, "combined_results_smp_summary.csv")
    smp_detailed_csv_path = os.path.join(output_dir, "combined_results_smp_detailed.csv")


    if summary_csv_path!='' and detailed_csv_path!='' and not rewrite:
        print('Read mROC results......')
        all_combined_results, thresholds=load_combined_results_from_csv(summary_csv_path=summary_csv_path,
                                                                        detailed_csv_path=detailed_csv_path)

        smp_combined_results, _ = load_combined_results_from_csv(summary_csv_path=smp_summary_csv_path,
                                                                          detailed_csv_path=smp_detailed_csv_path)


    else:
        print('Calculate mROC results for HM models......')

        #################### Speed up if needed, modify as required! ####################
        # Define threshold range
        thresholds = np.linspace(0, 1, 51)
        # Step size when computing; window is 10 so should be 10, but 100 speeds things up
        step=10
        #################### Speed up if needed, modify as required! ####################

        # Store results for all groups
        all_combined_results = []

        # Process each model group
        for group_id, (base_continuous_dir, base_discrete_file_prefix, path_name) in tqdm(enumerate(
                zip(base_continuous_dirs, base_discrete_file_prefixes, path_names)),desc='FP per hour for each models',total=len(base_continuous_dirs)):

            print(f"\n{'=' * 50}")
            print(f"Processing model group {group_id}: {path_name}")
            print(f"{'=' * 50}")

            # Process all models in the current group
            group_results = process_single_model_group(
                base_continuous_dir=base_continuous_dir,
                continuous_test_data_list=continuous_test_data_list,
                base_discrete_file_prefix=base_discrete_file_prefix,
                group_id=group_id,
                model_indices=range(20),  # 0-19
                thresholds=thresholds,
                score_column='class_1_prob',
                prediction_column='class_1_prob',
                true_label_column='true',
                step=step
            )

            # Add to overall results
            all_combined_results.extend(group_results)

            # Print current group statistics
            if group_results:
                mroc_aucs = [r['mroc_auc'] for r in group_results]
                roc_aucs = [r['roc_auc'] for r in group_results]
                print(f"\n{path_name} group statistics:")
                print(f"  Models processed: {len(group_results)}")
                print(f"  mROC AUC: mean={np.mean(mroc_aucs):.4f}, std={np.std(mroc_aucs):.4f}")
                print(f"  ROC AUC: mean={np.mean(roc_aucs):.4f}, std={np.std(roc_aucs):.4f}")

        # Save combined results
        detailed_df, summary_df = save_combined_results_to_csv(
            all_combined_results, f"{output_dir}/combined_results.csv", thresholds)

        print('Calculate mROC results for parcnet, morgoth1, post......')
        # parcnet,morgoth1,post
        smp_combined_results = []

        # Add sparcnet
        sparcnet_results = process_single_model_group(
            base_continuous_dir=sparcnet_continuous_dir,
            continuous_test_data_list=continuous_test_data_list,
            base_discrete_file_prefix=sparcnet_discrete_path,
            group_id=-3,
            model_indices=range(1),
            thresholds=thresholds,
            score_column='class_1_prob',
            prediction_column='class_1_prob',
            true_label_column='true',
            post=True,
            step=step,
        )
        smp_combined_results.extend(sparcnet_results)

        # morgoth1
        morgoth1_results = process_single_model_group(
            base_continuous_dir=morgoth1_continuous_dir,
            continuous_test_data_list=continuous_test_data_list,
            base_discrete_file_prefix=morgoth1_discrete_path,
            group_id=-2,
            model_indices=range(1),
            thresholds=thresholds,
            score_column='class_1_prob',
            prediction_column='class_1_prob',
            true_label_column='true',
            post=True,
            step=step
        )
        smp_combined_results.extend(morgoth1_results)

        # post
        post_results = process_single_model_group(
            base_continuous_dir=post_continuous_dir,
            continuous_test_data_list=continuous_test_data_list,
            base_discrete_file_prefix=post_discrete_path,
            group_id=-1,
            model_indices=range(1),
            thresholds=thresholds,
            score_column='class_1_prob',
            prediction_column='class_1_prob',
            true_label_column='true',
            post=True,
            step=step
        )
        smp_combined_results.extend(post_results)

        save_combined_results_to_csv(
            smp_combined_results, f"{output_dir}/combined_results_smp.csv", thresholds)



    # Plot combined charts
    print(f"\n{'=' * 50}")
    print("Plotting combined result charts...")
    print(f"{'=' * 50}")

    # Plot combined mROC curves
    plot_combined_mroc_curves(all_combined_results,
                              f"{output_dir}/combined_mROC_all_groups.png",
                              group_colors=group_colors)

    # Plot group comparison
    plot_group_auc_comparison(group_names=path_names,
                              all_combined_results=all_combined_results,
                              figure_path=f"{output_dir}/combined_AUC_comparison_all_groups.png")

    # Plot group statistics
    groups_stats = plot_group_statistics(all_combined_results,
                                         f"{output_dir}/combined_group_statistics.png",
                                         group_names=path_names)


    # # Plot AUC trend comparison
    trend_stats = plot_combined_auc_trend(all_combined_results=all_combined_results,
                                          all_eROC_results=all_eROC_results,
                                          figure_path=f"{output_dir}/combined_AUC_trend_comparison.png",
                                          group_colors=group_colors_background)

    # Plot various ROC curve comparisons for the best model of each group
    group_colors = [cmap(i / (n_groups + 3-1)) for i in range(n_groups+3)]
    group_colors=group_colors[::-1]
    group_markers = ['o'] * n_groups + ['^'] * 3
    group_linestyles = ['-'] * (n_groups+3)

    best_models_info, best_models_details = add_best_models_plotting(
        group_names=path_names,
        smp_names=['sparcnet','morgoth1','post'],
        all_combined_results=all_combined_results,
        ROC_results_prefix='/data/seizure_hm/test_set/pred/pred_hm_',
        smp_combined_results=smp_combined_results,
        all_eROC_results=all_eROC_results,
        smp_eROC_results=smp_eROC_results,
        eROC_results_prefix='/data/seizure_hm/test_set/pred/pred_hm_',
        all_eROC2_results=all_eROC2_results,
        smp_eROC2_results=smp_eROC2_results,
        output_dir=output_dir,
        group_colors=group_colors,
        group_markers=group_markers,
        group_linestyles=group_linestyles)



    # Print final statistics
    print(f"\n{'=' * 50}")
    print("Final statistics")
    print(f"{'=' * 50}")
    print(f"Total models processed: {len(all_combined_results)}")

    for group_id, stats in groups_stats.items():
        print(f"\n{stats['name']} group:")
        print(f"  Number of models: {stats['count']}")
        print(
            f"  mROC AUC: {stats['mroc_mean']:.4f} ± {stats['mroc_std']:.4f} (range: {stats['mroc_min']:.4f}-{stats['mroc_max']:.4f})")
        print(
            f"  ROC AUC: {stats['roc_mean']:.4f} ± {stats['roc_std']:.4f} (range: {stats['roc_min']:.4f}-{stats['roc_max']:.4f})")

    # Find the best model
    best_mroc_result = max(all_combined_results, key=lambda x: x['mroc_auc'])
    best_roc_result = max(all_combined_results, key=lambda x: x['roc_auc'])

    print(f"\nBest model:")
    print(f"  Best mROC AUC: {best_mroc_result['model_id']} (AUC={best_mroc_result['mroc_auc']:.4f})")
    print(f"  Best ROC AUC: {best_roc_result['model_id']} (AUC={best_roc_result['roc_auc']:.4f})")

    # Print trend statistics
    print(f"\nContinuous trend statistics:")
    print(f"  Total models: {trend_stats['total_models']}")
    print(f"  Best mROC AUC: {trend_stats['best_mroc_model']} (AUC={trend_stats['best_mroc_value']:.4f})")
    print(f"  Worst mROC AUC: {trend_stats['worst_mroc_model']} (AUC={trend_stats['worst_mroc_value']:.4f})")
    print(f"  Best ROC AUC: {trend_stats['best_roc_model']} (AUC={trend_stats['best_roc_value']:.4f})")
    print(f"  Worst ROC AUC: {trend_stats['worst_roc_model']} (AUC={trend_stats['worst_roc_value']:.4f})")
    print(f"  mROC AUC mean: {trend_stats['mean_mroc']:.4f}, range: {trend_stats['mroc_range']:.4f}")
    print(f"  ROC AUC mean: {trend_stats['mean_roc']:.4f}, range: {trend_stats['roc_range']:.4f}")

    print(f"\n{'=' * 50}")
    print("All processing complete!")
    print(f"{'=' * 50}")

# sudo ~/miniconda3/envs/torchenv/bin/python vis_hm.py
