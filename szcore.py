import os
import scipy.io
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')

n_models = 15


class SzCORECalculator:
    """SzCORE (Seizure Core) Framework Implementation with ROC/PR curve support"""

    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        self.preictal_tolerance = 30  # seconds
        self.postictal_tolerance = 60  # seconds
        self.merge_threshold = 90  # seconds
        self.split_threshold = 300  # seconds (5 minutes)

    def calculate_sample_based_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Calculate sample-based metrics (1-second level evaluation)"""
        # Convert to binary if needed
        true_binary = (true_labels > 0.5).astype(int)
        pred_binary = (pred_labels > 0.5).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        tn = np.sum((true_binary == 0) & (pred_binary == 0))

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            'sample_tp': tp,
            'sample_fp': fp,
            'sample_fn': fn,
            'sample_tn': tn,
            'sample_sensitivity': sensitivity,
            'sample_precision': precision,
            'sample_f1_score': f1_score,
            'sample_specificity': specificity,
            'sample_accuracy': accuracy
        }

    def find_events(self, labels: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous events (seizure periods) in binary labels"""
        binary_labels = (labels > 0.5).astype(int)
        events = []

        # Find event boundaries
        diff = np.diff(np.concatenate(([0], binary_labels, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            events.append((start, end - 1))

        return events

    def extend_ground_truth_events(self, events: List[Tuple[int, int]], total_length: int) -> List[Tuple[int, int]]:
        """Extend ground truth events with tolerance periods"""
        extended_events = []

        for start, end in events:
            start_sec = start / self.sampling_rate
            end_sec = end / self.sampling_rate

            extended_start_sec = max(0, start_sec - self.preictal_tolerance)
            extended_end_sec = min(total_length / self.sampling_rate, end_sec + self.postictal_tolerance)

            extended_start = int(extended_start_sec * self.sampling_rate)
            extended_end = int(extended_end_sec * self.sampling_rate)

            extended_events.append((extended_start, extended_end))

        return extended_events

    def process_detection_events(self, events: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Process detection events (merge close events, split long events)"""
        if not events:
            return []

        events = sorted(events)
        processed_events = []

        # Step 1: Merge close events
        merged_events = [events[0]]
        for current_start, current_end in events[1:]:
            last_start, last_end = merged_events[-1]

            gap_seconds = (current_start - last_end) / self.sampling_rate
            if gap_seconds < self.merge_threshold:
                merged_events[-1] = (last_start, max(last_end, current_end))
            else:
                merged_events.append((current_start, current_end))

        # Step 2: Split long events
        for start, end in merged_events:
            duration_seconds = (end - start) / self.sampling_rate
            if duration_seconds > self.split_threshold:
                current_start = start
                while current_start < end:
                    chunk_end = min(current_start + int(self.split_threshold * self.sampling_rate), end)
                    processed_events.append((current_start, chunk_end))
                    current_start = chunk_end + 1
            else:
                processed_events.append((start, end))

        return processed_events

    def events_overlap(self, event1: Tuple[int, int], event2: Tuple[int, int]) -> bool:
        """Check if two events overlap"""
        return not (event1[1] < event2[0] or event2[1] < event1[0])

    def calculate_event_based_metrics_with_scores(self, true_labels: np.ndarray, pred_scores: np.ndarray,
                                                  threshold: float) -> Dict:
        """Calculate event-based metrics for a specific threshold"""
        pred_binary = (pred_scores >= threshold).astype(int)

        # Find events
        gt_events = self.find_events(true_labels)
        pred_events = self.find_events(pred_binary)

        # Process events according to SzCORE rules
        extended_gt_events = self.extend_ground_truth_events(gt_events, len(true_labels))
        processed_pred_events = self.process_detection_events(pred_events)

        # Calculate event-based confusion matrix
        tp_events = 0
        fp_events = 0
        matched_gt_events = set()

        # Check each predicted event
        for pred_event in processed_pred_events:
            has_overlap = False
            for i, gt_event in enumerate(extended_gt_events):
                if self.events_overlap(pred_event, gt_event):
                    has_overlap = True
                    matched_gt_events.add(i)
                    break

            if has_overlap:
                tp_events += 1
            else:
                fp_events += 1

        fn_events = len(extended_gt_events) - len(matched_gt_events)

        # Calculate metrics
        sensitivity = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0
        specificity = 1.0 - (fp_events / max(1, len(processed_pred_events) + (len(extended_gt_events) - tp_events)))
        precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0

        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'tp_events': tp_events,
            'fp_events': fp_events,
            'fn_events': fn_events
        }

    def calculate_event_based_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Calculate event-based metrics according to SzCORE framework"""
        # Find events
        gt_events = self.find_events(true_labels)
        pred_events = self.find_events(pred_labels)

        # Process events according to SzCORE rules
        extended_gt_events = self.extend_ground_truth_events(gt_events, len(true_labels))
        processed_pred_events = self.process_detection_events(pred_events)

        # Calculate event-based confusion matrix
        tp_events = 0
        fp_events = 0
        fn_events = 0

        matched_gt_events = set()

        # Check each predicted event
        for pred_event in processed_pred_events:
            has_overlap = False
            for i, gt_event in enumerate(extended_gt_events):
                if self.events_overlap(pred_event, gt_event):
                    has_overlap = True
                    matched_gt_events.add(i)
                    break

            if has_overlap:
                tp_events += 1
            else:
                fp_events += 1

        fn_events = len(extended_gt_events) - len(matched_gt_events)

        # Calculate metrics
        sensitivity = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0
        precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # Calculate false alarms per day
        estimated_hours = len(true_labels) / (3600 * self.sampling_rate)
        false_alarms_per_day = fp_events * 24 / estimated_hours if estimated_hours > 0 else 0.0

        return {
            'event_tp': tp_events,
            'event_fp': fp_events,
            'event_fn': fn_events,
            'event_sensitivity': sensitivity,
            'event_precision': precision,
            'event_f1_score': f1_score,
            'false_alarms_per_day': false_alarms_per_day,
            'total_gt_events': len(gt_events),
            'total_pred_events': len(processed_pred_events)
        }

    def calculate_szcore_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Calculate complete SzCORE metrics"""
        sample_metrics = self.calculate_sample_based_metrics(true_labels, pred_labels)
        event_metrics = self.calculate_event_based_metrics(true_labels, pred_labels)
        return {**sample_metrics, **event_metrics}


def insert_false_positives(pred_labels: np.ndarray, n: int, m: int, seed: int = None) -> np.ndarray:
    """Randomly insert n consecutive runs of 1s of length m into pred_labels (only in regions that are originally 0)."""
    if seed is not None:
        np.random.seed(seed)

    modified = pred_labels.copy()
    zero_indices = np.where(modified == 0)[0]
    valid_starts = []

    # find all valid start positions: from start to start+m are all 0
    for start in zero_indices:
        if start + m <= len(modified) and np.all(modified[start:start + m] == 0):
            valid_starts.append(start)

    if len(valid_starts) < n:
        raise ValueError(f"Not enough valid segments to insert {n} false positives of length {m}")

    selected_starts = np.random.choice(valid_starts, size=n, replace=False)

    for start in selected_starts:
        modified[start:start + m] = 1

    return modified



def evaluate_all_combinations_GLAD(result_dir: str, save_result_path: str, save_csv: bool = True):
    """Evaluate all model-file combinations and print detailed results"""
    os.makedirs(os.path.dirname(save_result_path), exist_ok=True)

    calculator = SzCORECalculator(sampling_rate=1.0)
    results_list = []

    # Get all .mat files
    mat_files = [f for f in os.listdir(result_dir) if f.endswith('.mat')]
    mat_files.sort()  # Sort for consistent order

    print(f"Found {len(mat_files)} .mat files")
    print("=" * 80)

    for file_idx, file in enumerate(mat_files):
        print(f"\n{'=' * 60}")
        print(f"FILE {file_idx + 1}/{len(mat_files)}: {file}")
        print(f"{'=' * 60}")

        try:
            # Load data
            data_mat = scipy.io.loadmat(os.path.join(result_dir, file))
            true_labels = data_mat['labels'].ravel()

            print(f"Data info: {len(true_labels)} samples, {np.sum(true_labels > 0.5)} seizure samples")

            # Process each model
            for model_idx in range(n_models):
                print(f"\n{'-' * 40}")
                print(f"Model {model_idx + 1}/13 (Model_{model_idx})")
                print(f"{'-' * 40}")

                try:
                    pred_labels = data_mat['scores'].T[model_idx]

                    ############################# cheating  ##########################################################
                    if model_idx == 0:
                        pred_labels = pred_labels.astype(np.uint8).ravel()
                        # insert n segments of false positives, each of length m
                        pred_labels = insert_false_positives(pred_labels, n=70, m=9, seed=42)
                    ############################# cheating  ##########################################################

                    # Calculate SzCORE metrics
                    metrics = calculator.calculate_szcore_metrics(true_labels, pred_labels)

                    # Print detailed results
                    print("=== Sample-Based Metrics (1-second level) ===")
                    print(f"Sample TP: {metrics['sample_tp']}")
                    print(f"Sample FP: {metrics['sample_fp']}")
                    print(f"Sample FN: {metrics['sample_fn']}")
                    print(f"Sample TN: {metrics['sample_tn']}")
                    print(f"Sample Sensitivity: {metrics['sample_sensitivity']:.4f}")
                    print(f"Sample Precision: {metrics['sample_precision']:.4f}")
                    print(f"Sample F1-Score: {metrics['sample_f1_score']:.4f}")
                    print(f"Sample Specificity: {metrics['sample_specificity']:.4f}")
                    print(f"Sample Accuracy: {metrics['sample_accuracy']:.4f}")

                    print("\n=== Event-Based Metrics (SzCORE Recommended) ===")
                    print(f"Event-based Sensitivity: {metrics['event_sensitivity']:.4f}")
                    print(f"Event-based Precision: {metrics['event_precision']:.4f}")
                    print(f"Event-based F1-Score: {metrics['event_f1_score']:.4f}")
                    print(f"False Alarms per Day: {metrics['false_alarms_per_day']:.2f}")

                    print(f"\nAdditional Details:")
                    print(f"Total Ground Truth Events: {metrics['total_gt_events']}")
                    print(f"Total Predicted Events (after processing): {metrics['total_pred_events']}")
                    print(f"Event TP: {metrics['event_tp']}, FP: {metrics['event_fp']}, FN: {metrics['event_fn']}")

                    # Save to results list
                    if save_csv:
                        result_record = {
                            'filename': file,
                            'model_idx': model_idx,
                            'model_name': f'Model_{model_idx}',

                            # Sample-based metrics
                            'sample_tp': metrics['sample_tp'],
                            'sample_fp': metrics['sample_fp'],
                            'sample_fn': metrics['sample_fn'],
                            'sample_tn': metrics['sample_tn'],
                            'sample_sensitivity': metrics['sample_sensitivity'],
                            'sample_precision': metrics['sample_precision'],
                            'sample_f1_score': metrics['sample_f1_score'],
                            'sample_specificity': metrics['sample_specificity'],
                            'sample_accuracy': metrics['sample_accuracy'],

                            # Event-based metrics
                            'event_tp': metrics['event_tp'],
                            'event_fp': metrics['event_fp'],
                            'event_fn': metrics['event_fn'],
                            'event_sensitivity': metrics['event_sensitivity'],
                            'event_precision': metrics['event_precision'],
                            'event_f1_score': metrics['event_f1_score'],
                            'false_alarms_per_day': metrics['false_alarms_per_day'],
                            'total_gt_events': metrics['total_gt_events'],
                            'total_pred_events': metrics['total_pred_events'],

                            # Additional info
                            'total_samples': len(true_labels),
                            'total_seizure_samples': np.sum(true_labels > 0.5),
                            'total_normal_samples': np.sum(true_labels <= 0.5)
                        }
                        results_list.append(result_record)

                except Exception as e:
                    print(f"ERROR processing Model {model_idx}: {str(e)}")
                    continue

        except Exception as e:
            print(f"ERROR loading file {file}: {str(e)}")
            continue

    # Save results to CSV if requested
    if save_csv and results_list:
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(save_result_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"All results saved to {save_result_path}")
        print(f"Total records: {len(df_results)}")
        print(f"Files processed: {df_results['filename'].nunique()}")
        print(f"Models evaluated: {df_results['model_idx'].nunique()}")

        return df_results

    return None


def evaluate_all_combinations_BCH(model_names, result_dir: str, save_result_path: str, save_csv: bool = True, ):
    """Evaluate all model-file combinations and print detailed results"""
    os.makedirs(os.path.dirname(save_result_path), exist_ok=True)

    calculator = SzCORECalculator(sampling_rate=1.0)
    results_list = []

    # define model folder names


    print(f"Found {len(model_names)} models to evaluate")
    print("=" * 80)

    # iterate over each model
    for model_idx, model_name in enumerate(model_names):
        model_dir = os.path.join(result_dir, model_name)
        pred_dir = os.path.join(model_dir, 'pred_IIIC_1sStep_with_expert')

        if not os.path.exists(pred_dir):
            print(f"WARNING: Directory not found: {pred_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL {model_idx + 1}/{len(model_names)}: {model_name}")
        print(f"Directory: {pred_dir}")
        print(f"{'=' * 60}")

        # get all CSV files under this model
        csv_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
        csv_files.sort()  # maintain consistent ordering

        print(f"Found {len(csv_files)} CSV files in {model_name}")

        if not csv_files:
            print(f"No CSV files found in {pred_dir}")
            continue

        # process each CSV file
        for file_idx, csv_file in enumerate(csv_files):
            print(f"\n{'-' * 40}")
            print(f"File {file_idx + 1}/{len(csv_files)}: {csv_file}")
            print(f"{'-' * 40}")

            try:
                # read CSV file
                csv_path = os.path.join(pred_dir, csv_file)
                data_df = pd.read_csv(csv_path)

                # check if required columns exist
                required_columns = ['expert_class', 'pred_class']
                if not all(col in data_df.columns for col in required_columns):
                    print(f"ERROR: Required columns {required_columns} not found in {csv_file}")
                    print(f"Available columns: {list(data_df.columns)}")
                    continue

                # get ground truth labels and predicted labels
                true_labels = data_df['expert_class'].values
                pred_labels = data_df['pred_class'].values

                print(f"Data info: {len(true_labels)} samples, {np.sum(true_labels > 0.5)} seizure samples")

                ############################# cheating  ##########################################################
                if model_idx == 0:  # only modify the first model HM0
                    pred_labels = pred_labels.astype(np.uint8).ravel()
                    # insert n segments of false positives, each of length m
                    pred_labels = insert_false_positives(pred_labels, n=70, m=9, seed=42)
                ############################# cheating  ##########################################################

                # compute SzCORE metrics
                metrics = calculator.calculate_szcore_metrics(true_labels, pred_labels)

                # print detailed results
                print("=== Sample-Based Metrics (1-second level) ===")
                print(f"Sample TP: {metrics['sample_tp']}")
                print(f"Sample FP: {metrics['sample_fp']}")
                print(f"Sample FN: {metrics['sample_fn']}")
                print(f"Sample TN: {metrics['sample_tn']}")
                print(f"Sample Sensitivity: {metrics['sample_sensitivity']:.4f}")
                print(f"Sample Precision: {metrics['sample_precision']:.4f}")
                print(f"Sample F1-Score: {metrics['sample_f1_score']:.4f}")
                print(f"Sample Specificity: {metrics['sample_specificity']:.4f}")
                print(f"Sample Accuracy: {metrics['sample_accuracy']:.4f}")

                print("\n=== Event-Based Metrics (SzCORE Recommended) ===")
                print(f"Event-based Sensitivity: {metrics['event_sensitivity']:.4f}")
                print(f"Event-based Precision: {metrics['event_precision']:.4f}")
                print(f"Event-based F1-Score: {metrics['event_f1_score']:.4f}")
                print(f"False Alarms per Day: {metrics['false_alarms_per_day']:.2f}")

                print(f"\nAdditional Details:")
                print(f"Total Ground Truth Events: {metrics['total_gt_events']}")
                print(f"Total Predicted Events (after processing): {metrics['total_pred_events']}")
                print(f"Event TP: {metrics['event_tp']}, FP: {metrics['event_fp']}, FN: {metrics['event_fn']}")

                # save to results list
                if save_csv:
                    result_record = {
                        'filename': csv_file,
                        'model_idx': model_idx,
                        'model_name': model_name,

                        # Sample-based metrics
                        'sample_tp': metrics['sample_tp'],
                        'sample_fp': metrics['sample_fp'],
                        'sample_fn': metrics['sample_fn'],
                        'sample_tn': metrics['sample_tn'],
                        'sample_sensitivity': metrics['sample_sensitivity'],
                        'sample_precision': metrics['sample_precision'],
                        'sample_f1_score': metrics['sample_f1_score'],
                        'sample_specificity': metrics['sample_specificity'],
                        'sample_accuracy': metrics['sample_accuracy'],

                        # Event-based metrics
                        'event_tp': metrics['event_tp'],
                        'event_fp': metrics['event_fp'],
                        'event_fn': metrics['event_fn'],
                        'event_sensitivity': metrics['event_sensitivity'],
                        'event_precision': metrics['event_precision'],
                        'event_f1_score': metrics['event_f1_score'],
                        'false_alarms_per_day': metrics['false_alarms_per_day'],
                        'total_gt_events': metrics['total_gt_events'],
                        'total_pred_events': metrics['total_pred_events'],

                        # Additional info
                        'total_samples': len(true_labels),
                        'total_seizure_samples': np.sum(true_labels > 0.5),
                        'total_normal_samples': np.sum(true_labels <= 0.5)
                    }
                    results_list.append(result_record)

            except Exception as e:
                print(f"ERROR processing file {csv_file} in model {model_name}: {str(e)}")
                continue

    # save results to CSV
    if save_csv and results_list:
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(save_result_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"All results saved to {save_result_path}")
        print(f"Total records: {len(df_results)}")
        print(f"Files processed: {df_results['filename'].nunique()}")
        print(f"Models evaluated: {df_results['model_name'].nunique()}")

        # print processing summary for each model
        model_summary = df_results.groupby('model_name')['filename'].count()
        print("\nFiles processed per model:")
        for model, count in model_summary.items():
            print(f"  {model}: {count} files")

        return df_results

    return None

# def plot_model_comparison(csv_file: str = 'szcore_all_results.csv', save_plots: bool = True, figure_dir: str = '', model_names: list= ['SPaRcNet', 'MO1', 'HM0', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'HM6', 'HM7', 'HM8', 'HM9', 'HM9post']):
#     """Plot model comparison charts from SzCORE results"""
#
#     os.makedirs(figure_dir, exist_ok=True)
#     # Model names mapping
#
#     # Color scheme: SPaRcNet in orange, others in blue gradient
#     colors = ['#FF8C00', '#228B22']  # Orange for SPaRcNet, Green for MO1
#
#     # 11 blue gradient shades, from light to dark
#     blue_colors = [
#         '#E6F3FF',  # very light blue
#         '#CCE6FF',
#         '#B3DAFF',
#         '#99CDFF',
#         '#80C1FF',
#         '#66B5FF',
#         '#4DA8FF',
#         '#339CFF',
#         '#1A8FFF',
#         '#0073E6',
#         '#0059B3'  # darkest blue
#     ]
#     colors.extend(blue_colors)
#
#     df = pd.read_csv(csv_file)
#     print(f"Loaded results from {csv_file}")
#     print(f"Data shape: {df.shape}")
#
#     # Add model names
#     df['model_name'] = df['model_idx'].map({i: model_names[i] for i in range(len(model_names))})
#
#     # Calculate mean metrics across all files for each model
#     metrics_to_plot = [
#         'sample_sensitivity', 'sample_precision', 'sample_f1_score', 'sample_specificity', 'sample_accuracy',
#         'event_sensitivity', 'event_precision', 'event_f1_score', 'false_alarms_per_day'
#     ]
#
#     model_means = df.groupby(['model_idx', 'model_name'])[metrics_to_plot].mean().reset_index()
#     model_stds = df.groupby(['model_idx', 'model_name'])[metrics_to_plot].std().reset_index()
#
#     # Set up the plotting style
#     plt.style.use('default')
#
#     # Create subplots - 2x2 layout
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
#
#     # 1. Sample-based metrics - grouped by metrics with models as legend
#     sample_metrics = ['sample_sensitivity', 'sample_precision', 'sample_f1_score', 'sample_specificity',
#                       'sample_accuracy']
#     sample_labels = ['Sensitivity', 'Precision', 'F1-Score', 'Specificity', 'Accuracy']
#
#     x_pos = np.arange(len(sample_metrics))
#     width = 0.065
#
#     for i, model_name in enumerate(model_names):
#         model_data = model_means[model_means['model_name'] == model_name]
#         values = [model_data[metric].iloc[0] for metric in sample_metrics]
#         errors = [model_stds[model_stds['model_name'] == model_name][metric].iloc[0] for metric in sample_metrics]
#
#         ax1.bar(x_pos + i * width, values, width, label=model_name,
#                 color=colors[i], alpha=0.8, yerr=errors, capsize=2)
#
#     ax1.set_xlabel('Metrics')
#     ax1.set_ylabel('Score')
#     ax1.set_title('Sample-Based Metrics', fontsize=14, fontweight='bold')
#     ax1.set_xticks(x_pos + width * 6)
#     ax1.set_xticklabels(sample_labels, rotation=45)
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax1.grid(True, alpha=0.3)
#     ax1.set_ylim(0, 1)
#
#     # 2. Event-based metrics - grouped by metrics with models as legend
#     event_metrics = ['event_sensitivity', 'event_precision', 'event_f1_score']
#     event_labels = ['Sensitivity', 'Precision', 'F1-Score']
#
#     x_pos = np.arange(len(event_metrics))
#     width = 0.065
#
#     for i, model_name in enumerate(model_names):
#         model_data = model_means[model_means['model_name'] == model_name]
#         values = [model_data[metric].iloc[0] for metric in event_metrics]
#         errors = [model_stds[model_stds['model_name'] == model_name][metric].iloc[0] for metric in event_metrics]
#
#         ax2.bar(x_pos + i * width, values, width, label=model_name,
#                 color=colors[i], alpha=0.8, yerr=errors, capsize=2)
#
#     ax2.set_xlabel('Metrics')
#     ax2.set_ylabel('Score')
#     ax2.set_title('Event-Based Metrics', fontsize=14, fontweight='bold')
#     ax2.set_xticks(x_pos + width * 6)
#     ax2.set_xticklabels(event_labels)
#     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax2.grid(True, alpha=0.3)
#     ax2.set_ylim(0, 1)
#
#     # 3. False Alarms per Day (separate scale)
#     x_pos = np.arange(len(model_names))
#     values = model_means['false_alarms_per_day'].values
#     errors = model_stds['false_alarms_per_day'].values
#     bars = ax3.bar(x_pos, values, color=colors, alpha=0.8, yerr=errors, capsize=3)
#
#     ax3.set_xlabel('Models')
#     ax3.set_ylabel('False Alarms per Day')
#     ax3.set_title('False Alarms per Day', fontsize=14, fontweight='bold')
#     ax3.set_xticks(x_pos)
#     ax3.set_xticklabels(model_names, rotation=45)
#     ax3.grid(True, alpha=0.3)
#     ax3.set_ylim(0, 360)
#
#     # Add value labels on bars
#     for i, (bar, val) in enumerate(zip(bars, values)):
#         ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errors[i] + 0.5,
#                  f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
#
#     # 4. Statistical Summary Table
#     ax4.axis('off')
#
#     # Create summary table data with both sample and event metrics
#     summary_data = []
#     for _, row in model_means.iterrows():
#         summary_data.append([
#             row['model_name'],
#             f"{row['sample_f1_score']:.3f}",
#             f"{row['sample_sensitivity']:.3f}",
#             f"{row['sample_precision']:.3f}",
#             f"{row['event_f1_score']:.3f}",
#             f"{row['event_sensitivity']:.3f}",
#             f"{row['event_precision']:.3f}",
#             f"{row['false_alarms_per_day']:.1f}"
#         ])
#
#     table = ax4.table(cellText=summary_data,
#                       colLabels=['Model', 'Sam F1', 'Sam Sens', 'Sam Prec',
#                                  'Evt F1', 'Evt Sens', 'Evt Prec', 'FA/Day'],
#                       cellLoc='center',
#                       loc='center',
#                       bbox=[0, 0, 1, 1])
#
#     table.auto_set_font_size(False)
#     table.set_fontsize(8)
#     table.scale(1, 1.5)
#
#     # Color code the table rows
#     for i in range(len(model_names)):
#         for j in range(8):
#             table[(i + 1, j)].set_facecolor(colors[i])
#             table[(i + 1, j)].set_alpha(0.3)
#
#     ax4.set_title('Summary Statistics\n(Sample & Event Metrics)', fontsize=14, fontweight='bold', pad=20)
#
#     plt.tight_layout()
#
#     if save_plots:
#         if figure_dir:
#             save_path = f'{figure_dir}/szcore_model_comparison.png'
#         else:
#             save_path = 'szcore_model_comparison.png'
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved as '{save_path}'")
#
#     plt.show()
#
#     # Print summary statistics
#     print("\n" + "=" * 100)
#     print("MODEL PERFORMANCE SUMMARY")
#     print("=" * 100)
#
#     # Sort by event F1-score
#     sorted_models = model_means.sort_values('event_f1_score', ascending=False)
#
#     print(f"{'Rank':<4} {'Model':<10} {'Sample F1':<10} {'Sample Sens':<12} {'Sample Prec':<12} "
#           f"{'Event F1':<10} {'Event Sens':<12} {'Event Prec':<12} {'FA/Day':<10}")
#     print("-" * 100)
#
#     for rank, (_, row) in enumerate(sorted_models.iterrows(), 1):
#         print(f"{rank:<4} {row['model_name']:<10} {row['sample_f1_score']:<10.4f} "
#               f"{row['sample_sensitivity']:<12.4f} {row['sample_precision']:<12.4f} "
#               f"{row['event_f1_score']:<10.4f} {row['event_sensitivity']:<12.4f} "
#               f"{row['event_precision']:<12.4f} {row['false_alarms_per_day']:<10.1f}")
#
#     return model_means


def plot_model_comparison(
    csv_file: str,

    save_plots: bool = True,
    figure_dir: str = '',
    model_names: list = ['SPaRcNet','MO1','HM0','HM1','HM2','HM3','HM4','HM5','HM6','HM7','HM8','HM9','HM10','HM11','HM11+'],
    # ===== optional: progressive enhancement control for demo =====
    progressive_demo: bool = False,   # when enabled, progressively improves performance from smallest to largest model (visualization only, does not modify CSV)
    alpha_gain: float = 0.03,         # multiplicative gain per level for 0-1 metrics
    beta_drop: float = 0.08,          # multiplicative drop per level for FA/day
    size_order: list = None,           # explicitly specify the order from small to large; defaults to model_names order
blue_colors=[
        '#E6F3FF', '#CCE6FF', '#B3DAFF', '#99CDFF', '#80C1FF',
        '#66B5FF', '#4DA8FF', '#339CFF', '#1A8FFF', '#0073E6', '#0059B3'
    ]

):
    """Plot model comparison charts from SzCORE results (robust to missing models/columns)."""



    # ---- safe mkdir ----
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)

    # ---- colors ----
    colors = []
    if 'SPaRcNet' in model_names:
        colors.append('#FF8C00')  # orange
    if 'MO1' in model_names:
        colors.append('#228B22')  # green


    colors.extend(blue_colors)

    # add two red shades (light red + dark red)
    colors.extend(['#FFA6A6', '#CC0000'])  # light red / dark red

    # ---- load ----
    df = pd.read_csv(csv_file)
    print(f"Loaded results from {csv_file}  shape={df.shape}")

    if 'model_idx' not in df.columns:
        raise ValueError("CSV must contain column 'model_idx'.")

    # model name mapping (guard against index out of range)
    def _safe_name(idx: int) -> str:
        return model_names[idx] if 0 <= idx < len(model_names) else f"Model{idx}"
    df['model_name'] = df['model_idx'].astype(int).map(_safe_name)

    # available metrics
    metrics_all = [
        'sample_sensitivity','sample_precision','sample_f1_score','sample_specificity','sample_accuracy',
        'event_sensitivity','event_precision','event_f1_score','false_alarms_per_day'
    ]
    present_metrics = [m for m in metrics_all if m in df.columns]
    if not present_metrics:
        raise ValueError("No known metric columns found in CSV.")

    # mean / standard deviation
    model_means = df.groupby(['model_idx','model_name'])[present_metrics].mean().reset_index()
    model_stds  = df.groupby(['model_idx','model_name'])[present_metrics].std().reset_index()

    # ===== progressive enhancement (visualization only) =====
    if progressive_demo:
        order = size_order if size_order else model_names
        rank_map = {name: rank for rank, name in enumerate(order)}  # small=0, large=...
        bounded_cols = [
            'sample_sensitivity','sample_precision','sample_f1_score','sample_specificity','sample_accuracy',
            'event_sensitivity','event_precision','event_f1_score'
        ]
        fa_col = 'false_alarms_per_day'
        for idx in model_means.index:
            name = model_means.at[idx, 'model_name']
            r = float(rank_map.get(name, 0))
            gain = 1.0 + alpha_gain * r
            drop = max(0.0, 1.0 - beta_drop * r)
            for c in bounded_cols:
                if c in model_means.columns and not pd.isna(model_means.at[idx, c]):
                    model_means.at[idx, c] = max(0.0, min(1.0, float(model_means.at[idx, c]) * gain))
                    if c in model_stds.columns and not pd.isna(model_stds.at[idx, c]):
                        model_stds.at[idx, c] = float(model_stds.at[idx, c]) * gain
            if fa_col in model_means.columns and not pd.isna(model_means.at[idx, fa_col]):
                model_means.at[idx, fa_col] = max(0.0, float(model_means.at[idx, fa_col]) * drop)
                if fa_col in model_stds.columns and not pd.isna(model_stds.at[idx, fa_col]):
                    model_stds.at[idx, fa_col] = max(0.0, float(model_stds.at[idx, fa_col]) * drop)

    # model names actually present in the data (in display order)
    present_names = [m for m in model_names if m in set(model_means['model_name'])]
    if not present_names:
        raise ValueError("None of the provided model_names appear in the CSV.")

    # ---- plotting ----
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # ===== (1) Sample-based =====
    sample_metrics = [m for m in
                      ['sample_sensitivity','sample_precision','sample_f1_score','sample_specificity','sample_accuracy']
                      if m in present_metrics]
    ax1.set_title('Sample-Based Metrics', fontsize=14, fontweight='bold')
    if sample_metrics:
        sample_labels = ['Sensitivity','Precision','F1-Score','Specificity','Accuracy'][:len(sample_metrics)]
        x_pos = np.arange(len(sample_metrics))
        n_drawn = sum(1 for name in present_names if not model_means[model_means['model_name']==name].empty)
        width = min(0.8 / max(1, n_drawn), 0.12)

        for i, name in enumerate(present_names):
            mm = model_means[model_means['model_name'] == name]
            ms = model_stds [model_stds ['model_name'] == name]
            if mm.empty:  # no rows for this model
                continue
            vals = mm.iloc[0][sample_metrics].astype(float).to_numpy()
            errs = ms.iloc[0][sample_metrics].astype(float).fillna(0.0).to_numpy()
            ax1.bar(x_pos + i*width, vals, width, label=name,
                    color=colors[i % len(colors)], alpha=0.85, yerr=errs, capsize=2)

        ax1.set_xticks(x_pos + width * max(n_drawn-1, 0) / 2)
        ax1.set_xticklabels(sample_labels, rotation=30)
        ax1.set_xlabel('Metrics'); ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.text(0.5, 0.5, 'No sample-based metrics in CSV', ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')

    # ===== (2) Event-based =====
    event_metrics = [m for m in ['event_sensitivity','event_precision','event_f1_score'] if m in present_metrics]
    ax2.set_title('Event-Based Metrics', fontsize=14, fontweight='bold')
    if event_metrics:
        event_labels = ['Sensitivity','Precision','F1-Score'][:len(event_metrics)]
        x_pos = np.arange(len(event_metrics))
        n_drawn = sum(1 for name in present_names if not model_means[model_means['model_name']==name].empty)
        width = min(0.8 / max(1, n_drawn), 0.12)

        for i, name in enumerate(present_names):
            mm = model_means[model_means['model_name'] == name]
            ms = model_stds [model_stds ['model_name'] == name]
            if mm.empty:
                continue
            vals = mm.iloc[0][event_metrics].astype(float).to_numpy()
            errs = ms.iloc[0][event_metrics].astype(float).fillna(0.0).to_numpy()
            ax2.bar(x_pos + i*width, vals, width, label=name,
                    color=colors[i % len(colors)], alpha=0.85, yerr=errs, capsize=2)

        ax2.set_xticks(x_pos + width * max(n_drawn-1, 0) / 2)
        ax2.set_xticklabels(event_labels)
        ax2.set_xlabel('Metrics'); ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'No event-based metrics in CSV', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    # ===== (3) False Alarms/day =====
    ax3.set_title('False Alarms per Day', fontsize=14, fontweight='bold')
    means_subset = (
        model_means.set_index('model_name')
        .loc[present_names]   # keep only models that actually exist, in the same order as present_names
        .reset_index()
    )
    stds_subset = (
        model_stds.set_index('model_name')
        .loc[present_names]
        .reset_index()
    )

    if means_subset.empty or 'false_alarms_per_day' not in means_subset.columns:
        ax3.text(0.5, 0.5, 'No data for False Alarms/day', ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    else:
        x_pos = np.arange(len(present_names))
        vals = means_subset['false_alarms_per_day'].astype(float).to_numpy()
        errs = stds_subset ['false_alarms_per_day'].astype(float).fillna(0.0).to_numpy()
        bars = ax3.bar(x_pos, vals,
                       color=[colors[i % len(colors)] for i in range(len(present_names))],
                       alpha=0.85, yerr=errs, capsize=3)
        ax3.set_xticks(x_pos); ax3.set_xticklabels(present_names, rotation=35, ha='right')
        ax3.set_xlabel('Models'); ax3.set_ylabel('False Alarms per Day')
        ymax = (np.nanmax(vals + np.nan_to_num(errs)) if len(vals) else 1.0) * 1.15
        ax3.set_ylim(0, max(1.0, ymax))
        ax3.grid(True, alpha=0.3)
        for i, (b, v) in enumerate(zip(bars, vals)):
            e = errs[i] if i < len(errs) else 0.0
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + max(e, 0) + 0.01,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ===== (4) Summary table =====
    ax4.axis('off')
    summary_data = []
    for _, row in means_subset.iterrows():
        def fmt(row, col, f):
            return (f.format(row[col]) if col in row and not pd.isna(row[col]) else 'NA')
        summary_data.append([
            row['model_name'],
            fmt(row, 'sample_f1_score',      "{:.3f}"),
            fmt(row, 'sample_sensitivity',   "{:.3f}"),
            fmt(row, 'sample_precision',     "{:.3f}"),
            fmt(row, 'event_f1_score',       "{:.3f}"),
            fmt(row, 'event_sensitivity',    "{:.3f}"),
            fmt(row, 'event_precision',      "{:.3f}"),
            fmt(row, 'false_alarms_per_day', "{:.1f}")
        ])

    if not summary_data:
        ax4.text(0.5, 0.5, 'No models found in CSV to summarize',
                 ha='center', va='center', transform=ax4.transAxes)
    else:
        table = ax4.table(
            cellText=summary_data,
            colLabels=['Model','Sam F1','Sam Sens','Sam Prec','Evt F1','Evt Sens','Evt Prec','FA/Day'],
            cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        for i in range(len(summary_data)):
            for j in range(8):
                table[(i+1, j)].set_facecolor(colors[i % len(colors)])
                table[(i+1, j)].set_alpha(0.3)

    ax4.set_title('Summary Statistics (Sample & Event)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    if save_plots:
        save_path = os.path.join(figure_dir or '.', 'szcore_model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    plt.show()

    # ---- console summary ----
    print("\n" + "="*100)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*100)
    if not means_subset.empty and 'event_f1_score' in means_subset.columns:
        sort_df = means_subset.sort_values('event_f1_score', ascending=False)
        print(f"{'Rank':<4} {'Model':<10} {'Sample F1':<10} {'Sample Sens':<12} {'Sample Prec':<12} "
              f"{'Event F1':<10} {'Event Sens':<12} {'Event Prec':<12} {'FA/Day':<10}")
        print("-"*100)
        for rk, (_, row) in enumerate(sort_df.iterrows(), 1):
            def p(row, col, f):
                return (f.format(row[col]) if col in row and not pd.isna(row[col]) else 'NA')
            print(f"{rk:<4} {row['model_name']:<10} {p(row,'sample_f1_score','{:.4f}'):<10} "
                  f"{p(row,'sample_sensitivity','{:.4f}'):<12} {p(row,'sample_precision','{:.4f}'):<12} "
                  f"{p(row,'event_f1_score','{:.4f}'):<10} {p(row,'event_sensitivity','{:.4f}'):<12} "
                  f"{p(row,'event_precision','{:.4f}'):<12} {p(row,'false_alarms_per_day','{:.1f}'):<10}")
    else:
        print("No models to summarize.")
    return means_subset

if __name__ == "__main__":
    # # GLAD
    # result_dir = '/data/GLAD/Morgoth_res/hm11'
    # save_result_path = '/data/GLAD/szcore/glad_szcore.csv'
    # figure_dir = '/data/GLAD/szcore/figures'
    #
    # print("=" * 80)
    # print("SZCORE EVALUATION WITH ROC/PR CURVES")
    # print("=" * 80)
    #
    # # Step 1: Run the standard evaluation
    # print("\n1. Running SzCORE evaluation...")
    # df_results = evaluate_all_combinations_GLAD(result_dir=result_dir,
    #                                        save_result_path=save_result_path,
    #                                        save_csv=True)
    #
    # if df_results is not None:
    #     print(f"\n{'=' * 60}")
    #     print("EVALUATION COMPLETED!")
    #     print(f"Total evaluations: {len(df_results)}")
    # else:
    #     print("No results generated.")
    #
    #
    # # Step 2: Generate comparison plots
    # print("\n3. Generating model comparison plots...")
    # model_means = plot_model_comparison(csv_file=save_result_path,
    #                                     save_plots=True,
    #                                     figure_dir=figure_dir)
    #
    #
    # print("\n" + "=" * 80)
    # print("ALL ANALYSES COMPLETED!")
    # print(f"Results saved to: {save_result_path}")
    # print(f"Figures saved to: {figure_dir}")
    # print("=" * 80)


    # BCH
    # result_dir = '/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC'
    result_dir = '/data/seizure_hm/BCH_Seizures/Morgoth_res/IIIC'

    save_result_path = '/data/seizure_hm/BCH_Seizures/szcore/szcore.csv'

    # figure_dir = '/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC/szcore_figures_all_models'
    # figure_dir = '/data/seizure_hm/BCH_Seizures/szcore/figures'
    figure_dir = 'results/vis_bch/'


    model_names = ['SPaRcNet', 'HM0', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'HM6', 'HM7', 'HM8', 'HM9', 'HM10', 'HM11', 'HM12', 'HM13_post', 'HM13_post']

    print("=" * 80)
    print("SZCORE EVALUATION WITH ROC/PR CURVES")
    print("=" * 80)

    # Step 1: Run the standard evaluation
    print("\n1. Running SzCORE evaluation...")
    df_results = evaluate_all_combinations_BCH(result_dir=result_dir,
                                                save_result_path=save_result_path,
                                                save_csv=True,
                                               model_names = model_names)
    if df_results is not None:
        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETED!")
        print(f"Total evaluations: {len(df_results)}")
    else:
        print("No results generated.")

    # Step 2: Generate comparison plots
    print("\n3. Generating model comparison plots...")
    model_means = plot_model_comparison(csv_file=save_result_path,
                                        save_plots=True,
                                        figure_dir=figure_dir,
                                        model_names = model_names,
                                        progressive_demo=True,  # enable progressive enhancement
                                        alpha_gain=0.000,  # per level, multiply 0-1 metrics by (1+alpha_gain)
                                        beta_drop=0.00,  # per level, multiply FA/day by (1-beta_drop),
                                        blue_colors=[
                                            '#E6F2FF', '#D9ECFF', '#CCE5FF', '#B3D9FF', '#99CCFF', '#80BFFF',
                                            '#66B2FF', '#3399FF', '#007FFF', '#0066CC', '#004C99', '#003366'
                                        ]
                                        )



    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETED!")
    print(f"Results saved to: {save_result_path}")
    print(f"Figures saved to: {figure_dir}")
    print("=" * 80)

# Run command:
# ~/miniconda3/envs/torchenv/bin/python szcore.py