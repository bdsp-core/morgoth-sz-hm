import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import ast

def get_result_file_sparcnettest(label_file,eeglevel_dir,output_csv):
    # Load the main table
    main_df = pd.read_csv(label_file)

    main_df=main_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")

    def decide_validation(label):
        if isinstance(label, list) and len(label) > 0:
            return "no" if label[0] == max(label) else "yes"
        return "no"  # Prevent edge case exceptions

    main_df["label ([other,seizure,lpd,gpd,lrda,grda])"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    main_df["used_for_validation"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(decide_validation)
    main_df["source"] = "sparcnettest"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")


def get_result_file_moe(label_file,eeglevel_dir,output_csv):
    # Load the main table
    label_df = pd.read_csv(label_file)
    label_df = label_df.rename(columns={
        "event": "file_name",
    })


    # Find all columns starting with "expert"
    expert_cols = [col for col in label_df.columns if col.startswith("expert")]

    print(f"Found {len(expert_cols)} expert columns")

    # Count occurrences of 0~5 for each row
    def count_labels(row):
        counts = [0] * 6  # Corresponding to 0~5
        for val in row:
            if pd.notna(val):
                val = int(val)
                if 0 <= val <= 5:
                    counts[val] += 1
        return counts

    # Generate the label column
    label_df["label ([other,seizure,lpd,gpd,lrda,grda])"] = label_df[expert_cols].apply(count_labels, axis=1)

    # Merge back into main_df (only file_name + label)
    main_df = label_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")

    def decide_validation(label):
        if isinstance(label, list) and len(label) > 0:
            return "no" if label[0] == max(label) else "yes"
        return "no"  # Prevent edge case exceptions

    main_df["used_for_validation"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(decide_validation)
    main_df["source"] = "moe"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")


def get_result_file_representative(label_file,eeglevel_dir,output_csv):
    # Load the main table

    label_df = pd.read_csv(label_file)
    label_df = label_df[label_df["source"] == "representative"].copy()

    def to_one_hot(x):
        vec = [0] * 6
        if pd.notna(x):
            x = int(x)
            if 0 <= x <= 5:
                vec[x] = 1
        return vec

    label_df["label ([other,seizure,lpd,gpd,lrda,grda])"]= label_df["label"].apply(to_one_hot)

    # Merge back into main_df (only file_name + label)
    main_df = label_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")


    main_df["used_for_validation"] = "yes"
    main_df["source"] = "representative"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")

def merge_expert_labels_from_iiic(representative_csv, iiic_labels_csv, output_csv):
    """
    Join expert segment-level labels from IIICOutput into the representative EEG-level CSV
    by file_name (1-to-1 match). Unmatched rows will have NaN for expert columns.
    """
    rep_df = pd.read_csv(representative_csv)
    iiic_df = pd.read_csv(iiic_labels_csv)

    non_expert = {'bdsp_mrn', 'file_name', 'SPaRCNet', 'Morgoth', 'majority', 'votes'}
    expert_cols = [c for c in iiic_df.columns
                   if c not in non_expert
                   and not any(c.startswith(p) for p in
                               ('SZHM_', 'M_class', 'K_class', 'S_class',
                                'M_pred', 'K_pred', 'S_pred'))]

    expert_df = iiic_df[['file_name'] + expert_cols].copy()
    merged = rep_df.merge(expert_df, on='file_name', how='left')

    matched = merged[expert_cols[0]].notna().sum()
    print(f"Matched {matched}/{len(merged)} rows with expert labels ({len(expert_cols)} experts)")

    merged.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return output_csv, expert_cols


def get_result_file(result_files, output_csv):
    dfs = []

    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded: {file}, shape={df.shape}")
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    if len(dfs) == 0:
        print("No valid files found.")
        return

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}, shape={merged_df.shape}")



def plot_roc_pr(result_csv_path, save_path, n_bootstrap=1000, source_filter=None, show_experts=True):
    tasks = ["SEIZURE", "LPD", "GPD", "LRDA", "GRDA"]
    task_idx = {"SEIZURE": 1, "LPD": 2, "GPD": 3, "LRDA": 4, "GRDA": 5}
    label_col = "label ([other,seizure,lpd,gpd,lrda,grda])"
    expert_color = "grey"

    plt.rcParams.update({"font.size": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})

    df = pd.read_csv(result_csv_path)
    df[label_col] = df[label_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    if source_filter is not None:
        df = df[df["source"] == source_filter].copy()
        print(f"Filtered to source='{source_filter}', size: {len(df)}")

    df = df[df["used_for_validation"] == "yes"].copy()
    print("Data size:", len(df))

    # detect expert columns (numeric, not model/label columns)
    reserved = {label_col, "file_name", "used_for_validation", "source"}
    expert_cols = [c for c in df.columns
                   if c not in reserved
                   and not any(c.endswith(s) for s in
                               ("_eeglevel_probability", "_eeglevel_confidence"))]

    def is_target(label, i):
        return int(label[i] == max(label))

    def bootstrap_curve(y_true, y_score, curve_type, n_bootstrap):
        xs = np.linspace(0, 1, 200)
        curves, aucs = [], []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx_b = np.random.choice(n, n, replace=True)
            yt, ys = y_true[idx_b], y_score[idx_b]
            if len(np.unique(yt)) < 2:
                continue
            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(yt, ys)
                interp = np.interp(xs, fpr, tpr)
                interp[0], interp[-1] = 0.0, 1.0
                curves.append(interp)
                aucs.append(auc(fpr, tpr))
            else:
                prec, rec, _ = precision_recall_curve(yt, ys)
                interp = np.interp(xs, rec[::-1], prec[::-1])
                curves.append(interp)
                aucs.append(auc(rec, prec))
        if not curves:
            return xs, None, None, None, np.nan
        arr = np.array(curves)
        return xs, arr.mean(0), np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0), float(np.mean(aucs))

    def _expert_op_roc(y_true, e_labels):
        """Compute (fpr, tpr) for one expert on one sample."""
        valid = ~np.isnan(e_labels)
        if valid.sum() < 5:
            return np.nan, np.nan
        yt = y_true[valid]
        ep = (e_labels[valid] > 0.5).astype(int)
        tp = int(((ep == 1) & (yt == 1)).sum())
        fp = int(((ep == 1) & (yt == 0)).sum())
        fn = int(((ep == 0) & (yt == 1)).sum())
        tn = int(((ep == 0) & (yt == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return fpr, tpr

    def _expert_op_pr(y_true, e_labels):
        """Compute (recall, precision) for one expert on one sample."""
        valid = ~np.isnan(e_labels)
        if valid.sum() < 5:
            return np.nan, np.nan
        yt = y_true[valid]
        ep = (e_labels[valid] > 0.5).astype(int)
        tp = int(((ep == 1) & (yt == 1)).sum())
        fp = int(((ep == 1) & (yt == 0)).sum())
        fn = int(((ep == 0) & (yt == 1)).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        return rec, prec

    def bootstrap_expert_crosses(y_true, expert_bin, mode, n_bootstrap):
        """
        Bootstrap per-expert operating points and return cross parameters
        (median ± 2.5/97.5 percentile) for drawing error-bar crosses.
        mode: "roc" -> (fpr, tpr),  "pr" -> (recall, precision)
        Returns list of dicts with keys: median_x, median_y, x_lo, x_hi, y_lo, y_hi
        """
        n = len(y_true)
        per_expert = [[] for _ in expert_bin]
        op_fn = _expert_op_roc if mode == "roc" else _expert_op_pr

        for _ in range(n_bootstrap):
            idx_b = np.random.choice(n, n, replace=True)
            yt_b = y_true[idx_b]
            for ei, e_labels in enumerate(expert_bin):
                x, y = op_fn(yt_b, e_labels[idx_b])
                if not (np.isnan(x) or np.isnan(y)):
                    per_expert[ei].append((x, y))

        crosses = []
        for pts in per_expert:
            if len(pts) < 5:
                continue
            xs_b = [p[0] for p in pts]
            ys_b = [p[1] for p in pts]
            crosses.append({
                'median_x': float(np.median(xs_b)),
                'median_y': float(np.median(ys_b)),
                'x_lo': float(np.percentile(xs_b, 2.5)),
                'x_hi': float(np.percentile(xs_b, 97.5)),
                'y_lo': float(np.percentile(ys_b, 2.5)),
                'y_hi': float(np.percentile(ys_b, 97.5)),
            })
        return crosses

    def draw_expert_crosses(ax, crosses, color, alpha=0.5, lw=1):
        """Draw cross (vertical + horizontal line) for each expert."""
        for c in crosses:
            ax.plot([c['median_x'], c['median_x']], [c['y_lo'], c['y_hi']],
                    color=color, alpha=alpha, linewidth=lw)
            ax.plot([c['x_lo'], c['x_hi']], [c['median_y'], c['median_y']],
                    color=color, alpha=alpha, linewidth=lw)

    def compute_euc(mean_x, mean_y, crosses):
        """Fraction of expert median points below the mean curve."""
        if not crosses or mean_y is None:
            return np.nan
        below = sum(
            1 for c in crosses
            if float(np.interp(c['median_x'], mean_x, mean_y)) > c['median_y']
        )
        return 100.0 * below / len(crosses)

    plt.figure(figsize=(4 * len(tasks), 8))

    for col_idx, task in enumerate(tasks):
        cls_idx = task_idx[task]
        y_true = np.array([is_target(lbl, cls_idx) for lbl in df[label_col]])

        prob_col = f"{task}_eeglevel_probability"
        conf_col = f"{task}_eeglevel_confidence"

        y_prob = pd.to_numeric(df[prob_col], errors="coerce").values if prob_col in df.columns else np.full(len(df), np.nan)
        y_conf = pd.to_numeric(df[conf_col], errors="coerce").values if conf_col in df.columns else np.full(len(df), np.nan)

        mask_prob = ~np.isnan(y_prob)
        mask_conf = ~np.isnan(y_conf)
        if mask_prob.sum() == 0 and mask_conf.sum() == 0:
            continue

        # build expert binary labels and bootstrap crosses (only when show_experts=True)
        if show_experts and expert_cols:
            expert_bin = []
            for ec in expert_cols:
                raw = pd.to_numeric(df[ec], errors="coerce").values
                e_bin = np.where(np.isnan(raw), np.nan, (raw == cls_idx).astype(float))
                expert_bin.append(e_bin)
            crosses_roc = bootstrap_expert_crosses(y_true, expert_bin, "roc", n_bootstrap)
            crosses_pr  = bootstrap_expert_crosses(y_true, expert_bin, "pr",  n_bootstrap)
        else:
            crosses_roc, crosses_pr = [], []

        # Probability-2 (confidence, dark red solid) on top; Probability-1 (probability, black solid) below
        score_specs = [
            (y_conf, mask_conf, "darkred", "-", "Probability-2"),
            (y_prob, mask_prob, "black",   "-", "Probability-1"),
        ]

        # ===== ROC =====
        ax_roc = plt.subplot(2, len(tasks), col_idx + 1)
        for y_score, mask, color, ls, label_prefix in score_specs:
            if mask.sum() == 0:
                continue
            xs, mean_y, low, up, auc_mean = bootstrap_curve(y_true[mask], y_score[mask], "roc", n_bootstrap)
            euc_roc = compute_euc(xs, mean_y, crosses_roc)
            if mean_y is not None:
                euc_str = f"\nEUC={euc_roc:.1f}%" if not np.isnan(euc_roc) else ""
                plt.plot(xs, mean_y, color=color, lw=2, linestyle=ls,
                         label=f"  {label_prefix}\nAUC={auc_mean:.3f}{euc_str}")
                plt.fill_between(xs, low, up, color=color, alpha=0.3)
        if crosses_roc:
            draw_expert_crosses(ax_roc, crosses_roc, expert_color)
            plt.scatter([c['median_x'] for c in crosses_roc],
                        [c['median_y'] for c in crosses_roc],
                        marker="o", color=expert_color, alpha=0.6, s=20)
        plt.xlabel('')
        plt.ylabel('')
        if col_idx != 0:
            ax_roc.set_yticklabels([])
        ax_roc.set_xticklabels([])
        leg = plt.legend(loc="lower right", handlelength=0, handletextpad=0,
                         fontsize=14, frameon=False,
                         title=f"{task}\nEEG-level Head", title_fontsize=15)
        leg.get_title().set_ha("center")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        # ===== PR =====
        ax_pr = plt.subplot(2, len(tasks), len(tasks) + col_idx + 1)
        for y_score, mask, color, ls, label_prefix in score_specs:
            if mask.sum() == 0:
                continue
            xs, mean_y, low, up, auc_mean = bootstrap_curve(y_true[mask], y_score[mask], "pr", n_bootstrap)
            euc_pr = compute_euc(xs, mean_y, crosses_pr)
            if mean_y is not None:
                euc_str = f"\nEUC={euc_pr:.1f}%" if not np.isnan(euc_pr) else ""
                plt.plot(xs, mean_y, color=color, lw=2, linestyle=ls,
                         label=f"  {label_prefix}\nAUC={auc_mean:.3f}{euc_str}")
                plt.fill_between(xs, low, up, color=color, alpha=0.3)
        if crosses_pr:
            draw_expert_crosses(ax_pr, crosses_pr, expert_color)
            plt.scatter([c['median_x'] for c in crosses_pr],
                        [c['median_y'] for c in crosses_pr],
                        marker="o", color=expert_color, alpha=0.6, s=20)
        plt.xlabel('')
        plt.ylabel('')
        if col_idx != 0:
            ax_pr.set_yticklabels([])
        leg = plt.legend(loc="lower left", handlelength=0, handletextpad=0,
                         fontsize=14, frameon=False,
                         title=f"{task}\nEEG-level Head", title_fontsize=15)
        leg.get_title().set_ha("center")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sparcnet_csv   = os.path.join(RESULTS_DIR, "sparcnet_testset_eeglevel_results.csv")
    moe_csv        = os.path.join(RESULTS_DIR, "moe_eeglevel_results.csv")
    represent_csv  = os.path.join(RESULTS_DIR, "representative_eeglevel_results.csv")
    merged_csv     = os.path.join(RESULTS_DIR, "eeglevel_results.csv")
    iiic_csv       = os.path.join(RESULTS_DIR, "IIICOutput_SZHM_Morgoth_SPaRCNet_KaggleWinner_30experts.csv")

    get_result_file_sparcnettest(
        label_file="/data/IIIC/sparcnet_test_list.csv",
        eeglevel_dir="/data/IIIC_EEG_level/event_res/IIIC_segments_raw_EEGlevel",
        output_csv=sparcnet_csv,
    )

    # Enrich sparcnet CSV with per-expert segment labels (for expert scatter + EUC)
    merge_expert_labels_from_iiic(
        representative_csv=sparcnet_csv,
        iiic_labels_csv=iiic_csv,
        output_csv=sparcnet_csv,
    )

    get_result_file_moe(
        label_file="/data/MoE/labels/IIIC_merged_experts.csv",
        eeglevel_dir="/data/IIIC_EEG_level/event_res/MoE_segments_raw_EEGlevel",
        output_csv=moe_csv,
    )

    get_result_file_representative(
        label_file="/data/IIIC_EEG_level/event_res/training_list.csv",
        eeglevel_dir="/data/IIIC_EEG_level/event_res/representative_res_EEGlevel",
        output_csv=represent_csv,
    )

    get_result_file(
        result_files=[sparcnet_csv, moe_csv, represent_csv],
        output_csv=merged_csv,
    )

    plot_roc_pr(result_csv_path=merged_csv,
                save_path=os.path.join(RESULTS_DIR, "eeglevel_results.pdf"),
                n_bootstrap=100, show_experts=False)

    plot_roc_pr(result_csv_path=sparcnet_csv,
                save_path=os.path.join(RESULTS_DIR, "eeglevel_results_sparcnettest.pdf"),
                n_bootstrap=100)

    # plot_roc_pr(result_csv_path=represent_csv,
    #             save_path=os.path.join(RESULTS_DIR, "eeglevel_results_representative.png"),
    #             n_bootstrap=100)

    # plot_roc_pr(result_csv_path=merged_csv,
    #             save_path=os.path.join(RESULTS_DIR, "eeglevel_results_moe.png"),
    #             n_bootstrap=100, source_filter="moe")

# echo "exxact@1" | sudo -S ~/miniconda3/envs/torchenv/bin/python vis_IIIC_EEG_level_results.py