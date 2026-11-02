from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm

def merge_csv_if_rows_equal(hm_result_dir, fsl_result_dir, out_dir, index_col=None, rewrite=False):
    dir1 = Path(hm_result_dir)
    dir2 = Path(fsl_result_dir)
    out_dir = Path(out_dir)
    if rewrite and out_dir.exists():
        print(f"[REWRITE] removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files1 = {p.name: p for p in dir1.glob("*.csv")}
    files2 = {p.name: p for p in dir2.glob("*.csv")}

    common_files = sorted(files1.keys() & files2.keys())
    if not common_files:
        print("No common CSV files found.")
        return []

    # rename columns for the SECOND csv (fsl_result_dir)
    rename_map = {
        "class_0_prob": "hm_other_0_prob",
        "class_1_prob": "hm_sz_1_prob",
        "class_2_prob": "hm_lpd_2_prob",
        "class_3_prob": "hm_gpd_3_prob",
        "class_4_prob": "hm_lrda_4_prob",
        "class_5_prob": "hm_grda_5_prob",
        "class_6_prob": "hm_chewing_6_prob",
        "pred_class":  "hm_pred_class",
    }

    hm_prob_cols = [
        "hm_other_0_prob",
        "hm_sz_1_prob",
        "hm_lpd_2_prob",
        "hm_gpd_3_prob",
        "hm_lrda_4_prob",
        "hm_grda_5_prob",
        "hm_chewing_6_prob",
    ]

    mismatched_files = []

    for fname in tqdm(common_files):
        f1 = files1[fname]
        f2 = files2[fname]

        df1 = pd.read_csv(f1, index_col=index_col)

        if not (df1["pred_class"] == 1).any():
            print(f"[SKIP] {fname} (no pred_class == 1)")
            continue

        df2 = pd.read_csv(f2, index_col=index_col)

        # check required columns exist in second csv before renaming
        missing_cols = sorted(set(rename_map.keys()) - set(df2.columns))
        if missing_cols:
            print(f"[MISMATCH] {fname} (missing columns in second CSV: {missing_cols})")
            print(f"  hm:  {f1}")
            print(f"  fsl: {f2}")
            mismatched_files.append(fname)
            continue

        # rename second csv columns first
        df2 = df2.rename(columns=rename_map)

        # NOTE: df1.equals(df2) requires identical columns too; here we check row/index equality only
        if (df1.shape[0] != df2.shape[0]) or (not df1.index.equals(df2.index)):
            print(f"[MISMATCH] {fname} (row/index mismatch)")
            print(f"  hm:  {f1}  shape={df1.shape}")
            print(f"  fsl: {f2}  shape={df2.shape}")
            mismatched_files.append(fname)
            continue

        # merge columns
        merged = pd.concat([df1, df2], axis=1)

        # post-processing:
        # if pred_class != 1, set hm_*_prob to NaN and set hm_pred_class = pred_class
        if "pred_class" not in merged.columns:
            print(f"[MISMATCH] {fname} (missing pred_class in merged)")
            print(f"  hm:  {f1}")
            print(f"  fsl: {f2}")
            mismatched_files.append(fname)
            continue

        if "hm_pred_class" not in merged.columns:
            print(f"[MISMATCH] {fname} (missing hm_pred_class in merged)")
            print(f"  hm:  {f1}")
            print(f"  fsl: {f2}")
            mismatched_files.append(fname)
            continue

        # ensure hm prob columns exist; if not, treat as mismatch
        missing_hm_cols = [c for c in hm_prob_cols if c not in merged.columns]
        if missing_hm_cols:
            print(f"[MISMATCH] {fname} (missing HM prob cols in merged: {missing_hm_cols})")
            print(f"  hm:  {f1}")
            print(f"  fsl: {f2}")
            mismatched_files.append(fname)
            continue

        mask = merged["pred_class"] != 1
        merged.loc[mask, hm_prob_cols] = np.nan
        merged.loc[mask, "hm_pred_class"] = merged.loc[mask, "pred_class"]

        merged.to_csv(out_dir / fname, index=(index_col is not None))
        print(f"[OK] merged {fname}")

    print("\n===== SUMMARY =====")
    if mismatched_files:
        print(f"Mismatched files ({len(mismatched_files)}):")
        for f in mismatched_files:
            print(f"  - {f}")
    else:
        print("All files matched successfully.")

    print(mismatched_files)

if __name__ == "__main__":


    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm5/hm_IIIC5_I0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0002_pred_1s_sz",
    #     out_dir="/data/representative/hm5/hm_IIIC5_I0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm5/hm_IIIC5_I0003_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0003_pred_1s_sz",
    #     out_dir="/data/representative/hm5/hm_IIIC5_I0003_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm5/hm_IIIC5_S0001_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0001_pred_1s_sz",
    #     out_dir="/data/representative/hm5/hm_IIIC5_S0001_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    #
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm5/hm_IIIC5_fsl_S0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0002_pred_1s_sz",
    #     out_dir="/data/representative/hm5/hm_IIIC5_fsl_S0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )



    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm11/hm_IIIC11_fsl_I0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0002_pred_1s_sz",
    #     out_dir="/data/representative/hm11/hm_IIIC11_fsl_I0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm11/hm_IIIC11_fsl_I0003_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0003_pred_1s_sz",
    #     out_dir="/data/representative/hm11/hm_IIIC11_fsl_I0003_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm11/hm_IIIC11_fsl_S0001_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0001_pred_1s_sz",
    #     out_dir="/data/representative/hm11/hm_IIIC11_fsl_S0001_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    #
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm11/hm_IIIC11_fsl_S0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0002_pred_1s_sz",
    #     out_dir="/data/representative/hm11/hm_IIIC11_fsl_S0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_S0001_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0001_pred_1s_sz",
    #     out_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_S0001_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    #
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_S0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0002_pred_1s_sz",
    #     out_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_S0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_I0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0002_pred_1s_sz",
    #     out_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_I0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_I0003_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0003_pred_1s_sz",
    #     out_dir="/data/representative/hm12_1/hm_IIIC12_1_fsl_I0003_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    #
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm13/hm_IIIC13_fsl_S0001_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0001_pred_1s_sz",
    #     out_dir="/data/representative/hm13/hm_IIIC13_fsl_S0001_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    #
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm13/hm_IIIC13_fsl_S0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0002_pred_1s_sz",
    #     out_dir="/data/representative/hm13/hm_IIIC13_fsl_S0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )

    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm13/hm_IIIC13_fsl_I0002_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0002_pred_1s_sz",
    #     out_dir="/data/representative/hm13/hm_IIIC13_fsl_I0002_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    # merge_csv_if_rows_equal(
    #     hm_result_dir="/data/representative/hm13/hm_IIIC13_fsl_I0003_pred_1s_sz/original_results",
    #     fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0003_pred_1s_sz",
    #     out_dir="/data/representative/hm13/hm_IIIC13_fsl_I0003_pred_1s_sz/revised_results_fsl3",
    #     index_col=None  # if you have an explicit index, you can pass a column name or column number
    # )
    #
    merge_csv_if_rows_equal(
        hm_result_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_S0001_pred_1s_sz/original_results",
        fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0001_pred_1s_sz",
        out_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_S0001_pred_1s_sz/revised_results_fsl3",
        index_col=None  # if you have an explicit index, you can pass a column name or column number

    )

    merge_csv_if_rows_equal(
        hm_result_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_S0002_pred_1s_sz/original_results",
        fsl_result_dir="/data/representative/IIIC_chewing3_results_1/S0002_pred_1s_sz",
        out_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_S0002_pred_1s_sz/revised_results_fsl3",
        index_col=None  # if you have an explicit index, you can pass a column name or column number
    )

    merge_csv_if_rows_equal(
        hm_result_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_I0002_pred_1s_sz/original_results",
        fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0002_pred_1s_sz",
        out_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_I0002_pred_1s_sz/revised_results_fsl3",
        index_col=None  # if you have an explicit index, you can pass a column name or column number
    )


    merge_csv_if_rows_equal(
        hm_result_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_I0003_pred_1s_sz/original_results",
        fsl_result_dir="/data/representative/IIIC_chewing3_results_1/I0003_pred_1s_sz",
        out_dir="/data/representative/hm13_2/hm_IIIC13_2_fsl_I0003_pred_1s_sz/revised_results_fsl3",
        index_col=None  # if you have an explicit index, you can pass a column name or column number
    )





# sudo ~/miniconda3/envs/torchenv/bin/python combine_hm_fsl_results.py