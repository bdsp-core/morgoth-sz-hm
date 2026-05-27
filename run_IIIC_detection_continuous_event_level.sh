#!/bin/bash
# =============================================================================
# run_IIIC_detection_continuous_event_level.sh
#
# IIIC seizure detection with three modes:
#
#   MODE=hm_only      — Hard mining IIIC model only
#                       Output: Other / Seizure / LPD / GPD / LRDA / GRDA (6 classes)
#
#   MODE=chewing_only — Few-shot chewing artifact model only
#                       Useful for evaluating chewing artifact rejection in isolation
#
#   MODE=combined     — HM model + chewing artifact rejection (refer_hm_model)
#                       Recommended for clinical use; chewing artifact windows are
#                       suppressed before IIIC classification
#
# Set MODE below, then set EEG_DIR and RESULT_DIR for your data.
# =============================================================================

# ── Mode selector ─────────────────────────────────────────────────────────────
# Options: hm_only | chewing_only | combined
MODE="combined"

# ── Paths ─────────────────────────────────────────────────────────────────────
EEG_DIR="/your/eeg/data/dir"       # input: directory of EDF files
RESULT_DIR="/your/results/dir"     # output: per-subject result folders

# ── Shared parameters ─────────────────────────────────────────────────────────
password=""
data_format="edf"
sampling_rate=0
already_format_channel_order="no"
already_average_montage="no"
allow_missing_channels="no"
max_length_hour="no"
rewrite_results="no"
slipping_step_second=1

# ── Checkpoints ───────────────────────────────────────────────────────────────
HM_MODEL="checkpoints/IIIC_sz_hm13_1.pth"
CHEWING_MODEL="checkpoints/IIIC_sz_hm13_3_fs.pth"

# =============================================================================

if [ "$MODE" = "hm_only" ]; then
    # ── Mode 1: Hard mining IIIC model only ───────────────────────────────────
    # Output classes: Other / Seizure / LPD / GPD / LRDA / GRDA
    # Uses standard IIIC dataset pipeline (no chewing artifact rejection)
    echo "Mode: hm_only  |  model: ${HM_MODEL}"
    echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) \
        -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 \
        finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model ${HM_MODEL} \
            --dataset IIIC \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${EEG_DIR} \
            --eval_results_dir ${RESULT_DIR}/hm_only_pred_${slipping_step_second}sStep \
            --prediction_slipping_step_second ${slipping_step_second} \
            --rewrite_results ${rewrite_results}

elif [ "$MODE" = "chewing_only" ]; then
    # ── Mode 2: Few-shot chewing artifact model only ───────────────────────────
    # Useful to inspect chewing artifact rejection scores in isolation
    # before combining with the HM IIIC model
    echo "Mode: chewing_only  |  model: ${CHEWING_MODEL}"
    echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) \
        -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 \
        finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model ${CHEWING_MODEL} \
            --dataset IIIC_chewing \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${EEG_DIR} \
            --eval_results_dir ${RESULT_DIR}/chewing_only_pred_${slipping_step_second}sStep \
            --prediction_slipping_step_second ${slipping_step_second} \
            --rewrite_results ${rewrite_results}

elif [ "$MODE" = "combined" ]; then
    # ── Mode 3: HM model + chewing artifact rejection (recommended) ────────────
    # refer_hm_model runs the few-shot chewing classifier; windows identified as
    # chewing artifact are suppressed before IIIC classification.
    # Output classes: Other / Seizure / LPD / GPD / LRDA / GRDA
    echo "Mode: combined  |  HM model: ${HM_MODEL}  |  chewing model: ${CHEWING_MODEL}"
    echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) \
        -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 \
        finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model ${HM_MODEL} \
            --refer_hm_model ${CHEWING_MODEL} \
            --dataset IIIC_hm \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${EEG_DIR} \
            --eval_results_dir ${RESULT_DIR}/combined_pred_${slipping_step_second}sStep \
            --prediction_slipping_step_second ${slipping_step_second} \
            --rewrite_results ${rewrite_results}

else
    echo "[ERROR] Unknown MODE: '${MODE}'. Choose from: hm_only | chewing_only | combined"
    exit 1
fi
