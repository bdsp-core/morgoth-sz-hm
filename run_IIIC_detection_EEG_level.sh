#!/bin/bash
# =============================================================================
# run_IIIC_detection_EEG_level.sh
#
# Two-step IIIC EEG-level detection pipeline:
#
#   Step 1 — Event-level inference (1s sliding window, hm_only mode)
#             Uses the hard mining IIIC model to produce per-second predictions.
#             This is equivalent to running run_IIIC_detection_continuous_event_level.sh
#             with MODE=hm_only. If you already have event-level results in
#             EVENT_RESULT_DIR, set SKIP_EVENT_LEVEL=yes to skip this step.
#
#   Step 2 — EEG-level inference
#             Aggregates event-level predictions into EEG-level scores using
#             a separate lightweight head for each IIIC class (SEIZURE, LPD,
#             GPD, LRDA, GRDA). The input test_csv_dir is the output of Step 1.
#
# Output: one result directory per IIIC class under RESULT_DIR/
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
EEG_DIR="/your/eeg/data/dir"       # input: directory of EDF files
RESULT_DIR="/your/results/dir"     # output: result directories will be created here

# ── Skip event-level step if results already exist ────────────────────────────
# Set to "yes" if you already ran run_IIIC_detection_continuous_event_level.sh
# with MODE=hm_only and the results are in RESULT_DIR/hm_only_pred_1sStep/
SKIP_EVENT_LEVEL="no"

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

# EEG-level head checkpoints (one per IIIC class)
IIIC_DATASETS=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA")

# =============================================================================

# Event-level results directory (Step 1 output = Step 2 input)
EVENT_RESULT_DIR="${RESULT_DIR}/hm_only_pred_${slipping_step_second}sStep"

# ── Step 1: Event-level inference ─────────────────────────────────────────────
# Produces per-second IIIC predictions (Other/Seizure/LPD/GPD/LRDA/GRDA).
# Must use hm_only mode (dataset=IIIC, no refer_hm_model) so that the output
# CSV format matches what EEG_level_head.py expects.
if [ "$SKIP_EVENT_LEVEL" = "yes" ]; then
    echo "Skipping Step 1 — using existing event-level results in: ${EVENT_RESULT_DIR}"
else
    echo "Step 1: Running event-level inference → ${EVENT_RESULT_DIR}"
    echo "$password" | sudo -S OMP_NUM_THREADS=1 "$(which python)" \
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
            --eval_results_dir ${EVENT_RESULT_DIR} \
            --prediction_slipping_step_second ${slipping_step_second} \
            --rewrite_results ${rewrite_results}
fi

# ── Step 2: EEG-level inference ───────────────────────────────────────────────
# Aggregates the per-second event-level CSVs into EEG-level scores.
# Runs separately for each IIIC class using its own trained head.
echo "Step 2: Running EEG-level inference for all IIIC classes..."
for IIIC_DATASET in "${IIIC_DATASETS[@]}"; do
    echo "  → ${IIIC_DATASET}"
    echo "$password" | sudo -S "$(which python)" EEG_level_head.py \
        --mode predict \
        --dataset ${IIIC_DATASET} \
        --task_model checkpoints/${IIIC_DATASET}_EEGlevel_2.pth \
        --test_csv_dir ${EVENT_RESULT_DIR} \
        --result_dir ${RESULT_DIR}/${IIIC_DATASET}_EEGlevel
done

echo "Done. EEG-level results saved to ${RESULT_DIR}/<class>_EEGlevel/"
