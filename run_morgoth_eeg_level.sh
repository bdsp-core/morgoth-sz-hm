#!/bin/bash

# Steps:
#   First, perform continuous 1-second step event-level prediction;
#   Second, perform EEG_level prediction based on outputs from first step

# The event-level output results (--eval_results_dir) are the EEG_level inputs (--test_csv_dir)
# If you already have event-level outputs, you can skip step 1, format them as 1s-step event-level prediction to input to the step 2
# for spike, gen_spike and focal_spike, the --result_dir should be the same




######################################## IF USING GPU #######################################
password="exxact@1"

# edf raw case
dataset_dir="test_data/edf"
data_format="edf"
sampling_rate=0
polarity=-1
result_dir="test_data/edf_results"
already_format_channel_order='no'
already_average_montage='no'
allow_missing_channels='no'
max_length_hour='no'
rewrite_results='no'
need_spikes_1s_result='yes'


## mat raw case
#dataset_dir="test_data/mat"
#data_format="mat"
#sampling_rate=0
#polarity=-1
#result_dir="test_data/mat_results"
#already_format_channel_order='no'
#already_average_montage='no'
#allow_missing_channels='no'
#max_length_hour='no'
#rewrite_results='no'
#need_spikes_1s_result='yes'


#  Normal:
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/NORMAL.pth \
            --abs_pos_emb \
            --dataset NORMAL \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_NORMAL_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}

    # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
        --mode predict \
        --dataset NORMAL \
        --task_model checkpoints/morgoth/NORMAL_EEGlevel.pth \
        --test_csv_dir ${result_dir}/pred_NORMAL_1sStep\
        --result_dir ${result_dir}


#  Slowing:

    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=2 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SLOWING.pth \
            --abs_pos_emb \
            --dataset SLOWING \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order}  \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLOWING_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}

    # (2). EEG_level prediction (FOC_SLOWING and GEN_SLOWING are seperated)
SLOWING_datasets=("FOC_SLOWING" "GEN_SLOWING")
for SLOWING_dataset in "${SLOWING_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${SLOWING_dataset} \
            --task_model checkpoints/morgoth/${SLOWING_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_SLOWING_1sStep \
            --result_dir ${result_dir}
done


# BS
  # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=2 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/BS.pth \
            --abs_pos_emb \
            --dataset BS \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_BS_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


  # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
          --mode predict \
          --dataset BS \
          --task_model checkpoints/morgoth/BS_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_BS_1sStep  \
          --result_dir ${result_dir}



# FOC GEN SPIKES:
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=3 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/FOCGENSPIKES.pth \
            --abs_pos_emb \
            --dataset FOC_GEN_SPIKES \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --polarity ${polarity} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir  ${result_dir}/pred_FOCGENSPIKES_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


    # (2). EEG_level prediction (FOC_SPIKES and GEN_SPIKES are seperated)
FOC_GEN_SPIKES_datasets=("FOC_SPIKES" "GEN_SPIKES")
for FOC_GEN_SPIKES_dataset in "${FOC_GEN_SPIKES_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${FOC_GEN_SPIKES_dataset} \
            --task_model checkpoints/morgoth/${FOC_GEN_SPIKES_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_FOCGENSPIKES_1sStep \
            --result_dir ${result_dir}
done

# Spike
  # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=2 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SPIKES.pth \
            --abs_pos_emb \
            --dataset SPIKES \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --polarity ${polarity} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SPIKES_2pStep \
            --prediction_slipping_step 2 \
            --need_spikes_1s_result ${need_spikes_1s_result} \
            --rewrite_results ${rewrite_results}

  # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
          --mode predict \
          --dataset SPIKES \
          --task_model checkpoints/morgoth/SPIKES_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_SPIKES_1sStep  \
          --result_dir ${result_dir} \
          --align_spike_detection_and_location


# IIIC
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=4 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/IIIC_sz_hm13_1.pth \
            --abs_pos_emb \
            --dataset IIIC \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_IIIC_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}



    # (2). EEG_level prediction (SEIZURE, LPD, GPD, LRDA, GRDA are seperated)
IIIC_datasets=("SEIZURE" "LPD" "GPD") #"LRDA" "GRDA"
for IIIC_dataset in "${IIIC_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${IIIC_dataset} \
            --task_model checkpoints/morgoth/${IIIC_dataset}_EEGlevel_2.pth \
            --test_csv_dir ${result_dir}/pred_IIIC_1sStep \
            --result_dir ${result_dir}

done




# Sleep 5 stage
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=4 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SLEEPPSG.pth \
            --abs_pos_emb \
            --dataset SLEEPPSG \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEPPSG_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


echo password | sudo -S $(which python)  EEG_level_head.py \
        --mode predict \
        --dataset SLEEPPSG \
        --test_csv_dir  ${result_dir}/pred_SLEEPPSG_1sStep \
        --result_dir  ${result_dir}


# Sleep 3 stage
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=4 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SLEEP.pth \
            --abs_pos_emb \
            --dataset MGBSLEEP3stages \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEP3stages_1sStep \
            --prediction_slipping_step_second 1


echo "$password" | sudo -S $(which python)  EEG_level_head.py \
        --mode predict \
        --dataset SLEEP3stages \
        --test_csv_dir  ${result_dir}/pred_SLEEP3stages_1sStep \
        --result_dir  ${result_dir}





