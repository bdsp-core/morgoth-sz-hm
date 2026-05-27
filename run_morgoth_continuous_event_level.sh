#!/bin/bash

## Required parameters
#--dataset IIIC or SPIKES or FOC_GEN_SPIKES or BS or SLOWING or NORMAL or SLEEPPSG or MGBSLEEP3stages
#--data_format edf or mat
#--eval_sub_dir /xxx/xxx/ (input data dir)
#--eval_results_dir /xxx/xxx/ (output result dir)
#--prediction_slipping_step xxx (Step size in points; if original hz>200 prediction_slipping_step better to be 100 or 128)
#  or use --prediction_slipping_step_second xxx (step size in seconds)

##### Optional parameters: If the original data contains channel names and sampling rate information, the following parameters can be omitted from the command.
#--sampling_rate 0 or xxx (If the raw data does not contain this information, it should be assigned here; 0 indicates that the information is present in the data.)
#--already_format_channel_order yes (If the data does not include channel information, it needs to be sorted as required before being input.)
#--already_average_montage yes (If the data has already been average montaged, it should be specified.)
#--allow_missing_channels yes or no (If the data does not include all 19 channels, processing is still allowed — the missing channels will be zero-filled.)

##### Optional parameters: For 1-second spike detection
#--smooth_result ema or window_ema or ''
#--need_spikes_10s_result yes (summarize 10-second results from 1-second predictions.)
#--spikes_10s_result_slipping_step_second xx (sliding step in second for 10-second spike detection)

#--rewrite_results Overwrite the original results when new results are available.

password=""

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

need_spikes_1s_result='no'
need_vw_1s_result='no'
need_spike_localization_1s_result="no"

smooth_spike_result='ema' # if not use, set ''
smooth_vw_result='ema' # if not use, set ''
smooth_spike_localization_result='ema' # if not use, set ''

PD_detection_location='both' # right | left | both

# 1. IIIC-------------------------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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

# IIIC with fs model for chewing (add refer_hm_model)
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/IIIC_sz_hm13_1.pth  \
            --refer_hm_model checkpoints/morgoth/IIIC_sz_hm13_3_fs.pth \
            --abs_pos_emb \
            --dataset IIIC_hm \
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



# 2. SPIKES--------------------------------------------------------------------------

echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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
            --smooth_result ${smooth_spike_result} \
            --rewrite_results ${rewrite_results}


# 3. Focal/Generalized Spikes--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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


# 4. Slowing--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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

# 5. BS--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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


# 6. NORMAL--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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

# 7. SLEEP 3 stages with 19 channels --------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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
            --eval_results_dir ${result_dir}/pred_SLEEPTAGING3class_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


# 8. SLEEP 5 stages with 6 channels --------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
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
            --eval_results_dir ${result_dir}/pred_SLEEPSTAGING5class_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


# 9. Vertex wave --------------------------------------------------------
# option 1: run vw model and use spike results (should run spike model first, see 2.)
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
              --predict \
              --model base_patch200_200 \
              --task_model checkpoints/morgoth/VW_resampled_4.pth \
              --abs_pos_emb \
              --dataset VW \
              --refer_spike_result_dir ${result_dir}/pred_SPIKES_2pStep \
              --data_format ${data_format} \
              --sampling_rate ${sampling_rate} \
              --already_format_channel_order ${already_format_channel_order} \
              --already_average_montage ${already_average_montage} \
              --allow_missing_channels ${allow_missing_channels} \
              --max_length_hour ${max_length_hour} \
              --polarity ${polarity} \
              --eval_sub_dir ${dataset_dir} \
              --eval_results_dir ${result_dir}/VW_pred_2p \
              --prediction_slipping_step 2 \
              --smooth_result ${smooth_vw_result} \
              --need_vw_1s_result ${need_vw_1s_result} \
              --rewrite_results ${rewrite_results}

# option 2: run vw model and run spike model
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
              --predict \
              --model base_patch200_200 \
              --task_model checkpoints/morgoth/VW_resampled_4.pth \
              --abs_pos_emb \
              --dataset VW \
              --refer_spike_model checkpoints/morgoth/SPIKES.pth \
              --data_format ${data_format} \
              --sampling_rate ${sampling_rate} \
              --already_format_channel_order ${already_format_channel_order} \
              --already_average_montage ${already_average_montage} \
              --allow_missing_channels ${allow_missing_channels} \
              --max_length_hour ${max_length_hour} \
              --polarity ${polarity} \
              --eval_sub_dir ${dataset_dir} \
              --eval_results_dir ${result_dir}/VW_pred_8p \
              --prediction_slipping_step 8 \
              --smooth_result ${smooth_vw_result} \
              --need_vw_1s_result ${need_vw_1s_result} \
              --rewrite_results ${rewrite_results}

# 10. Spike localization --------------------------------------------------------
# with single_channel mode
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --abs_pos_emb \
            --task_model checkpoints/morgoth/SPIKE_localization_4.pth \
            --task_model_2 checkpoints/morgoth/SPIKE_1channel_1.pth \
            --model_selection_for_spike_localization single_channel \
            --refer_spike_result_dir ${result_dir}/pred_SPIKES_2pStep\
            --dataset SPIKE_localization \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --polarity ${polarity} \
            --max_length_hour ${max_length_hour} \
            --eval_sub_dir ${dataset_dir}  \
            --eval_results_dir ${result_dir}/pred_SPIKES_LOC_single_channel_mode_2pStep \
            --prediction_slipping_step 2 \
            --smooth_result ${smooth_spike_localization_result} \
            --rewrite_results ${rewrite_results} \
            --need_spike_localization_1s_result ${need_spike_localization_1s_result}

# with multi_channel mode
echo "$password"  | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --abs_pos_emb \
            --dataset SPIKE_localization \
            --task_model checkpoints/morgoth/SPIKE_LOCALIZATION.pth \
            --task_model_2 checkpoints/morgoth/SPIKE_1CHANNEL.pth  \
            --refer_spike_model checkpoints/morgoth/SPIKES.pth \
            --model_selection_for_spike_localization multi_channel \
            --refer_spike_result_dir ${result_dir}/pred_SPIKES_2pStep \
            --data_format ${data_format}\
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SPIKES_LOC_multi_channel_mode_2pStep \
            --prediction_slipping_step 2 \
            --smooth_result ${smooth_spike_localization_result} \
            --rewrite_results ${rewrite_results} \
            --need_spike_localization_1s_result ${need_spike_localization_1s_result}


# 11. Sleep 6 stage (5 stage + other)--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SLEEPPSG_6class.pth \
            --abs_pos_emb \
            --dataset SLEEPPSG_6class \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEPSTAGING6class_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


# 12. Arousal--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/SLEEPPSG_arousal.pth \
            --abs_pos_emb \
            --dataset SLEEP_AROUSAL \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEPAROUSAL_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}


# 13. BIRD--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/BIRD.pth \
            --abs_pos_emb \
            --dataset BIRD \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_BIRD_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}

# 14. BIPD--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/BIPD.pth \
            --abs_pos_emb \
            --dataset BIPD \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_BIPD_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}

# 15. Partial PD--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/morgoth/PD.pth \
            --abs_pos_emb \
            --dataset PD \
            --detection_location ${PD_detection_location} \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --max_length_hour ${max_length_hour} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_${PD_detection_location}_PD_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results ${rewrite_results}

######################################## IF USING CPU #######################################
# if you do not have gpu, to use cpu by using python command and adding "-device cpu" and "--distributed False"
######################################## IF USING CPU #######################################