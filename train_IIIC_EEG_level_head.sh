
####### make event level resulst ################


#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/MoE/events_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/MoE_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no

#OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run \
#  --nnodes=1 \
#  --nproc_per_node=2 \
#  --master_port=12345 \
#  finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/GLAD/data/MAT" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/GLAD_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no

#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/IIIC/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/IIIC_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/SEIZURE/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/SEIZURE_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/SEIZURE_BCH/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/SEIZURE_BCH_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/LPD/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/LPD_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/GPD/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/GPD_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/LRDA/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/LRDA_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#
#sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/sz_hm13_1.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order yes \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir "/data/GRDA/segments_raw" \
#            --eval_results_dir "/data/IIIC_EEG_level/event_res/GRDA_segments_raw" \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#
#



#num_epochs=10
#IIIC_datasets=("GRDA" "GPD" "LPD") # "SEIZURE" "LRDA" "GRDA"
#for IIIC_dataset in "${IIIC_datasets[@]}"; do
#  sudo $(which python) EEG_level_head.py \
#          --mode train \
#          --dataset ${IIIC_dataset} \
#          --train_csv_dirs "/data/IIIC_EEG_level/event_res/SEIZURE_segments_raw /data/IIIC_EEG_level/event_res/SEIZURE_BCH_segments_raw /data/IIIC_EEG_level/event_res/LPD_segments_raw /data/IIIC_EEG_level/event_res/GPD_segments_raw /data/IIIC_EEG_level/event_res/LRDA_segments_raw /data/IIIC_EEG_level/event_res/GRDA_segments_raw /data/IIIC_EEG_level/event_res/representative_res /data/IIIC_EEG_level/event_res/IIIC_segments_raw /data/IIIC_EEG_level/event_res/MoE_segments_raw /data/IIIC_EEG_level/event_res/GLAD_segments_raw" \
#          --file_list_path /data/IIIC_EEG_level/event_res/training_list.csv \
#          --id_distinguished \
#          --output_dir checkpoints/EEG_level_${IIIC_dataset}  \
#          --num_epochs ${num_epochs} \
#          --pe_max_length 15000 \
#          --focal_alpha "0.25" \
#          --lr 5e-4 \
#          --save_freq 1 \
#          --resume_training \
#          --batch_size 100
#done
#
##  --add_data_transformation\



IIIC_datasets=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA") #

dataset_dirs=(
"/data/IIIC_EEG_level/event_res/SEIZURE_segments_raw"
"/data/IIIC_EEG_level/event_res/SEIZURE_BCH_segments_raw"
"/data/IIIC_EEG_level/event_res/LPD_segments_raw"
"/data/IIIC_EEG_level/event_res/GPD_segments_raw"
"/data/IIIC_EEG_level/event_res/LRDA_segments_raw"
"/data/IIIC_EEG_level/event_res/GRDA_segments_raw"
"/data/IIIC_EEG_level/event_res/representative_res"
"/data/IIIC_EEG_level/event_res/IIIC_segments_raw"
"/data/IIIC_EEG_level/event_res/MoE_segments_raw"
"/data/IIIC_EEG_level/event_res/GLAD_segments_raw"
)


#for IIIC_dataset in "${IIIC_datasets[@]}"; do
#    for dataset_dir in "${dataset_dirs[@]}"; do
#        sudo $(which python) EEG_level_head.py \
#            --mode predict \
#            --dataset ${IIIC_dataset} \
#            --task_model checkpoints/morgoth/${IIIC_dataset}_EEGlevel_2.pth \
#            --test_csv_dir ${dataset_dir} \
#            --result_dir ${dataset_dir}_EEGlevel
#
#    done
#done


for IIIC_dataset in "${IIIC_datasets[@]}"; do
    for dataset_dir in "${dataset_dirs[@]}"; do
        sudo $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${IIIC_dataset} \
            --task_model /home/exx/Documents/EEG_report/Morgoth2/checkpoints/EEG_level_${IIIC_dataset}_3/checkpoint-best.pth \
            --test_csv_dir ${dataset_dir} \
            --result_dir ${dataset_dir}_EEGlevel

    done
done

