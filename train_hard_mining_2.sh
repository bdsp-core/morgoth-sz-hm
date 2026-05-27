# train seizure hm

#EPOCH=5
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1234 finetune_classification.py \
#        --output_dir checkpoints/hm_IIIC13_2 \
#        --log_dir log/hm_IIIC13_2 \
#        --dataset IIIC \
#        --hardmining no \
#        --training_data_dir /data/IIIC/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.25 0.25 0.25 0.25 0.25 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 1 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
#
##       --weight_decay 0.05 \
##       --hardmining yes \
##       --hardmining_data_dir /data/seizure_hm/round1/train \
#
#
#
EPOCH=20
echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1234 finetune_classification.py \
        --output_dir checkpoints/IIIC_chewing3 \
        --log_dir log/IIIC_chewing3 \
        --dataset IIIC_chewing \
        --hardmining no \
        --training_data_dir /data/seizure_hm/chewing/processed_10second \
        --model base_patch200_200 \
        --epochs ${EPOCH} \
        --finetune pretrained_model/base6.pth \
        --focalloss \
        --focal_gamma 2 \
        --focal_alpha "0.2 0.2 0.2 0.2 0.2 0.2 0.4" \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 1 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 0

# test seizure hm on test set
#
#new_model_round='hm_IIIC11'
#echo "exxact@1" | sudo -S mkdir /data/seizure_hm/test_set/pred/pred_${new_model_round}/
#for i in {0..19}; do
#    echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#                --abs_pos_emb \
#                --eval \
#                --model base_patch200_200 \
#                --task_model checkpoints/${new_model_round}/checkpoint-${i}.pth \
#                --dataset IIIC \
#                --nb_classes 6 \
#                --test_data_format mat \
#                --eval_sub_dir /data/seizure_hm/test_set/sparcnet_test_mat
#   echo "exxact@1" | sudo mv /data/seizure_hm/test_set/sparcnet_test_mat/pred.csv /data/seizure_hm/test_set/pred/pred_${new_model_round}/pred_discrete_checkpoint${i}.csv
#done
#
#
#model_rounds=("hm_IIIC11" "hm_IIIC10" "hm_IIIC1" "hm_IIIC1_1" "hm_IIIC2" "hm_IIIC3" "hm_IIIC4" "hm_IIIC5" "hm_IIIC6" "hm_IIIC7" "hm_IIIC8_1" "hm_IIIC9")
#for model_round in "${model_rounds[@]}"; do
#  for j in {0..19}; do
#        echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#                    --abs_pos_emb \
#                    --model base_patch200_200 \
#                    --predict \
#                    --task_model checkpoints/${model_round}/checkpoint-${j}.pth \
#                    --dataset IIIC \
#                    --data_format mat \
#                    --sampling_rate 0 \
#                    --already_format_channel_order no \
#                    --already_average_montage no \
#                    --allow_missing_channels no \
#                    --max_length_hour no \
#                    --eval_sub_dir /data/seizure_hm/test_set/possible_szfree_mat \
#                    --eval_list_file /data/seizure_hm/test_set/continuous_szfree_list_20250929.csv \
#                    --eval_list_column file_name \
#                    --eval_results_dir /data/seizure_hm/test_set/pred/pred_${model_round}/pred_continuousnonsz_checkpoint${j} \
#                    --prediction_slipping_step_second 1 \
#                    --rewrite_results no
#  done
#done



# EEG level

#continuous_dirs=(
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC1"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC1_1"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC2"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC3"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC4"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC5"
#                 "/data/seizure_hm/test_set/pred/pred_hm_IIIC6"
#                "/data/seizure_hm/test_set/pred/pred_hm_IIIC7"
#                "/data/seizure_hm/test_set/pred/pred_hm_IIIC8"
#                "/data/seizure_hm/test_set/pred/pred_hm_IIIC9"
#                "/data/seizure_hm/test_set/pred/pred_hm_IIIC10"
#                "/data/seizure_hm/test_set/pred/pred_hm_IIIC11"
#                 )

#inx_array=({0..19})
#
#for continuous_dir in "${continuous_dirs[@]}"; do
#    for inx in "${inx_array[@]}"; do
#        echo "exxact@1" | sudo -S $(which python) EEG_level_head.py \
#            --mode predict \
#            --dataset SEIZURE \
#            --task_model ../EEGfounder/checkpoints/SEIZURE_EEGlevel.pth \
#            --test_csv_dir ${continuous_dir}/pred_continuousnonsz_checkpoint${inx} \
#            --result_dir ${continuous_dir}/pred_EEGlevel_checkpoint${inx}
#    done
#done

#echo "exxact@1" | sudo -S $(which python) EEG_level_head.py \
#            --mode predict \
#            --dataset SEIZURE \
#            --task_model ../EEGfounder/checkpoints/SEIZURE_EEGlevel.pth \
#            --test_csv_dir /data/seizure_hm/test_set/pred/pred_hm_post/pred_continuousnonsz \
#            --result_dir /data/seizure_hm/test_set/pred/pred_hm_post/pred_EEGlevel
#
#
#echo "exxact@1" | sudo -S $(which python) EEG_level_head.py \
#            --mode predict \
#            --dataset SEIZURE \
#            --task_model ../EEGfounder/checkpoints/SEIZURE_EEGlevel.pth \
#            --test_csv_dir /data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_continuousnonsz \
#            --result_dir /data/seizure_hm/test_set/pred/pred_hm_sparcnet/pred_EEGlevel



# run on all edf raw data

#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC12_1/checkpoint-11.pth \
#            --dataset IIIC \
#            --data_format edf \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#             --max_length_hour no \
#            --eval_sub_dir "/data/seizure_hm/bch_trail/edf" \
#            --eval_results_dir /data/seizure_hm/bch_trail/pred_hm12_1\
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no


# run on all 17k edf raw data on the new machine --nproc_per_node=4

#OMP_NUM_THREADS=1 python -m torch.distributed.run \
#  --nnodes=1 --nproc_per_node=3 --master_port=12346 \
#  finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC6/checkpoint-11.pth \
#            --dataset IIIC \
#            --data_format edf \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#             --max_length_hour no \
#            --eval_sub_dir '/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/test_set/bids/S0001' \
#            --eval_results_dir '/data/representative/S0001_pred_1s_sz'\
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no



#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=57 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC12_1/checkpoint-11.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#             --max_length_hour no \
#            --eval_sub_dir /data/seizure_hm/seizures_10min_17k \
#            --eval_results_dir /data/seizure_hm/seizures_10min_17k_12results\
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no
#
#
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=57 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC11/checkpoint-19.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir /data/seizure_hm/seizures_10min_17k \
#            --eval_results_dir /data/seizure_hm/seizures_10min_17k_11results\
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no

## test seizure hm chewing

#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC13_2/checkpoint-1.pth \
#            --refer_hm_model checkpoints/IIIC_chewing2/checkpoint-9.pth \
#            --dataset IIIC_hm \
#            --data_format edf \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --eval_sub_dir /data/seizure_hm/edf/ \
#            --eval_results_dir /data/seizure_hm/hm_IIIC13_3_fs2_results/\
#            --prediction_slipping_step_second 1 \
#            --rewrite_results no


#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1234 finetune_classification.py \
#            --abs_pos_emb \
#            --model base_patch200_200 \
#            --predict \
#            --task_model checkpoints/hm_IIIC13_1/checkpoint-9.pth \
#            --refer_hm_model checkpoints/IIIC_chewing1/checkpoint-9.pth \
#            --dataset IIIC_hm \
#            --data_format edf \
#            --sampling_rate 0 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#             --max_length_hour no \
#            --eval_sub_dir /data/seizure_hm/chewing/test_chewing_edf \
#            --eval_results_dir /data/seizure_hm/chewing/test_chewing_edf_results \
#            --prediction_slipping_step_second 1 \
#            --rewrite_results yes





#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model base_patch200_200 \
#            --task_model checkpoints/IIIC_chewing3/checkpoint-9.pth \
#            --dataset IIIC_chewing \
#            --nb_classes 7 \
#            --test_data_format mat \
#            --eval_sub_dir /data/seizure_hm/test_set/sparcnet_test_mat


#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model base_patch200_200 \
#            --task_model checkpoints/hm_IIIC12_1/checkpoint-11.pth \
#            --dataset IIIC \
#            --nb_classes 6 \
#            --test_data_format mat \
#            --eval_sub_dir /data/seizure_hm/test_set/sparcnet_test_mat


