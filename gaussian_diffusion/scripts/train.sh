CUDA_VISIBLE_DEVICES=0 python -u sampling.py --data_path datasets/random_sin_64.pt \
--save_path ckpts/tfno_sin_64_bs_1024_dummy.pt \
--epochs 50000 --batch_size 1024 \
--model_type tfno \
--lr 5e-4 \
&> logs/tfno_bs_1024_lr_5e4_layers_4_rank_1_cosine_scheduled.txt &