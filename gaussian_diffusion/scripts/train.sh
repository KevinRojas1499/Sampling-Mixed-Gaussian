CUDA_VISIBLE_DEVICES=2 python -u sampling.py --data_path datasets/random_sin_64_uniform.pt \
--save_path ckpts/fno_coarse_sin_score.pt \
--epochs 50000 --batch_size 1024 --model_type fno \
--time_embed mlp \
&> logs/fno_large_logs.txt &