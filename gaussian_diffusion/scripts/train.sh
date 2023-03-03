CUDA_VISIBLE_DEVICES=3 python -u sampling.py --data_path datasets/random_sin_64_uniform.pt \
--save_path ckpts/simple_coarse_sin_score.pt \
--epochs 50000 --batch_size 1024 \
--model_type simple \
--time_embed_type mlp \
&> logs/simple_logs.txt &