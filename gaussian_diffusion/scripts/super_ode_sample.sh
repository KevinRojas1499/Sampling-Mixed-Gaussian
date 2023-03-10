CUDA_VISIBLE_DEVICES=1 python sampling.py --data_path datasets/random_sin_128.pt \
--sample_path samples/tfno_ode_sin_super_128_samples.pt \
--checkpoint_path ckpts/tfno_sin_64_bs_1024.pt \
--model_type tfno \
--time_embed_type mlp \
--res_layer_type linear \
--mode eval \
--ode_sampling True