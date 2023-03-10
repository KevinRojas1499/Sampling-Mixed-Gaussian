CUDA_VISIBLE_DEVICES=1 python sampling.py --data_path datasets/random_sin_64.pt \
--sample_path samples/scheduled_ode_tfno_sin_64_samples.pt \
--checkpoint_path ckpts/tfno_sin_64_bs_1024_dummy.pt \
--model_type tfno \
--time_embed_type mlp \
--res_layer_type linear \
--mode eval \
--ode_sampling True