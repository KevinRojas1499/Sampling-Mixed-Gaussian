accelerate launch --config_file configs/accelerate_configs/test_config.yaml train.py  \
 --dataset_name="tglcourse/lsun_church_train"  --output_dir="working/fno-ddpm-ema-lsun-church" \
   --train_batch_size=2  --num_epochs=100   --gradient_accumulation_steps=1   --learning_rate=1e-4 \
    --lr_warmup_steps=500   --mixed_precision=no --model_config_path configs/model_configs/lsun_256/fno.yaml
