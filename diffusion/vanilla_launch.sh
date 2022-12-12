accelerate launch --config_file configs/accelerate_configs/default_config.yaml train.py  \
 --dataset_name="huggan/flowers-102-categories"  --output_dir="working/unet-ddpm-ema-flowers-64" \
   --train_batch_size=8  --num_epochs=100   --gradient_accumulation_steps=1   --learning_rate=1e-4 \
    --lr_warmup_steps=500   --mixed_precision=no --model_config_path configs/model_configs/original_config.yaml
