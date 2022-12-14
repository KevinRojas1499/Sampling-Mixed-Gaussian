accelerate launch --config_file configs/accelerate_configs/test_config.yaml train.py  \
 --dataset_name="cifar10"  --output_dir="working/fno-small-ddpm-ema-cifar" \
   --train_batch_size=16  --num_epochs=100   --gradient_accumulation_steps=1   --learning_rate=1e-4 \
    --lr_warmup_steps=500   --mixed_precision=no --model_config_path configs/model_configs/simple_config.yaml
