CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone --nproc_per_node=6 \
  main_finetune.py \
  --batch_size 30 \
  --blr 3e-5 \
  --warmup_epochs 2 \
  --epochs 20 \
  --model metaspikformer_8_512 \
  --data_path ../../py_project/ViT/dataset/imagenet \
  --output_dir outputs/55M_T4 \
  --log_dir outputs/55M_T4 \
  --model_mode ms \
  --finetune ./55M_kd.pth \
  --time_steps 4 \
  --kd \
  --teacher_model caformer_b36_in21ft1k \
  --distillation_type hard