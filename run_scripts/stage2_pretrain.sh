export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=5,6,7

python -m torch.distributed.run --master_port=29500 --nproc_per_node=3 ../train.py --cfg-path ../train_configs/minigpt4_stage2_finetune.yaml