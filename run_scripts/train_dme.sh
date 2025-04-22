export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6,7

python -m torch.distributed.run --master_port=29900 --nproc_per_node=2 ../train.py --cfg-path ../train_configs/minigpt4_dme.yaml
