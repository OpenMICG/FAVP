export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=7

python -m torch.distributed.run --master_port=29900 --nproc_per_node=1 ../train.py --cfg-path ../train_configs/slake.yaml