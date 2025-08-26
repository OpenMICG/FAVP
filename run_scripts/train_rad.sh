export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 ../train.py --cfg-path ../train_configs/train_rad.yaml
