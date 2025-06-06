export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=5,6,7
python -m torch.distributed.run --master_port=29600 --nproc_per_node=3 ../train.py --cfg-path ../train_configs/stage1_pretrain.yaml