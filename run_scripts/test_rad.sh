export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.run --nproc_per_node=2 ../test.py --cfg-path ../train_configs/test_rad.yaml