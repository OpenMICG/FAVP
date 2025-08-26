export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python -m torch.distributed.run --master_port=29500 --nproc_per_node=6 ../train.py --cfg-path ../train_configs/stage2_finetune.yaml