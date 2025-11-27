base_dir="./train_output/FRDNet_Main"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./train_val.py \
    --model FRDNet_Main \
    --pretrain_pth_path /home/csh/disk/AAAI2026/mit_b2.pth \
    --world_size 1 \
    --batch_size 16 \
    --data_path /disk/csh/IMDLBenCo/balanced_dataset.json \
    --epochs 100 \
    --lr 5e-5 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path /disk/csh/IMDLBenCo/test_data/IDT-Casiav1-manip.json \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1 \
    --num_workers 8 \
#  2> ${base_dir}/error.log 1>${base_dir}/logs.log