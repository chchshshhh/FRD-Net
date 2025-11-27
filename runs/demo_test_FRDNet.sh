base_dir="/home/csh/disk/AAAI2026/FRD-Net/weights"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test.py \
    --model FRDNet_Main\
    --pretrain_pth_path /home/csh/disk/AAAI2026/mit_b2.pth \
    --world_size 1 \
    --edge_mask 7 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/home/csh/disk/AAAI2026/FRD-Net/weights" \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
