#!/bin/bash

# Test using ConvNext backbone with SDv1-4 pretrained model
# python test_drct.py \
#     --config configs/drct_genimage_chameleon_geneval.yaml \
#     --model_name convnext_base_in22k \
#     --model_path /root/autodl-tmp/code/DRCT-ours/pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth \
#     --embedding_size 1024 \
#     --input_size 224 \
#     --batch_size 64 \
#     --device_id 2 \
#     --is_crop \
#     --result_dir ./results_convnext_sdv1 \

# Test using UnivFD (CLIP-ViT) backbone with SDv1-4 pretrained model
python test_drct.py \
    --config configs/drct_genimage_chameleon_geneval.yaml \
    --model_name clip-ViT-L-14 \
    --model_path /root/autodl-tmp/code/DRCT-ours/pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth \
    --embedding_size 1024 \
    --input_size 224 \
    --batch_size 64 \
    --device_id 2 \
    --is_crop \
    --result_dir ./results_univfd_sdv1 \

# Test using ConvNext backbone with SDv2 pretrained model (if available)
python test_drct.py \
    --config configs/drct_genimage_chameleon_geneval.yaml \
    --model_name convnext_base_in22k \
    --model_path /root/autodl-tmp/code/DRCT-ours/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth \
    --embedding_size 1024 \
    --input_size 224 \
    --batch_size 64 \
    --device_id 2 \
    --is_crop \
    --result_dir ./results_convnext_sdv2 \

# Test using UnivFD (CLIP-ViT) backbone with SDv2 pretrained model (if available)
python test_drct.py \
    --config configs/drct_genimage_chameleon_geneval.yaml \
    --model_name clip-ViT-L-14 \
    --model_path /root/autodl-tmp/code/DRCT-ours/pretrained/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth \
    --embedding_size 1024 \
    --input_size 224 \
    --batch_size 64 \
    --device_id 2 \
    --is_crop \
    --result_dir ./results_univfd_sdv2 \