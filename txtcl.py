# import json
#
# # 定义txt文件的路径
# txt_file_path = "/disk/csh/MMFusion-IML-main/data/CAT-Net_splits/val/bcm_COCO.txt"
# # 定义输出json文件的路径
# json_file_path = "/disk/csh/IMDLBenCo/val_data/bcm_COCO.json"
#
# # 用于存储转换后的数据
# data = []
#
# # 读取txt文件并处理
# with open(txt_file_path, 'r') as file:
#     for line in file:
#         # 拆分每一行内容，假设以空格分隔
#         parts = line.strip().split()
#         if len(parts) == 3:
#             # image_path = parts[0][1:]
#             # mask_path = parts[1][1:]
#             image_path = parts[0][1:]
#             mask_path = parts[1][1:]
#             label = parts[2]
#
#             # 将图像路径和mask路径存储到列表中
#             data.append([image_path, mask_path])
#
# # 将数据转换为JSON格式并写入文件
# with open(json_file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)
#
# print(f"已成功将txt文件转换为JSON格式，并保存到 {json_file_path}")
#
# # import json
# #
# # # 定义txt文件的路径
# # txt_file_path = "/disk/csh/segdataset/casia/tp_list.txt"
# # # 定义输出json文件的路径
# # json_file_path = "/disk/csh/IMDLBenCo/train_data/CASIAv2_tp.json"
# #
# # # 用于存储转换后的数据
# # data = []
# #
# # # 读取txt文件并处理
# # with open(txt_file_path, 'r') as file:
# #     for line in file:
# #         # 拆分每一行内容，假设以空格分隔
# #         parts = '/casia/Tp/'+line.strip()
# #         mask='/casia/Gt/'+line.strip()[:-4]+'_gt.png'
# #         # if len(parts) == 3:
# #         #     image_path = parts[0][1:]
# #         #     mask_path = parts[1][1:]
# #         #     label = parts[2]
# #
# #             # 将图像路径和mask路径存储到列表中
# #         data.append([parts, mask])
# #
# # # 将数据转换为JSON格式并写入文件
# # with open(json_file_path, 'w') as json_file:
# #     json.dump(data, json_file, indent=4)
# #
# # print(f"已成功将txt文件转换为JSON格式，并保存到 {json_file_path}")
#
# import os
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
#
#
# def process_image(filename, directory, output_directory, target_size):
#     try:
#         with Image.open(os.path.join(directory, filename)) as img:
#             width, height = img.size
#             print(f'Processing Image: {filename} | Resolution: {width}x{height}')
#
#             # 确定长边为1024的缩放比例
#             if max(width, height) > target_size:
#                 if width > height:
#                     new_width = target_size
#                     new_height = int((target_size / width) * height)
#                 else:
#                     new_height = target_size
#                     new_width = int((target_size / height) * width)
#
#                 # 调整图片尺寸
#                 img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
#
#                 # 保存图片到指定文件夹
#                 output_path = os.path.join(output_directory, filename)
#                 img_resized.save(output_path)
#                 print(f'Resized and saved {filename} to {output_directory} with resolution {new_width}x{new_height}')
#             else:
#                 # 如果图片不需要调整，直接复制到目标文件夹
#                 img.save(os.path.join(output_directory, filename))
#                 print(f'Image {filename} already meets the target size and was saved without resizing.')
#             return 1  # 返回处理成功的计数
#     except Exception as e:
#         print(f"Cannot process {filename}: {e}")
#         return 0  # 返回处理失败的计数
#
#
# def get_image_resolutions_and_resize(directory='.', output_directory='resized_images', target_size=1024):
#     # 创建输出文件夹，如果不存在则创建
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     # 获取所有图片文件
#     image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))]
#
#     # 使用线程池处理图片
#     total_processed = 0
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_image, filename, directory, output_directory, target_size) for filename in
#                    image_files]
#
#         # 等待所有线程完成并累加处理的数量
#         for future in futures:
#             total_processed += future.result()
#
#     # 输出总图片数量
#     print(f"\nTotal number of images processed: {total_processed}")
#
#
# # 执行函数
# get_image_resolutions_and_resize(
#     directory="/disk/csh/segdataset/cat_net/compRAISE",
#     output_directory="/disk/csh/segdataset/cat_net/compRAISE1024"
# )

import torch
import torch.nn as nn

# # 生成一个大小为 16x16 的随机张量
# input_tensor = torch.randn(5, 5, 256, 256)
# print("Input Tensor:\n", input_tensor)
#
# # 定义 2x2 的平均池化层
# avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
# avg_pooled_output = avg_pool(input_tensor)
#
#
# # 定义自适应平均池化层，输出固定为 8*8
# adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(128, 128))
# adaptive_avg_pooled_output = adaptive_avg_pool(input_tensor)
#
# print(adaptive_avg_pooled_output==adaptive_avg_pooled_output)
# import torch
#
# # 加载权重文件
# weights_path = "/disk/csh/IMDLBenCo2/train_output/fecflow_edge_normdiffblaux/checkpoint-0.pth"
# state_dict = torch.load(weights_path)
#
# # 查看权重的键
# print(state_dict.keys())
from IMDLBenCo.model_zoo.mvss_net.mvssnet import MVSSNet
# from IMDLBenCo.model_zoo.cat_net.cat_net import Cat_Net
from IMDLBenCo.model_zoo.trufor.trufor import Trufor
from IMDLBenCo.model_zoo.mesorch.mesorch import Mesorch
from IMDLBenCo.modules.backbones.sparsevit.sparsevitmul import SparseViT_Mul
import torch
from IMDLBenCo.modules.backbones.segformer_fecflow_edge_norm_diffattbl import segformer_fecflow_edgendiffbl
from IMDLBenCo.modules.backbones.segformer_fecflow_edge_norm_hu11_3 import segformer_fecflow_edgenhu113
from IMDLBenCo.modules.backbones.segformer_edge import segformer_edge
# model = SparseViT_Mul('/home/csh/disk/objectformer/ckpt/uniformer_base_ls_in1k.pth').cuda()
# model = Mesorch().cuda()
# model=Trufor(np_pretrain_weights="/home/csh/disk/MMFusion-IML-main/pretrained/np++.pth",mit_b2_pretrain_weights="/disk/csh/objectformer/ckpt/mit_b2.pth").cuda()
model = segformer_fecflow_edgenhu113('/disk/csh/objectformer/ckpt/mit_b2.pth').cuda()
# model = segformer_edge('/disk/csh/objectfo65rmer/ckpt/mit_b2.pth')
# model=MVSSNet(if_label=False).cuda()
# model=Cat_Net()
# checkpoint = torch.load("/disk/csh/IMDLBenCo2/backbonesegformer_edge/checkpoint-0.pth")
# checkpoint = torch.load("/disk/csh/IMDLBenCo2/train_output/fecflow_edge_normdiffbl/checkpoint-0.pth")
# checkpoint = torch.load("/home/csh/disk/IMDLBenCo/output_dir_segformer_casia/checkpoint-150.pth")
#
#
# model_state_dict = checkpoint['model']
# # 统计模型参数数量
# total_params = sum(p.numel() for p in model_state_dict.values())
# print(f"Total parameters: {total_params}")


# print(checkpoint['args'])

# import torch
# from thop import profile
# from thop import clever_format
#
#
#
# # 创建两个随机输入张量
input1 = torch.randn(1, 3, 512, 512).cuda()
input2 = torch.randn(1, 1, 512, 512).cuda()
#
# # 使用thop分析模型的运算量和参数量
# MACs, params = profile(model, inputs=(input1, input2,input2))
#
# # 将结果转换为更易于阅读的格式
# MACs, params = clever_format([MACs, params], '%.3f')
#
# print(f"运算量：{MACs}, 参数量：{params}")
#
#
# from thop import profile
# import torch.nn as nn
#
# def relu_hook(module, input, output):
#     # ReLU 的 FLOPs 和输入数量是一样的，每个元素判断一次
#     module.total_ops = torch.zeros(1)
#     module.total_ops += torch.prod(torch.tensor(input[0].shape))
#
# # 注册 hook 到 ReLU
# for m in model.modules():
#     if isinstance(m, nn.ReLU):
#         m.register_forward_hook(relu_hook)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# Warm-up
for _ in range(30):
    with torch.no_grad():
        _ = model(input1, input2,input2)

# Start timing
starter.record()

with torch.no_grad():
    output = model(input1, input2,input2)

ender.record()
torch.cuda.synchronize()  # 等待所有 CUDA 流完成
elapsed_time_ms = starter.elapsed_time(ender)  # 单位是毫秒

print(f"Inference time: {elapsed_time_ms:.3f} ms")