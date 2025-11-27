import torch
import sys
sys.path.append(".")
from IMDLBenCo.datasets import ManiDataset,JsonDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.model_zoo import cat_net
# from IMDLBench.model_zoo.cat_net.cat_net_post_function import cat_net_post_func
from IMDLBenCo.modules.backbones.segformer_fecflow_edge_norm_hu11_3 import segformer_fecflow_edgenhu113
import torch
from IMDLBenCo.evaluation import grad_camera_visualize
    



if __name__ == '__main__':
    model = segformer_fecflow_edgenhu113('/disk/csh/objectformer/ckpt/mit_b2.pth') # TODO 这里加载模型
    ckpt = '/home/csh/disk/IMDLBenCo4/train_output/fecflow_edge_normhu113/bb/checkpoint-70.pth' # TODO 这里填已经训练好的模型
    ckpt = torch.load(ckpt, map_location='cuda')
    model.load_state_dict(ckpt['model'])
    model.cuda()

    # dataset = ManiDataset(path='/mnt/data0/public_datasets/IML/CASIA1.0',
    #             is_padding=False,
    #             is_resizing=True,
    #             output_size=(512, 512),
    #             common_transforms=get_albu_transforms('test'),
    #             edge_width=7)
    dataset= JsonDataset(
        "/disk/csh/IMDLBenCo/test_data/IDT-Casiav1-manip.json",
        is_padding=False,
        is_resizing=True,
        output_size=(512, 512),
        common_transforms=get_albu_transforms('test'),
        edge_width=7
        # post_funcs=post_function
    )
                #post_funcs=cat_net_post_func -> this argument is only for Cat-Net
    
    # target_layers = [model.deconv_model[-1]]
    target_layers = [model.fecblock4[0].token_mixer.proj]
    grad_camera_visualize(model=model,
                          image=dataset,
                          target_layers=target_layers, # TODO 这里放你的模型结构中最后一个计算单元，用list装起来 
                          output_path='/home/csh/disk/IMDLBenCo4/train_output/fecflow_edge_normhu113/bb/casia/bbb') # TODO 这里放图片输出的文件夹地址
