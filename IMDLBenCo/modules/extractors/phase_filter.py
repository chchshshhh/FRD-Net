
import torch
from torch import nn



class phasefilter(nn.Module):
    def __init__(self):
        super(phasefilter,self).__init__()

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True)  # shape: (batch_size, 1, 256, 256)   将RGB通道转换为灰度
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img, dim=(-1, -2))  # 计算离散傅里叶变换，dim=(-1, -2) 指定在最后两个维度（即空间维度）上进行 2D 傅里叶变换
        # X = torch.fft.fftn(img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)  # 提取傅里叶变换结果的相位信息。相位谱 phase_spectrum 的形状与 X 相同
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)  # exp（j.phase）构造一个只包含相位、振幅为1的复数频谱
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x

    def forward(self, x):
        with torch.no_grad():
            return self.phase_without_amplitude(x).repeat(1, 3, 1, 1)
