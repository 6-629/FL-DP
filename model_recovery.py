import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import logging
from PIL import Image

class ModelRecovery:
    def __init__(self, global_model, target_client, current_epoch):
        """
        初始化模型恢复器
        :param global_model: 服务器端的全局模型
        :param target_client: 目标客户端
        :param current_epoch: 当前轮次
        """
        self.global_model = global_model
        self.target_client = target_client
        self.current_epoch = current_epoch
        self.device = next(global_model.parameters()).device
        
        # 获取目标数据集的图像尺寸
        sample_data = next(iter(self.target_client.train_loader))[0]
        self.image_shape = sample_data[0].shape  # (C, H, W)
        self.logger = logging.getLogger("ModelRecovery")

    def recover_data(self, iterations=1000):
        try:
            self.global_model.eval()
            
            # 获取目标梯度
            target_grads = self.target_client.local_train(self.global_model, self.current_epoch)
            
            if not isinstance(target_grads, list):
                raise TypeError(f"Expected target_grads to be a list of tensors, got {type(target_grads)}")
            
            # 确保梯度在正确的设备上
            target_grads = [torch.tensor(float(g), device=self.device) if isinstance(g, str) 
                           else g.to(self.device) for g in target_grads]
            
            # 创建与原始图像相同尺寸的虚拟数据
            dummy_data = torch.randn(1, *self.image_shape, requires_grad=True, device=self.device)
            dummy_label = torch.tensor([0], device=self.device)
            
            optimizer = torch.optim.Adam([dummy_data], lr=0.1)
            criterion = torch.nn.CrossEntropyLoss()
            
            for i in range(iterations):
                optimizer.zero_grad()
                
                # 前向传播
                output = self.global_model(dummy_data)
                loss = criterion(output, dummy_label)
                
                # 计算当前梯度
                current_grads = torch.autograd.grad(loss, self.global_model.parameters(), 
                                                  create_graph=True)
                
                # 计算梯度差异
                mse = torch.tensor(0.0, device=self.device)
                for g1, g2 in zip(current_grads, target_grads):
                    if g1.dtype != g2.dtype:
                        g2 = g2.to(dtype=g1.dtype)
                    mse += torch.mean((g1 - g2) ** 2)
                
                # 打印调试信息
                if i % 100 == 0:
                    self.logger.info(f'Iteration {i}, MSE: {mse.item()}')
                
                mse.backward()
                optimizer.step()
                
                # 确保像素值在合理范围内
                with torch.no_grad():
                    dummy_data.data = torch.clamp(dummy_data.data, -1, 1)
            
            # 计算最终的PSNR
            with torch.no_grad():
                mse_final = mse.item()
                if mse_final > 0:
                    psnr = 10 * torch.log10(torch.tensor(1.0 / mse_final, device=self.device))
                    psnr = psnr.item()
                else:
                    psnr = float('inf')
            
            # 将张量转换为图像，保持原始分辨率
            recovered_image = self.tensor_to_image(dummy_data)
            
            return recovered_image, mse_final, psnr
            
        except Exception as e:
            self.logger.error(f"Error in data recovery: {str(e)}")
            raise

    @staticmethod
    def tensor_to_image(tensor):
        """将张量转换为PIL图像，保持原始分辨率"""
        # 确保输入是4D张量 (B, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            
        # 移除批次维度
        tensor = tensor.squeeze(0)
        
        # 将值范围从[-1, 1]转换到[0, 1]
        tensor = tensor * 0.5 + 0.5
        
        # 确保值在[0, 1]范围内
        tensor = tensor.clamp(0, 1)
        
        # 转换为CPU并分离计算图
        tensor = tensor.cpu().detach()
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        return image

    @staticmethod
    def calculate_psnr(original, recovered):
        """
        计算PSNR
        """
        mse = np.mean((original - recovered) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr 