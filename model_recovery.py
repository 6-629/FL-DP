import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import logging
from PIL import Image

class ModelRecovery:
    def __init__(self, global_model, target_client, device=None):
        """
        初始化模型恢复器
        :param global_model: 服务器端的全局模型
        :param target_client: 目标客户端
        :param device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = global_model.to(self.device)
        self.target_client = target_client
        self.logger = logging.getLogger("ModelRecovery")

    def recover_data(self, iterations=1000, learning_rate=0.1):
        """
        使用梯度恢复方法恢复训练数据
        :param iterations: 恢复迭代次数
        :param learning_rate: 学习率
        :return: 恢复的数据、MSE和PSNR
        """
        try:
            # 获取目标客户端的模型更新
            target_update = self.target_client.local_train(self.global_model)
            
            # 初始化随机数据
            dummy_data = torch.randn(1, 3, 32, 32).to(self.device).requires_grad_(True)
            dummy_label = torch.zeros(1, dtype=torch.long).to(self.device)
            
            # 优化器
            optimizer = optim.Adam([dummy_data], lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            best_mse = float('inf')
            best_data = None
            
            # 迭代恢复
            for i in range(iterations):
                optimizer.zero_grad()
                
                # 计算当前数据的梯度
                self.global_model.zero_grad()
                output = self.global_model(dummy_data)
                loss = criterion(output, dummy_label)
                loss.backward()
                
                # 获取当前梯度
                current_gradients = {}
                for name, param in self.global_model.named_parameters():
                    if param.grad is not None:
                        current_gradients[name] = param.grad.clone()
                
                # 计算与目标更新的MSE
                mse = 0
                for name in target_update:
                    if name in current_gradients:
                        mse += torch.mean((current_gradients[name] - target_update[name]) ** 2)
                
                # 更新虚拟数据
                mse.backward()
                optimizer.step()
                
                # 记录最佳结果
                if mse.item() < best_mse:
                    best_mse = mse.item()
                    best_data = dummy_data.clone().detach()
                
                if i % 100 == 0:
                    self.logger.info(f'Iteration {i}, MSE: {mse.item():.6f}')
            
            # 计算PSNR
            mse_final = best_mse
            psnr = 10 * np.log10(1 / mse_final) if mse_final > 0 else 100
            
            # 将恢复的数据转换为图像格式
            recovered_data = self.tensor_to_image(best_data)
            
            return recovered_data, mse_final, psnr
            
        except Exception as e:
            self.logger.error(f"Error in data recovery: {str(e)}")
            raise

    @staticmethod
    def tensor_to_image(tensor):
        """
        将张量转换为PIL图像
        """
        # 反归一化
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        tensor = tensor * std + mean
        
        # 转换为PIL图像
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor * 255, 0, 255)
        tensor = tensor.permute(1, 2, 0).numpy().astype('uint8')
        image = Image.fromarray(tensor)
        
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