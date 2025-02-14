import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import logging
from PIL import Image
import os

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
        self.image_shape = sample_data[0].shape
        self.logger = logging.getLogger("ModelRecovery")
        
        # 添加图像预处理参数
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(self.device)

    def recover_data(self, iterations=3000):
        try:
            self.global_model.eval()
            
            # 加载目标客户端的已训练模型
            client_model_path = f'model_weights/client_{self.target_client.client_id}/epoch_{self.current_epoch}.pt'
            if not os.path.exists(client_model_path):
                raise FileNotFoundError(f"找不到客户端模型文件: {client_model_path}")
            
            # 加载客户端模型状态
            client_state = torch.load(client_model_path)
            if isinstance(client_state, dict) and 'model_state_dict' in client_state:
                client_weights = client_state['model_state_dict']
            else:
                client_weights = client_state
                
            # 计算权重差异
            target_grads = []
            for name, param in self.global_model.named_parameters():
                if name in client_weights:
                    grad = client_weights[name] - param.data
                    target_grads.append(grad.to(self.device))
            
            # 多尺度优化
            scales = [0.5, 1.0, 2.0]
            best_result = None
            best_mse = float('inf')
            
            for scale in scales:
                init_data = self.initialize_data(scale)
                current_result = self.optimize_single_scale(init_data, target_grads, iterations)
                
                if current_result['mse'] < best_mse:
                    best_mse = current_result['mse']
                    best_result = current_result
            
            return best_result['image'], best_result['mse'], best_result['psnr']
            
        except Exception as e:
            self.logger.error(f"Error in data recovery: {str(e)}")
            raise
    
    def initialize_data(self, scale=1.0):
        """智能初始化数据"""
        inits = [
            torch.randn(1, *self.image_shape, device=self.device) * 0.1,
            torch.zeros(1, *self.image_shape, device=self.device),
            torch.ones(1, *self.image_shape, device=self.device) * 0.5
        ]
        return [init * scale for init in inits]
    
    def optimize_single_scale(self, init_data, target_grads, iterations):
        """单尺度优化"""
        best_dummy = None
        best_mse = float('inf')
        
        for init in init_data:
            dummy_data = init.clone().requires_grad_(True)
            
            # 多阶段优化
            stage_iters = [int(iterations * r) for r in [0.3, 0.3, 0.4]]
            stage_lrs = [0.1, 0.01, 0.001]
            
            for stage, (stage_iter, lr) in enumerate(zip(stage_iters, stage_lrs)):
                optimizer = torch.optim.Adam([dummy_data], lr=lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=stage_iter//6, T_mult=2
                )
                
                for i in range(stage_iter):
                    loss_dict = self.compute_loss(dummy_data, target_grads, stage)
                    total_loss = sum(loss_dict.values())
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_([dummy_data], max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    with torch.no_grad():
                        dummy_data.data = self.apply_constraints(dummy_data.data)
                    
                    if i % 100 == 0:
                        self.log_progress(stage, i, loss_dict)
                
                current_mse = loss_dict['grad_loss'].item()
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_dummy = dummy_data.clone()
        
        psnr = self.calculate_psnr(best_mse)
        
        return {
            'image': self.tensor_to_image(best_dummy),
            'mse': best_mse,
            'psnr': psnr
        }
    
    def compute_loss(self, dummy_data, target_grads, stage):
        """计算损失函数"""
        output = self.global_model(dummy_data)
        loss = torch.nn.functional.cross_entropy(output, torch.zeros(1, device=self.device).long())
        current_grads = torch.autograd.grad(loss, self.global_model.parameters(), create_graph=True)
        
        grad_loss = torch.tensor(0.0, device=self.device)
        for g1, g2 in zip(current_grads, target_grads):
            grad_loss += torch.mean((g1 - g2) ** 2)
        
        losses = {'grad_loss': grad_loss}
        
        if stage > 0:
            losses['tv_loss'] = self.total_variation_loss(dummy_data) * 0.01
            losses['smooth_loss'] = self.smoothness_loss(dummy_data) * 0.005
        
        return losses
    
    def apply_constraints(self, data):
        """应用数据约束"""
        # 值范围约束
        data = torch.clamp(data, -1, 1)
        
        # 标准化
        data = (data - data.mean()) / (data.std() + 1e-7)
        data = data * self.std + self.mean
        
        return data

    def total_variation_loss(self, x):
        """计算总变差损失"""
        diff_h = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        diff_v = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return diff_h + diff_v
    
    def smoothness_loss(self, x):
        """计算平滑度损失"""
        diff_h = torch.mean((x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2)
        diff_v = torch.mean((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2)
        return diff_h + diff_v
    
    def calculate_psnr(self, mse):
        """计算PSNR"""
        if mse > 0:
            psnr = 10 * torch.log10(torch.tensor(1.0 / mse, device=self.device))
            return psnr.item()
        return float('inf')
    
    def log_progress(self, stage, iteration, loss_dict):
        """记录优化进度"""
        log_msg = f'Stage {stage}, Iteration {iteration}, '
        log_msg += ', '.join([f'{k}: {v.item():.6f}' for k, v in loss_dict.items()])
        self.logger.info(log_msg)
    
    @staticmethod
    def tensor_to_image(tensor):
        """转换张量为图像"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.squeeze(0)
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().detach()
        transform = transforms.ToPILImage()
        return transform(tensor) 