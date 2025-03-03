import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import logging
from PIL import Image
import os
import torchvision

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
            
            # 修改客户端模型路径的处理方式
            target_id = self.target_client.client_id
            client_model_path = os.path.join('model_weights', f'client_{target_id}', f'epoch_{self.current_epoch}.pt')
            
            # 检查文件是否存在
            if not os.path.exists(client_model_path):
                available_files = []
                client_dir = os.path.join('model_weights', f'client_{target_id}')
                if os.path.exists(client_dir):
                    available_files = os.listdir(client_dir)
                
                error_msg = (f"找不到客户端模型文件: {client_model_path}\n"
                            f"客户端 {target_id} 可用的模型文件: {available_files}")
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 加载客户端模型状态
            self.logger.info(f"正在加载客户端模型: {client_model_path}")
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
        
        # 修改权重配置，增加颜色相关损失的权重
        weight_config = {
            'grad_weight': 1.0,
            'tv_weight': 0.005,  # 降低总变差约束
            'smooth_weight': 0.002,  # 降低平滑度约束
            'color_weight': 0.2,  # 增加颜色约束权重
            'diversity_weight': 0.15  # 新增颜色多样性约束
        }
        
        # 改进初始化策略
        if init_data is None:
            init_data = []
            # 使用多种初始化策略
            for _ in range(3):
                # 随机初始化
                rand_init = torch.rand(1, 3, 224, 224, device=self.device) * 2 - 1
                init_data.append(rand_init)
                # 高斯噪声初始化
                gauss_init = torch.randn(1, 3, 224, 224, device=self.device) * 0.1
                init_data.append(gauss_init)
                # 均匀灰度初始化
                gray_init = torch.ones(1, 3, 224, 224, device=self.device) * 0.5
                init_data.append(gray_init)
        
        for init in init_data:
            dummy_data = init.clone().requires_grad_(True)
            
            # 优化阶段设置
            stage_iters = [int(iterations * r) for r in [0.2, 0.3, 0.3, 0.2]]
            stage_lrs = [0.1, 0.05, 0.01, 0.001]
            
            for stage, (stage_iter, lr) in enumerate(zip(stage_iters, stage_lrs)):
                optimizer = torch.optim.RAdam(
                    [dummy_data],
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=1e-5  # 降低权重衰减
                )
                
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    total_steps=stage_iter,
                    pct_start=0.3,
                    cycle_momentum=False
                )
                
                self.adjust_weights(weight_config, stage, stage_iter)
                
                for i in range(stage_iter):
                    loss_dict = self.compute_loss(dummy_data, target_grads, stage, weight_config)
                    total_loss = sum(loss_dict.values())
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_([dummy_data], max_norm=0.5)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    with torch.no_grad():
                        # 改进的数据约束
                        dummy_data.data = self.apply_enhanced_constraints(dummy_data.data)
                        
                        # 周期性添加颜色扰动
                        if i % 50 == 0 and stage < 2:
                            color_noise = torch.randn(1, 3, 1, 1, device=self.device) * 0.05 * (1 - i/stage_iter)
                            color_noise = color_noise.expand(-1, -1, dummy_data.shape[2], dummy_data.shape[3])
                            dummy_data.data += color_noise
                    
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
    
    def compute_loss(self, dummy_data, target_grads, stage, weight_config):
        """改进的损失函数计算"""
        losses = {}
        
        # 1. 梯度损失
        output = self.global_model(dummy_data)
        loss = torch.nn.functional.cross_entropy(output, torch.zeros(1, device=self.device).long())
        current_grads = torch.autograd.grad(loss, self.global_model.parameters(), create_graph=True)
        
        grad_loss = torch.tensor(0.0, device=self.device)
        for g1, g2 in zip(current_grads, target_grads):
            grad_loss += torch.mean((g1 - g2) ** 2)
        losses['grad_loss'] = grad_loss * weight_config['grad_weight']
        
        if stage > 0:
            # 2. 总变差损失
            tv_loss = self.total_variation_loss(dummy_data)
            losses['tv_loss'] = tv_loss * weight_config['tv_weight']
            
            # 3. 平滑度损失
            smooth_loss = self.smoothness_loss(dummy_data)
            losses['smooth_loss'] = smooth_loss * weight_config['smooth_weight']
            
            # 4. 颜色多样性损失
            diversity_loss = self.color_diversity_loss(dummy_data)
            losses['diversity_loss'] = diversity_loss * weight_config['diversity_weight']
            
            # 5. 颜色一致性损失
            color_loss = self.color_consistency_loss(dummy_data)
            losses['color_loss'] = color_loss * weight_config['color_weight']
        
        return losses
    
    def adjust_weights(self, weight_config, stage, total_iters):
        """动态调整损失权重"""
        if stage == 0:
            weight_config['grad_weight'] = 1.0
            weight_config['tv_weight'] = 0.001
            weight_config['smooth_weight'] = 0.001
            weight_config['color_weight'] = 0.05
            weight_config['diversity_weight'] = 0.01
        elif stage == 1:
            weight_config['grad_weight'] = 0.8
            weight_config['tv_weight'] = 0.01
            weight_config['smooth_weight'] = 0.005
            weight_config['color_weight'] = 0.1
            weight_config['diversity_weight'] = 0.03
        else:
            weight_config['grad_weight'] = 0.6
            weight_config['tv_weight'] = 0.02
            weight_config['smooth_weight'] = 0.01
            weight_config['color_weight'] = 0.15
            weight_config['diversity_weight'] = 0.05
    
    def apply_enhanced_constraints(self, data):
        """增强的数据约束"""
        # 基本范围约束
        data = torch.clamp(data, -1, 1)
        
        # 保持每个通道的独立性
        for c in range(3):
            channel_data = data[:, c:c+1]
            # 对每个通道单独进行标准化
            mean = channel_data.mean()
            std = channel_data.std()
            if std > 0:
                channel_data = (channel_data - mean) / std
                # 重新缩放到合理范围
                channel_data = channel_data * 0.5 + 0.5
                data[:, c:c+1] = channel_data
        
        # 确保颜色差异
        mean_color = data.mean(dim=[2, 3], keepdim=True)
        data = data + (data - mean_color) * 0.1
        
        # 最终范围约束
        data = torch.clamp(data, -1, 1)
        
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
    
    def color_consistency_loss(self, x):
        """颜色一致性损失"""
        # 计算相邻像素的颜色差异
        diff_h = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        diff_v = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        
        # 计算颜色通道之间的相关性
        mean_channels = torch.mean(x, dim=[2, 3], keepdim=True)
        color_correlation = torch.mean((x - mean_channels) ** 2)
        
        return diff_h + diff_v + color_correlation
    
    def color_diversity_loss(self, x):
        """颜色多样性损失"""
        # 计算每个通道的统计特征
        mean_channels = x.mean(dim=[2, 3])
        std_channels = x.std(dim=[2, 3])
        
        # 鼓励不同通道间的差异
        channel_diff = torch.pdist(mean_channels)
        diversity_loss = -torch.mean(channel_diff)
        
        # 鼓励每个通道内的变化
        std_loss = -torch.mean(std_channels)
        
        return diversity_loss + std_loss
    
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