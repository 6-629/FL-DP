import torch
import copy
from tqdm import tqdm
import logging
import numpy as np
import os
import torch.nn as nn


class Client(object):
    def __init__(self, conf, model, train_dataset, id=-1):
        """
        初始化联邦学习客户端
        """
        self.conf = conf
        self.client_id = id

        # 首先设置日志记录器
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Client_{id}")
        self.logger.info(f"初始化客户端 {id}")

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")

        # 初始化本地模型
        self.local_model = copy.deepcopy(model)
        self.local_model = self.local_model.to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf.get('momentum', 0.9),
            weight_decay=self.conf.get('weight_decay', 0.0005),
            nesterov=True  # 使用Nesterov动量
        )

        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.conf.get('scheduler_step', 5),
            gamma=self.conf.get('scheduler_gamma', 0.1)
        )

        # 设置训练数据
        self.setup_training_data(train_dataset, id)

        # 添加权重保存路径
        self.weights_dir = os.path.join('model_weights', f'client_{id}')
        os.makedirs(self.weights_dir, exist_ok=True)

        # 添加温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def setup_training_data(self, train_dataset, id):
        """设置训练数据"""
        try:
            # 计算数据分片
            all_range = list(range(len(train_dataset)))
            data_len = int(len(train_dataset) / self.conf['no_models'])
            
            # 确保每个客户端至少有一个样本
            data_len = max(1, data_len)
            
            # 计算当前客户端的数据范围
            start_idx = id * data_len
            end_idx = min(start_idx + data_len, len(train_dataset))
            
            # 确保索引有效
            if start_idx >= len(train_dataset):
                raise ValueError(f"客户端 {id} 的起始索引 {start_idx} 超出数据集大小 {len(train_dataset)}")
            
            train_indices = all_range[start_idx:end_idx]
            
            if not train_indices:
                raise ValueError(f"客户端 {id} 没有分配到任何训练数据")
            
            # 创建数据加载器
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.conf["batch_size"],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                num_workers=self.conf.get("num_workers", 0),
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            self.logger.info(
                f"客户端 {id} 数据设置完成: "
                f"样本数量={len(train_indices)}, "
                f"批次大小={self.conf['batch_size']}, "
                f"批次数量={len(self.train_loader)}"
            )
            
        except Exception as e:
            self.logger.error(f"设置训练数据失败: {str(e)}")
            raise

    def update_learning_rate(self, new_lr):
        """更新学习率"""
        try:
            if not hasattr(self, 'optimizer'):
                raise AttributeError("Optimizer not initialized")

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.logger.debug(f"Learning rate updated to {new_lr}")
        except Exception as e:
            self.logger.error(f"Error updating learning rate: {str(e)}")
            raise

    def local_train(self, model, global_epoch):
        """本地训练过程"""
        try:
            # 复制全局模型参数到本地模型
            self.local_model.load_state_dict(copy.deepcopy(model.state_dict()))
            self.local_model.train()
            
            best_acc = 0
            best_state = None
            
            # 记录差分隐私设置
            self.logger.info(
                f"客户端 {self.client_id} 使用 {self.conf['dp_noise_type']} "
                f"差分隐私机制, ε={self.conf['epsilon']}, "
                f"噪声比例={self.conf['dp_noise_scale']}"
            )
            
            # 检查训练数据是否为空
            if len(self.train_loader) == 0:
                raise ValueError(f"客户端 {self.client_id} 的训练数据为空")
            
            self.logger.info(f"客户端 {self.client_id} 开始训练，数据批次数: {len(self.train_loader)}")

            # 训练循环
            for epoch in range(self.conf["local_epochs"]):
                epoch_loss = 0
                correct = 0
                total = 0
                total_noise_magnitude = 0  # 记录每个epoch的总噪声大小
                total_sensitivity = 0      # 记录每个epoch的总敏感度

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if data.size(0) == 0:
                        self.logger.warning(f"跳过空批次 {batch_idx}")
                        continue
                        
                    data, target = data.to(self.device), target.to(self.device).long()
                    
                    self.optimizer.zero_grad()
                    outputs = self.local_model(data)
                    scaled_outputs = outputs / self.temperature
                    loss = torch.nn.functional.cross_entropy(scaled_outputs, target)
                    loss.backward()

                    # 计算当前批次的梯度敏感度
                    batch_sensitivity = 0
                    batch_noise_magnitude = 0
                    
                    # 在更新参数前添加差分隐私噪声
                    for param in self.local_model.parameters():
                        if param.grad is not None:
                            # 计算梯度敏感度
                            grad_norm = torch.norm(param.grad).item()
                            batch_sensitivity += grad_norm
                            
                            # 动态调整噪声比例
                            adaptive_noise_scale = min(
                                self.conf['dp_noise_scale'],
                                self.conf['dp_noise_scale'] * (1.0 / max(grad_norm, 1e-6))
                            )
                            
                            # 应用差分隐私
                            noise_type = self.conf.get('dp_noise_type', 'gaussian')
                            epsilon = float(self.conf.get('epsilon', 1.0))
                            delta = float(self.conf.get('delta', 1e-5))
                            sensitivity = grad_norm * adaptive_noise_scale
                            
                            # 应用差分隐私到梯度
                            noised_grad = self.apply_differential_privacy(
                                param_diff=param.grad,
                                noise_type=noise_type,
                                epsilon=epsilon,
                                sensitivity=sensitivity,
                                delta=delta
                            )
                            
                            # 计算添加的噪声大小
                            noise_magnitude = torch.norm(noised_grad - param.grad).item()
                            batch_noise_magnitude += noise_magnitude
                            
                            # 更新梯度
                            param.grad = noised_grad
                    
                    # 梯度裁剪
                    max_norm = self.conf.get('max_grad_norm', 0.5)
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.parameters(),
                        max_norm
                    )

                    self.optimizer.step()

                    # 更新统计信息
                    total_sensitivity += batch_sensitivity
                    total_noise_magnitude += batch_noise_magnitude
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    # 记录每个批次的噪声信息
                    if batch_idx % 50 == 0:
                        self.logger.info(
                            f'Client {self.client_id}, Global Epoch: {global_epoch}, Local Epoch: {epoch}, '
                            f'Batch: [{batch_idx}/{len(self.train_loader)}], '
                            f'Loss: {loss.item():.4f}, '
                            f'Batch Sensitivity: {batch_sensitivity:.6f}, '
                            f'Noise Magnitude: {batch_noise_magnitude:.6f}, '
                            f'Noise/Gradient Ratio: {(batch_noise_magnitude/batch_sensitivity):.6f}, '
                            f'Effective Privacy Budget: {epsilon * (1 - batch_noise_magnitude/batch_sensitivity):.6f}'
                        )

                # 更新学习率
                self.scheduler.step()
                
                # 计算epoch统计信息
                accuracy = 100. * correct / total
                avg_loss = epoch_loss / len(self.train_loader)
                avg_noise = total_noise_magnitude / len(self.train_loader)
                avg_sensitivity = total_sensitivity / len(self.train_loader)
                
                # 保存最佳模型
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_state = copy.deepcopy(self.local_model.state_dict())
                
                self.logger.info(
                    f'Client {self.client_id}, Global Epoch: {global_epoch}, Local Epoch: {epoch}, '
                    f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, '
                    f'Avg Noise: {avg_noise:.6f}, '
                    f'Avg Sensitivity: {avg_sensitivity:.6f}, '
                    f'LR: {self.scheduler.get_last_lr()[0]:.6f}'
                )

            # 恢复最佳模型状态
            if best_state is not None:
                self.local_model.load_state_dict(best_state)

            # 在每个epoch结束时记录当前状态
            self.current_loss = avg_loss
            self.current_accuracy = accuracy
            self.current_sensitivity = avg_sensitivity
            self.current_noise_magnitude = avg_noise
            self.current_noise_ratio = avg_noise / avg_sensitivity if avg_sensitivity != 0 else 0
            self.current_privacy_budget = self.conf['epsilon'] * (1 - self.current_noise_ratio)

            # 保存当前epoch的权重
            self.save_model_weights(global_epoch)

            # 返回模型更新
            return self.compute_model_update_without_noise(model)

        except Exception as e:
            self.logger.error(f"客户端 {self.client_id} 训练错误: {str(e)}")
            raise

    def save_model_weights(self, epoch):
        """保存每个epoch的模型权重"""
        try:
            # 生成权重文件名
            filename = f'epoch_{epoch}.pt'
            save_path = os.path.join(self.weights_dir, filename)

            # 保存模型状态
            state_dict = {
                'model_state_dict': self.local_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'client_id': self.client_id,
                'training_stats': {
                    'loss': self.current_loss,
                    'accuracy': self.current_accuracy,
                    'sensitivity': self.current_sensitivity,
                    'noise_magnitude': self.current_noise_magnitude,
                    'noise_ratio': self.current_noise_ratio,
                    'effective_privacy_budget': self.current_privacy_budget
                },
                'dp_config': {
                    'noise_type': self.conf['dp_noise_type'],
                    'epsilon': self.conf['epsilon'],
                    'noise_scale': self.conf['dp_noise_scale']
                }
            }
            
            torch.save(state_dict, save_path)
            self.logger.info(f'已保存epoch {epoch}的模型权重到 {save_path}')

        except Exception as e:
            self.logger.error(f"保存模型权重失败: {str(e)}")
            raise

    def compute_sensitivity(self, global_model):
        """计算模型更新的敏感度"""
        sensitivity = 0.0
        for name, local_param in self.local_model.state_dict().items():
            global_param = global_model.state_dict()[name]

            # 确保所有参数在同一设备上并转换为浮点类型
            local_param = local_param.to(self.device).float()
            global_param = global_param.to(self.device).float()

            param_diff = torch.norm(local_param.detach() - global_param.detach(), p=2)
            if param_diff > sensitivity:
                sensitivity = param_diff.item()
        return sensitivity

    def apply_gaussian_noise(self, param_diff, epsilon, delta, sensitivity):
        """
        应用高斯机制
        :param param_diff: 参数差异
        :param epsilon: 隐私预算
        :param delta: 松弛参数
        :param sensitivity: 敏感度
        :return: 添加噪声后的参数差异
        """
        try:
            sigma = float(np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon)
            noise = torch.normal(
                mean=torch.zeros_like(param_diff),
                std=torch.full_like(param_diff, sigma)
            ).float()
            return param_diff + noise
        except Exception as e:
            self.logger.error(f"Error in Gaussian mechanism: {str(e)}")
            raise

    def apply_laplace_noise(self, param_diff, epsilon, sensitivity):
        """
        应用拉普拉斯机制
        :param param_diff: 参数差异
        :param epsilon: 隐私预算
        :param sensitivity: 敏感度
        :return: 添加噪声后的参数差异
        """
        try:
            b = float(sensitivity / epsilon)
            noise = torch.from_numpy(
                np.random.laplace(0, b, size=param_diff.size())
            ).float().to(param_diff.device)
            return param_diff + noise
        except Exception as e:
            self.logger.error(f"Error in Laplace mechanism: {str(e)}")
            raise

    def apply_exponential_noise(self, param_diff, epsilon, sensitivity):
        """
        应用指数机制
        :param param_diff: 参数差异
        :param epsilon: 隐私预算
        :param sensitivity: 敏感度
        :return: 添加噪声后的参数差异
        """
        try:
            param_diff = param_diff.float()
            
            def utility_function(noise):
                noised_diff = param_diff + noise.float()
                return -float(torch.norm(noised_diff).item())

            noise_candidates = []
            scale = float(sensitivity / epsilon)
            for _ in range(10):
                noise = torch.randn_like(param_diff).float() * scale
                noise_candidates.append(noise)

            utility_values = [utility_function(noise) for noise in noise_candidates]
            # 将效用值转换为非负数
            scores = torch.tensor(utility_values, dtype=torch.float32, device=param_diff.device)
            scores = scores - scores.min()  # 确保所有值非负
            
            # 处理概率计算
            exp_scores = torch.exp(torch.tensor(epsilon, dtype=torch.float32) * 
                                 scores / (2.0 * float(sensitivity)))
            # 确保概率和为1且非负
            probabilities = exp_scores / exp_scores.sum()
            # 处理数值稳定性
            probabilities = torch.clamp(probabilities, min=0.0, max=1.0)
            probabilities = probabilities / probabilities.sum()

            # 确保概率张量为浮点类型
            selected_idx = int(torch.multinomial(probabilities.float(), 1).item())
            selected_noise = noise_candidates[selected_idx].float()

            return param_diff + selected_noise
        except Exception as e:
            self.logger.error(f"Error in Exponential mechanism: {str(e)}")
            raise

    def apply_differential_privacy(self, param_diff, noise_type, epsilon, sensitivity, delta=1e-5):
        """
        选择并应用差分隐私机制
        :param param_diff: 参数差异
        :param noise_type: 噪声类型 ('gaussian', 'laplace', 'exponential')
        :param epsilon: 隐私预算
        :param sensitivity: 敏感度
        :param delta: 松弛参数 (仅用于高斯机制)
        :return: 添加噪声后的参数差异
        """
        try:
            if noise_type == 'gaussian':
                return self.apply_gaussian_noise(param_diff, epsilon, delta, sensitivity)
            elif noise_type == 'laplace':
                return self.apply_laplace_noise(param_diff, epsilon, sensitivity)
            elif noise_type == 'exponential':
                return self.apply_exponential_noise(param_diff, epsilon, sensitivity)
            else:
                self.logger.warning(f"Unknown noise type: {noise_type}, no noise added")
                return param_diff
        except Exception as e:
            self.logger.error(f"Error applying differential privacy: {str(e)}")
            raise

    def compute_model_update(self, global_model):
        """计算模型更新并添加差分隐私噪声"""
        try:
            diff = {}
            noise_type = self.conf.get('dp_noise_type', 'gaussian')
            epsilon = float(self.conf.get('epsilon', 1.0))
            delta = float(self.conf.get('delta', 1e-5))
            
            # 添加梯度裁剪
            max_norm = self.conf.get('max_grad_norm', 1.0)
            
            for name, local_param in self.local_model.state_dict().items():
                global_param = global_model.state_dict()[name]
                original_dtype = global_param.dtype
                
                # 转换为浮点类型进行计算
                local_param = local_param.to(self.device).float()
                global_param = global_param.to(self.device).float()
                
                # 计算参数差异
                param_diff = local_param.detach() - global_param.detach()
                
                # 梯度裁剪
                param_norm = torch.norm(param_diff)
                if param_norm > max_norm:
                    param_diff = param_diff * max_norm / param_norm
                
                # 计算每层的敏感度
                sensitivity = torch.norm(param_diff).item()
                
                # 应用差分隐私，降低噪声强度
                noise_scale = self.conf.get('dp_noise_scale', 0.001)  # 降低默认噪声强度
                noised_diff = self.apply_differential_privacy(
                    param_diff=param_diff,
                    noise_type=noise_type,
                    epsilon=epsilon,
                    sensitivity=sensitivity * noise_scale,  # 缩放敏感度
                    delta=delta
                )
                
                # 将结果转换回原始类型
                diff[name] = noised_diff.to(dtype=original_dtype)

            return diff

        except Exception as e:
            self.logger.error(f"Error computing model update: {str(e)}")
            raise

    def compute_model_update_without_noise(self, global_model):
        """计算模型更新（不添加噪声）"""
        try:
            diff = {}
            for name, local_param in self.local_model.state_dict().items():
                global_param = global_model.state_dict()[name]
                original_dtype = global_param.dtype
                
                # 转换为浮点类型进行计算
                local_param = local_param.to(self.device).float()
                global_param = global_param.to(self.device).float()
                
                # 计算参数差异
                param_diff = local_param.detach() - global_param.detach()
                
                # 将结果转换回原始类型
                diff[name] = param_diff.to(dtype=original_dtype)

            return diff

        except Exception as e:
            self.logger.error(f"Error computing model update: {str(e)}")
            raise

    def evaluate(self, data_loader):
        """评估模型性能"""
        self.local_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        return accuracy, avg_loss