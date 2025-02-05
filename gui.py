import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torchvision
import torchvision.transforms as transforms
from client import Client
from server import Server
import models
import json
import os
import logging
from model_recovery import ModelRecovery
import copy

class FederatedLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("联邦学习隐私保护系统")
        self.root.geometry("1100x600")
        
        # 初始化日志记录器
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FederatedLearningGUI")
        
        # 保存当前的客户端和服务器实例
        self.clients = []
        self.server = None
        self.global_model = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # 创建左侧框架 - 基本设置
        left_frame = tk.LabelFrame(self.root, text="基本设置", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH)

        # 差分隐私方案选择
        dp_label = tk.Label(left_frame, text="差分隐私方案:")
        dp_label.pack(anchor='w')
        self.dp_var = tk.StringVar(value="Laplace")
        self.dp_modes = ["Laplace", "Gaussian", "Exponential"]
        self.dp_menu = ttk.Combobox(left_frame, textvariable=self.dp_var, values=self.dp_modes)
        self.dp_menu.pack(anchor='w', pady=(0,10))

        # 数据集选择
        dataset_label = tk.Label(left_frame, text="数据集选择:")
        dataset_label.pack(anchor='w')
        self.dataset_var = tk.StringVar(value="CIFAR-10")
        self.datasets = ["CIFAR-10", "MNIST"]
        self.dataset_menu = ttk.Combobox(left_frame, textvariable=self.dataset_var, values=self.datasets)
        self.dataset_menu.pack(anchor='w', pady=(0,10))

        # 客户端数量设置
        clients_label = tk.Label(left_frame, text="客户端数量:")
        clients_label.pack(anchor='w')
        self.clients_var = tk.StringVar(value="10")
        self.clients_entry = ttk.Spinbox(left_frame, from_=5, to=50, textvariable=self.clients_var)
        self.clients_entry.pack(anchor='w', pady=(0,10))

        # 隐私预算设置
        privacy_label = tk.Label(left_frame, text="隐私预算 (ε):")
        privacy_label.pack(anchor='w')
        self.privacy_var = tk.StringVar(value="0.01")
        self.privacy_entry = ttk.Entry(left_frame, textvariable=self.privacy_var, width=10)
        self.privacy_entry.pack(anchor='w', pady=(0,10))

        # 创建中间框架 - 训练控制
        middle_frame = tk.LabelFrame(self.root, text="训练控制", padx=10, pady=10)
        middle_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH)

        # 保存路径显示框
        path_label = tk.Label(middle_frame, text="模型保存路径:")
        path_label.pack(anchor='w')
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(middle_frame, textvariable=self.path_var, width=40, state='readonly')
        self.path_entry.pack(anchor='w', pady=(0,10))

        # 在middle_frame中，result_text之后添加模型评价指标区域
        metrics_frame = ttk.LabelFrame(middle_frame, text="模型评价指标")
        metrics_frame.pack(anchor='w', pady=10, fill='x')

        # 准确率显示
        accuracy_frame = ttk.Frame(metrics_frame)
        accuracy_frame.pack(anchor='w', pady=5, fill='x')

        accuracy_label = ttk.Label(accuracy_frame, text="准确率:")
        accuracy_label.pack(side='left', padx=(5,10))

        self.accuracy_var = tk.StringVar(value="--")
        self.accuracy_display = ttk.Entry(accuracy_frame, textvariable=self.accuracy_var, 
                           state='readonly', width=10)
        self.accuracy_display.pack(side='left')

        self.accuracy_percent = ttk.Label(accuracy_frame, text="%")
        self.accuracy_percent.pack(side='left')

        # 损失值显示
        loss_frame = ttk.Frame(metrics_frame)
        loss_frame.pack(anchor='w', pady=5, fill='x')

        loss_label = ttk.Label(loss_frame, text="损失值:")
        loss_label.pack(side='left', padx=(5,10))

        self.loss_var = tk.StringVar(value="--")
        self.loss_display = ttk.Entry(loss_frame, textvariable=self.loss_var, 
                        state='readonly', width=10)
        self.loss_display.pack(side='left')

        # 更新评价指标的函数
        self.update_metrics = self.update_metrics

        # 启动训练按钮
        self.start_training = self.start_training
        start_button = ttk.Button(middle_frame, text="开始训练", command=self.start_training)
        start_button.pack(anchor='w', pady=10)

        # 参数聚合按钮
        self.aggregate_params = self.aggregate_params
        aggregate_button = ttk.Button(middle_frame, text="参数聚合", command=self.aggregate_params)
        aggregate_button.pack(anchor='w')

        # 添加进度显示
        progress_label = tk.Label(middle_frame, text="训练进度:")
        progress_label.pack(anchor='w', pady=(20,0))
        self.progress_bar = ttk.Progressbar(middle_frame, length=300, mode='determinate')
        self.progress_bar.pack(anchor='w', pady=(0,10))

        # 结果显示区域
        result_label = tk.Label(middle_frame, text="训练结果:")
        result_label.pack(anchor='w')
        self.result_text = tk.Text(middle_frame, height=10, width=40)
        self.result_text.pack(anchor='w', pady=(0,10))

        # 创建右侧框架 - 数据恢复控制
        right_frame = tk.LabelFrame(self.root, text="梯度恢复测试", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH)

        # 添加说明文本
        desc_label = tk.Label(right_frame, text="使用梯度恢复方法\n从服务器恢复客户端训练数据", 
                     justify=tk.LEFT, wraplength=200)
        desc_label.pack(anchor='w', pady=(0,20))

        # 目标客户端选择
        target_label = tk.Label(right_frame, text="目标客户端ID:")
        target_label.pack(anchor='w')
        self.target_var = tk.StringVar(value="1")
        self.target_entry = ttk.Spinbox(right_frame, from_=1, to=50, textvariable=self.target_var)
        self.target_entry.pack(anchor='w', pady=(0,10))

        # 迭代次数设置
        iter_label = tk.Label(right_frame, text="恢复迭代次数:")
        iter_label.pack(anchor='w')
        self.iter_var = tk.StringVar(value="1000")
        self.iter_entry = ttk.Entry(right_frame, textvariable=self.iter_var, width=10)
        self.iter_entry.pack(anchor='w', pady=(0,10))

        # 开始恢复按钮
        self.start_recovery = self.start_recovery
        recovery_button = ttk.Button(right_frame, text="开始恢复", command=self.start_recovery)
        recovery_button.pack(anchor='w', pady=10)

        # 恢复结果显示
        recovery_result_label = tk.Label(right_frame, text="恢复结果:")
        recovery_result_label.pack(anchor='w')
        self.recovery_text = tk.Text(right_frame, height=8, width=35)
        self.recovery_text.pack(anchor='w', pady=(0,10))

        # MSE显示
        mse_label = tk.Label(right_frame, text="恢复误差(MSE):")
        mse_label.pack(anchor='w')
        self.mse_var = tk.StringVar(value="0.0")
        self.mse_display = ttk.Entry(right_frame, textvariable=self.mse_var, state='readonly', width=10)
        self.mse_display.pack(anchor='w', pady=(0,10))

        # 添加PSNR显示
        psnr_label = tk.Label(right_frame, text="峰值信噪比(PSNR):")
        psnr_label.pack(anchor='w')
        self.psnr_var = tk.StringVar(value="0.0")
        self.psnr_display = ttk.Entry(right_frame, textvariable=self.psnr_var, state='readonly', width=10)
        self.psnr_display.pack(anchor='w', pady=(0,10))

        # 更新评价指标的函数
        self.update_recovery_metrics = self.update_recovery_metrics

    def start_training(self):
        try:
            # 1. 收集配置
            conf = {
                "model_name": "resnet18",
                "no_models": int(self.clients_var.get()),
                "type": self.dataset_var.get().lower(),
                "global_epochs": 8,
                "local_epochs": 2,
                "k": 5,
                "batch_size": 32,
                "lr": 0.001,
                "momentum": 0.9,
                "lambda": 0.001,
                "dp_noise_type": self.dp_var.get().lower(),
                "dp_noise_scale": 0.001,
                "epsilon": float(self.privacy_var.get()),
                "clip_grad": 1.0
            }

            # 2. 准备数据集
            if conf["type"] == "mnist":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                train_dataset = torchvision.datasets.MNIST(
                    root='./data', train=True, download=True, transform=transform)
                eval_dataset = torchvision.datasets.MNIST(
                    root='./data', train=False, download=True, transform=transform)
            else:  # cifar-10
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                train_dataset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform)
                eval_dataset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform)

            # 3. 初始化服务器和客户端
            self.server = Server(conf, eval_dataset)
            self.global_model = self.server.global_model
            self.clients = []
            
            for c in range(conf["no_models"]):
                self.clients.append(Client(conf, self.global_model, train_dataset, c))

            # 4. 开始训练
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始训练...\n")
            
            for epoch in range(conf["global_epochs"]):
                # 更新进度条
                self.progress_bar["value"] = (epoch + 1) / conf["global_epochs"] * 100
                self.root.update_idletasks()

                # 训练过程
                try:
                    acc, loss = self.train_one_epoch(epoch, conf)
                    
                    # 更新指标显示
                    self.update_metrics(acc, loss)
                    
                    # 更新训练日志
                    log_message = "Epoch {}, 准确率: {:.2f}%, 损失: {:.4f}\n".format(
                        epoch, float(acc), float(loss)
                    )
                    self.result_text.insert(tk.END, log_message)
                    self.result_text.see(tk.END)
                except Exception as e:
                    self.logger.error(f"Training error in epoch {epoch}: {str(e)}")
                    raise

            messagebox.showinfo("完成", "训练完成！")
            
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def train_one_epoch(self, epoch, conf):
        """训练一个epoch"""
        try:
            # 确保k是整数
            k = int(conf.get("k", 5))
            
            # 确保选择的客户端数量不超过可用客户端数量
            k = min(k, len(self.clients))
            
            # 不随机选择客户端，依次选择客户端
            candidates = self.clients[:k]
            weight_accumulator = {}

            # 初始化权重累加器
            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)

            # 客户端训练
            for c in candidates:
                diff = c.local_train(self.server.global_model)
                for name, params in self.server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            # 服务器聚合
            self.server.model_aggregate(weight_accumulator)
            
            # 评估模型
            acc, loss = self.server.model_eval()
            
            # 确保返回浮点数
            return float(acc), float(loss)
        
        except Exception as e:
            self.logger.error(f"Error in training epoch: {str(e)}")
            raise

    def start_recovery(self):
        try:
            if not self.clients:
                raise ValueError("请先开始训练！")

            target_id = int(self.target_var.get())
            if target_id >= len(self.clients):
                raise ValueError("目标客户端ID超出范围！")

            iterations = int(self.iter_var.get())
            
            self.recovery_text.delete(1.0, tk.END)
            self.recovery_text.insert(tk.END, f"开始恢复客户端 {target_id} 的数据...\n")
            
            # 创建恢复器实例
            recovery = ModelRecovery(self.global_model, self.clients[target_id])
            
            # 执行恢复
            recovered_data, mse, psnr = recovery.recover_data(iterations=iterations)
            
            # 保存恢复的图像
            save_path = f"recovered_client_{target_id}.png"
            recovered_data.save(save_path)
            
            # 更新GUI显示
            self.recovery_text.insert(tk.END, f"恢复完成！\n")
            self.recovery_text.insert(tk.END, f"恢复的图像已保存至: {save_path}\n")
            
            # 更新恢复指标
            self.update_recovery_metrics(mse, psnr)
            
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def update_metrics(self, accuracy=None, loss=None):
        """更新训练指标显示"""
        try:
            if accuracy is not None:
                # 将准确率限制到2位小数
                acc_str = "{:.2f}".format(float(accuracy))
                self.accuracy_var.set(acc_str)
            if loss is not None:
                # 将损失值限制到4位小数
                loss_str = "{:.4f}".format(float(loss))
                self.loss_var.set(loss_str)
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def update_recovery_metrics(self, mse=None, psnr=None):
        """更新恢复指标显示"""
        try:
            if mse is not None:
                # 将MSE限制到6位小数
                mse_str = "{:.6f}".format(float(mse))
                self.mse_var.set(mse_str)
            if psnr is not None:
                # 将PSNR限制到2位小数
                psnr_str = "{:.2f}".format(float(psnr))
                self.psnr_var.set(psnr_str)
        except Exception as e:
            self.logger.error(f"Error updating recovery metrics: {str(e)}")

    def aggregate_params(self):
        """执行参数聚合"""
        try:
            if not self.server or not self.clients:
                messagebox.showerror("错误", "请先开始训练！")
                return
                
            self.result_text.insert(tk.END, f"执行参数聚合，使用{self.dp_var.get()}差分隐私方案...\n")
            
            # 执行一轮参数聚合
            conf = {
                "k": 5  # 选择参与聚合的客户端数量
            }
            acc, loss = self.train_one_epoch(0, conf)
            
            # 更新指标显示
            self.update_metrics(acc, loss)
            
            # 更新日志
            self.result_text.insert(tk.END, f"聚合完成，准确率: {acc:.2f}%, 损失: {loss:.4f}\n")
            self.result_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def local_train(self, model):
        """本地训练过程"""
        try:
            # 复制全局模型参数到本地模型
            self.local_model.load_state_dict(copy.deepcopy(model.state_dict()))
            self.local_model.train()

            # 训练循环
            for epoch in range(self.conf["local_epochs"]):
                epoch_loss = 0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # 移动数据到正确的设备
                    data, target = data.to(self.device), target.to(self.device).long()  # 确保target是LongTensor

                    # 训练步骤
                    self.optimizer.zero_grad()
                    output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()

                    # 梯度裁剪（如果配置中指定）
                    if "clip_grad" in self.conf:
                        torch.nn.utils.clip_grad_norm_(
                            self.local_model.parameters(),
                            self.conf["clip_grad"]
                        )

                    self.optimizer.step()

                    # 计算统计信息
                    epoch_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                # 记录每个epoch的统计信息
                accuracy = 100. * correct / total
                avg_loss = epoch_loss / len(self.train_loader)
                self.logger.info(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # 计算模型更新（带差分隐私）
            return self.compute_model_update(model)

        except Exception as e:
            self.logger.error(f"Error in local training: {str(e)}")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = FederatedLearningGUI(root)
    root.mainloop()
