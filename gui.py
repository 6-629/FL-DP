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
from datasets import get_dataset
import time
import random  # 添加在文件顶部的import部分
import numpy as np

class FederatedLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("联邦学习隐私保护系统")
        self.root.geometry("900x600")
        
        # 添加设备初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 添加当前轮次跟踪
        self.current_epoch = 0
        
        # 配置日志记录器，添加文件处理器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('federated_learning.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FederatedLearningGUI")
        
        # 保存当前的客户端和服务器实例
        self.clients = []
        self.server = None
        self.global_model = None
        
        # 添加成员变量存储第一轮选择的客户端
        self.selected_clients = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # 创建左侧框架 - 基本设置
        left_frame = tk.LabelFrame(self.root, text="基本设置", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH)

        # 差分隐私方案选择
        dp_label = tk.Label(left_frame, text="差分隐私方案:")
        dp_label.pack(anchor='w')
        self.dp_var = tk.StringVar(value="None")
        self.dp_modes = ["None", "Laplace", "Gaussian", "Exponential"]
        self.dp_menu = ttk.Combobox(left_frame, textvariable=self.dp_var, values=self.dp_modes)
        self.dp_menu.pack(anchor='w', pady=(0,10))

        # 数据集选择
        dataset_label = tk.Label(left_frame, text="数据集选择:")
        dataset_label.pack(anchor='w')
        self.dataset_var = tk.StringVar(value="CIFAR-10")
        self.datasets = [
            "CIFAR-10",
            "CIFAR-100",
            "VOC2007"  # 添加VOC2007选项
        ]
        self.dataset_menu = ttk.Combobox(left_frame, textvariable=self.dataset_var, values=self.datasets)
        self.dataset_menu.pack(anchor='w', pady=(0,10))

        # 客户端数量设置
        clients_label = tk.Label(left_frame, text="客户端数量:")
        clients_label.pack(anchor='w')
        self.clients_var = tk.StringVar(value="10")
        self.clients_entry = ttk.Spinbox(left_frame, from_=4, to=50, textvariable=self.clients_var)
        self.clients_entry.pack(anchor='w', pady=(0,10))

        # 添加k值设置
        k_label = tk.Label(left_frame, text="每轮选择的客户端数量(k):")
        k_label.pack(anchor='w')
        self.k_var = tk.StringVar(value="4")
        self.k_entry = ttk.Spinbox(
            left_frame, 
            from_=1, 
            to=50,  # 最大值将在验证时动态调整
            textvariable=self.k_var,
            width=10
        )
        self.k_entry.pack(anchor='w', pady=(0,10))

        # 添加客户端数量变化时的回调函数，用于更新k值的范围
        def update_k_max(*args):
            try:
                num_clients = int(self.clients_var.get())
                current_k = int(self.k_var.get())
                # 更新k的最大值
                self.k_entry.configure(to=num_clients)
                # 如果当前k值大于客户端数量，则调整k值
                if current_k > num_clients:
                    self.k_var.set(str(num_clients))
            except ValueError:
                pass

        # 绑定客户端数量变化事件
        self.clients_var.trace('w', update_k_max)

        # 隐私预算设置
        privacy_label = tk.Label(left_frame, text="隐私预算 (ε):")
        privacy_label.pack(anchor='w')
        self.privacy_var = tk.StringVar(value="8.0")
        self.privacy_entry = ttk.Entry(left_frame, textvariable=self.privacy_var, width=10)
        self.privacy_entry.pack(anchor='w', pady=(0,10))

        # 添加全局轮次设置
        global_epochs_label = tk.Label(left_frame, text="全局训练轮次:")
        global_epochs_label.pack(anchor='w')
        self.global_epochs_var = tk.StringVar(value="1")
        self.global_epochs_entry = ttk.Spinbox(
            left_frame, 
            from_=1, 
            to=100, 
            textvariable=self.global_epochs_var,
            width=10
        )
        self.global_epochs_entry.pack(anchor='w', pady=(0,10))

        # 添加本地训练轮次设置
        local_epochs_label = tk.Label(left_frame, text="本地训练轮次:")
        local_epochs_label.pack(anchor='w')
        self.local_epochs_var = tk.StringVar(value="2")
        self.local_epochs_entry = ttk.Spinbox(
            left_frame, 
            from_=1, 
            to=50, 
            textvariable=self.local_epochs_var,
            width=10
        )
        self.local_epochs_entry.pack(anchor='w', pady=(0,10))

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

        # 修改目标客户端选择控件
        target_label = tk.Label(right_frame, text="目标客户端ID:")
        target_label.pack(anchor='w')
        self.target_var = tk.StringVar(value="0")  # 从0开始
        
        def update_target_max(*args):
            try:
                num_clients = int(self.clients_var.get())
                current_target = int(self.target_var.get())
                # 更新目标客户端ID的最大值
                self.target_entry.configure(to=num_clients-1)  # 最大值为客户端数量-1
                # 如果当前选择的ID大于最大值，则调整
                if current_target >= num_clients:
                    self.target_var.set(str(num_clients-1))
            except ValueError:
                pass

        self.target_entry = ttk.Spinbox(
            right_frame, 
            from_=0,  # 从0开始
            to=50, 
            textvariable=self.target_var,
            width=10
        )
        self.target_entry.pack(anchor='w', pady=(0,10))

        # 绑定客户端数量变化事件，同时更新目标客户端ID的范围
        self.clients_var.trace('w', lambda *args: [update_k_max(*args), update_target_max(*args)])

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

        # 在右侧框架中添加原始PSNR显示
        original_psnr_label = tk.Label(right_frame, text="原始PSNR:")
        original_psnr_label.pack(anchor='w')
        self.original_psnr_var = tk.StringVar(value="0.0")
        self.original_psnr_display = ttk.Entry(right_frame, textvariable=self.original_psnr_var, 
                                             state='readonly', width=10)
        self.original_psnr_display.pack(anchor='w', pady=(0,10))

        # 修改现有PSNR标签使其更清晰
        psnr_label = tk.Label(right_frame, text="恢复后PSNR:")
        psnr_label.pack(anchor='w')
        self.psnr_var = tk.StringVar(value="0.0")
        self.psnr_display = ttk.Entry(right_frame, textvariable=self.psnr_var, state='readonly', width=10)
        self.psnr_display.pack(anchor='w', pady=(0,10))

        # 更新评价指标的函数
        self.update_recovery_metrics = self.update_recovery_metrics

    def start_training(self):
        try:
            # 1. 从GUI获取训练配置
            dataset_type = self.dataset_var.get().lower()  # 先获取数据集类型
            
            # 根据数据集类型设置类别数
            num_classes = 20 if dataset_type == "voc2007" else (100 if dataset_type == "cifar-100" else 10)
            
            conf = {
                "model_name": "resnet18",
                "no_models": int(self.clients_var.get()),
                "type": dataset_type,
                "global_epochs": int(self.global_epochs_var.get()),
                "local_epochs": int(self.local_epochs_var.get()),
                "k": int(self.k_var.get()),
                "batch_size": 32,
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "dp_noise_type": self.dp_var.get().lower(),
                "dp_noise_scale": 0.0001,
                "epsilon": float(self.privacy_var.get()),
                "clip_grad": 1.0,
                "max_grad_norm": 1.0,
                "delta": 1e-5,
                "scheduler_step": 5,
                "scheduler_gamma": 0.1,
                "num_classes": num_classes  # 使用正确的类别数
            }

            # 如果选择了 None，则禁用差分隐私相关的参数
            if conf["dp_noise_type"] == "none":
                conf["dp_noise_scale"] = 0.0
                conf["epsilon"] = float('inf')
                conf["clip_grad"] = float('inf')
                conf["max_grad_norm"] = float('inf')

            # 验证输入值
            if conf["global_epochs"] < 1:
                raise ValueError("全局训练轮次必须大于0")
            if conf["local_epochs"] < 1:
                raise ValueError("本地训练轮次必须大于0")

            # 2. 准备数据集
            try:
                # 使用统一的数据集加载函数
                train_dataset, eval_dataset = get_dataset('./data', conf["type"])
                self.logger.info(
                    f"成功加载{conf['type'].upper()}数据集 - "
                    f"训练集: {len(train_dataset)}张, "
                    f"测试集: {len(eval_dataset)}张"
                )
            except Exception as e:
                raise ValueError(f"加载{conf['type'].upper()}数据集失败: {str(e)}")

            # 3. 初始化服务器和客户端
            self.server = Server(conf, eval_dataset)
            self.global_model = self.server.global_model
            self.clients = []
            
            for c in range(conf["no_models"]):
                self.clients.append(Client(conf, self.global_model, train_dataset, c))

            # 4. 开始训练
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始训练...\n")
            
            # 更新进度条最大值
            self.progress_bar["maximum"] = conf["global_epochs"]
            
            for epoch in range(conf["global_epochs"]):
                self.current_epoch = epoch
                # 更新进度条
                self.progress_bar["value"] = (epoch + 1) * 100 / conf["global_epochs"]
                self.root.update_idletasks()

                # 训练过程
                try:
                    # 训练一个epoch
                    _, _ = self.train_one_epoch(epoch, conf)
                    
                    # 不再在这里更新指标，因为现在是手动聚合模式
                    # 而是添加提示信息
                    log_message = f"Epoch {epoch} 训练完成，请点击'参数聚合'按钮进行模型聚合和评估\n"
                    self.result_text.insert(tk.END, log_message)
                    self.result_text.see(tk.END)
                    
                    # 清空当前指标显示
                    self.update_metrics(0.00, 0.0000)
                    
                except Exception as e:
                    error_message = f"训练错误在第 {epoch} 轮:\n{str(e)}"
                    self.result_text.insert(tk.END, error_message + "\n")
                    self.logger.error(error_message)
                    raise

            # 训练完成后计算PSNR
            self.calculate_noisy_psnr()
            
            self.progress_bar["value"] = 100
            messagebox.showinfo("完成", "所有训练轮次完成！请进行最终的参数聚合。")
            
        except Exception as e:
            error_message = f"训练错误: {str(e)}"
            self.result_text.insert(tk.END, error_message + "\n")
            messagebox.showerror("错误", error_message)
            self.logger.error(error_message)

    def train_one_epoch(self, epoch, conf):
        """训练一个epoch，不进行自动聚合"""
        try:
            # 修改判断条件，使用epoch == 0作为第一轮的判断
            if self.selected_clients is None:
                self.selected_clients = random.sample(self.clients, conf["k"])
                self.logger.info(f"第一轮随机选择的客户端: {[c.client_id for c in self.selected_clients]}")
            else:
                self.logger.info(f"使用第一轮选择的客户端: {[c.client_id for c in self.selected_clients]}")
            
            # 本地训练阶段
            for c in self.selected_clients:
                self.logger.info(f"客户端 {c.client_id} 开始训练...")
                # 只进行本地训练，不返回差异值
                c.local_train(self.server.global_model, epoch)
                self.logger.info(f"客户端 {c.client_id} 完成训练")
            
            # 提示用户进行手动聚合
            self.result_text.insert(tk.END, 
                f"Epoch {epoch} 完成。请点击'参数聚合'按钮进行模型聚合。\n")
            self.result_text.see(tk.END)
            
            # 返回占位值（因为不进行自动评估）
            return 0, 0
            
        except Exception as e:
            self.logger.error(f"训练错误在第 {epoch} 轮:\n{str(e)}")
            raise

    def start_recovery(self):
        try:
            if not self.clients:
                raise ValueError("请先开始训练！")

            target_id = int(self.target_var.get())
            
            # 检查目标客户端ID是否有效
            if target_id < 0 or target_id >= len(self.clients):
                raise ValueError(f"目标客户端ID {target_id} 无效！有效范围: 0-{len(self.clients)-1}")

            # 检查目标客户端的模型文件是否存在
            client_dir = os.path.join('model_weights', f'client_{target_id}')
            if not os.path.exists(client_dir):
                raise FileNotFoundError(f"未找到客户端 {target_id} 的模型目录: {client_dir}")
            
            # 获取最新的epoch文件
            weight_files = [f for f in os.listdir(client_dir) if f.startswith('epoch_') and f.endswith('.pt')]
            if not weight_files:
                raise FileNotFoundError(f"客户端 {target_id} 的目录中没有模型文件")
            
            # 获取最新的epoch数
            latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in weight_files])
            
            iterations = int(self.iter_var.get())
            
            self.recovery_text.delete(1.0, tk.END)
            self.recovery_text.insert(tk.END, f"开始恢复客户端 {target_id} 的数据...\n")
            self.recovery_text.insert(tk.END, f"使用epoch_{latest_epoch}.pt模型文件\n")
            
            # 创建恢复器实例，使用最新的epoch
            recovery = ModelRecovery(self.global_model, self.clients[target_id], latest_epoch)
            
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
            import traceback
            error_info = traceback.format_exc()
            error_message = f"恢复错误:\n{str(e)}\n位置: {error_info}"
            self.recovery_text.insert(tk.END, error_message + "\n")
            self.logger.error(error_message)

    def update_metrics(self, accuracy=None, loss=None):
        """更新训练指标显示"""
        try:
            if accuracy is not None:
                # 直接设置StringVar，确保格式正确
                self.accuracy_var.set(f"{float(accuracy):.2f}")
            if loss is not None:
                # 直接设置StringVar，确保格式正确
                self.loss_var.set(f"{float(loss):.4f}")
            # 强制更新GUI
            self.root.update_idletasks()
        except Exception as e:
            self.logger.error(f"更新指标显示错误: {str(e)}")

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
        """手动触发的参数聚合过程"""
        try:
            # 1. 检查权重文件目录
            weights_base_dir = 'model_weights'
            if not os.path.exists(weights_base_dir):
                raise FileNotFoundError("未找到模型权重目录")
            
            # 2. 获取客户端目录
            client_dirs = [d for d in os.listdir(weights_base_dir) 
                         if os.path.isdir(os.path.join(weights_base_dir, d)) 
                         and d.startswith('client_')]
            
            if not client_dirs:
                raise FileNotFoundError("未找到任何客户端权重文件夹")
            
            self.result_text.insert(tk.END, f"开始聚合参数，发现{len(client_dirs)}个客户端的权重文件...\n")
            
            # 3. 初始化或检查服务器
            if not hasattr(self, 'server') or self.server is None:
                temp_conf = {
                    "model_name": "resnet18",
                    "batch_size": 64,
                    "type": self.dataset_var.get().lower()
                }
                
                # 尝试加载评估数据集
                _, eval_dataset = get_dataset('./data', temp_conf["type"])
                self.server = Server(temp_conf, eval_dataset)
            
            # 4. 初始化权重累加器
            weight_accumulator = {}
            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)
            
            # 5. 加载和累加权重
            loaded_weights_count = 0
            for client_dir in client_dirs:
                client_path = os.path.join(weights_base_dir, client_dir)
                weight_files = [f for f in os.listdir(client_path) 
                              if f.endswith('.pt')]
                
                if weight_files:
                    # 获取最新的权重文件
                    latest_weight = sorted(weight_files, 
                                        key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
                    weight_path = os.path.join(client_path, latest_weight)
                    
                    # 加载权重
                    state_dict = torch.load(weight_path)
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        model_weights = state_dict['model_state_dict']
                    else:
                        model_weights = state_dict
                    
                    # 累加权重，确保类型匹配
                    for name, params in model_weights.items():
                        if name in weight_accumulator:
                            # 获取目标参数的类型
                            target_dtype = weight_accumulator[name].dtype
                            # 确保参数类型匹配后再累加
                            if params.dtype != target_dtype:
                                params = params.to(dtype=target_dtype)
                            weight_accumulator[name].add_(params)
                    
                    loaded_weights_count += 1
                    self.result_text.insert(tk.END, 
                        f"已加载客户端 {client_dir} 的权重文件: {latest_weight}\n")
            
            # 6. 计算平均值
            if loaded_weights_count == 0:
                raise ValueError("未能加载任何权重文件")
                
            for name in weight_accumulator:
                # 获取原始参数的类型
                original_dtype = self.server.global_model.state_dict()[name].dtype
                # 使用float32进行除法运算，然后转换回原始类型
                weight_accumulator[name] = (weight_accumulator[name].float() / loaded_weights_count).to(dtype=original_dtype)
            
            # 7. 更新全局模型，确保类型匹配
            new_state_dict = {}
            for name, param in weight_accumulator.items():
                target_dtype = self.server.global_model.state_dict()[name].dtype
                new_state_dict[name] = param.to(dtype=target_dtype)
            
            self.server.global_model.load_state_dict(new_state_dict)
            
            # 8. 评估聚合后的模型
            if hasattr(self.server, 'eval_loader'):
                acc, loss = self.server.model_eval()
                # 确保使用正确的格式更新指标
                self.accuracy_var.set(f"{acc:.2f}")  # 直接设置StringVar
                self.loss_var.set(f"{loss:.4f}")     # 直接设置StringVar
                
                # 添加到结果文本
                self.result_text.insert(tk.END, 
                    f"聚合完成，准确率: {acc:.2f}%, 损失: {loss:.4f}\n")
                self.result_text.see(tk.END)
            
            # 9. 保存全局模型
            global_model_dir = os.path.join('model_weights', 'global')
            os.makedirs(global_model_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            global_model_path = os.path.join(global_model_dir, f'global_model_{timestamp}.pt')
            
            torch.save({
                'model_state_dict': self.server.global_model.state_dict(),
                'aggregation_time': timestamp,
                'clients_aggregated': len(client_dirs),
                'weights_loaded': loaded_weights_count
            }, global_model_path)
            
            self.result_text.insert(tk.END, f"全局模型已保存至: {global_model_path}\n")
            self.result_text.see(tk.END)
            
        except Exception as e:
            error_message = f"聚合错误: {str(e)}"
            self.result_text.insert(tk.END, error_message + "\n")
            messagebox.showerror("错误", error_message)
            self.logger.error(error_message)

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

    def aggregate_models(self):
        """手动触发模型聚合"""
        try:
            # 执行模型聚合
            self.server.aggregate_weights()
            self.logger.info("模型聚合完成")
            
            # 聚合后评估全局模型
            accuracy, loss = self.server.model_eval()
            self.logger.info(f"全局模型评估 - 准确率: {accuracy:.2f}%, 损失: {loss:.4f}")
            
            # 更新GUI显示
            self.result_text.insert(tk.END, f"聚合后评估 - 准确率: {accuracy:.2f}%, 损失: {loss:.4f}\n")
            self.result_text.see(tk.END)
            
            return accuracy, loss
            
        except Exception as e:
            self.logger.error(f"模型聚合错误: {str(e)}")
            self.result_text.insert(tk.END, f"模型聚合失败: {str(e)}\n")
            self.result_text.see(tk.END)
            raise

    def calculate_noisy_psnr(self):
        """计算添加噪声后图像的PSNR值"""
        try:
            if not self.clients or not self.server:
                return
            
            # 获取第一个客户端的数据作为示例
            test_batch = next(iter(self.clients[0].train_loader))
            original_images = test_batch[0].to(self.server.device)
            noisy_images = original_images.clone()  # 创建副本用于添加噪声
            
            # 添加与训练时相同类型的噪声
            noise_type = self.dp_var.get().lower()
            noise_scale = float(self.privacy_var.get())
            
            if noise_type != "none":
                if noise_type == "gaussian":
                    noise = torch.randn_like(original_images) * noise_scale
                elif noise_type == "laplace":
                    noise = torch.from_numpy(np.random.laplace(0, noise_scale, 
                            original_images.shape)).float().to(self.server.device)
                elif noise_type == "exponential":
                    noise = torch.from_numpy(np.random.exponential(noise_scale, 
                            original_images.shape)).float().to(self.server.device)
                
                noisy_images += noise
                
                # 计算MSE和PSNR
                self.server.global_model.eval()
                with torch.no_grad():
                    # 归一化原始图像和噪声图像到[0, 1]范围
                    original_images = (original_images - original_images.min()) / (original_images.max() - original_images.min())
                    noisy_images = (noisy_images - noisy_images.min()) / (noisy_images.max() - noisy_images.min())
                    
                    # 确保值在[0, 1]范围内
                    original_images = torch.clamp(original_images, 0, 1)
                    noisy_images = torch.clamp(noisy_images, 0, 1)
                    
                    # 计算归一化后的MSE
                    mse = torch.mean((original_images - noisy_images) ** 2)
                    
                    # 创建临时ModelRecovery实例来使用其PSNR计算方法
                    recovery = ModelRecovery(self.server.global_model, self.clients[0], self.current_epoch)
                    psnr = recovery.calculate_psnr(mse)
                    
                    # 更新PSNR显示
                    self.original_psnr_var.set(f"{float(psnr):.2f}")
                    
                    # 在结果文本中显示
                    self.result_text.insert(tk.END, f"添加{noise_type}噪声后的MSE: {mse:.6f}\n")
                    self.result_text.insert(tk.END, f"添加{noise_type}噪声后的PSNR: {psnr:.2f}dB\n")
                    self.result_text.see(tk.END)
                
        except Exception as e:
            self.logger.error(f"计算PSNR时出错: {str(e)}")
            self.result_text.insert(tk.END, f"计算PSNR时出错: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = FederatedLearningGUI(root)
    root.mainloop()