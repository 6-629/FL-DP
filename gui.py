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

class FederatedLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("联邦学习隐私保护系统")
        self.root.geometry("1100x600")
        
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
                acc, loss = self.train_one_epoch(epoch, conf)
                
                # 更新指标显示
                self.update_metrics(acc, loss)
                
                # 更新训练日志
                self.result_text.insert(tk.END, 
                    f"Epoch {epoch}, 准确率: {acc:.2f}%, 损失: {loss:.4f}\n")
                self.result_text.see(tk.END)

            messagebox.showinfo("完成", "训练完成！")
            
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def train_one_epoch(self, epoch, conf):
        import random
        candidates = random.sample(self.clients, conf["k"])
        weight_accumulator = {}

        for name, params in self.server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(self.server.global_model)
            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        self.server.model_aggregate(weight_accumulator)
        return self.server.model_eval()

    def start_recovery(self):
        try:
            if not self.clients:
                raise ValueError("请先开始训练！")

            target_id = int(self.target_var.get())
            if target_id >= len(self.clients):
                raise ValueError("目标客户端ID超出范围！")

            iterations = int(self.iter_var.get())
            
            # TODO: 实现梯度恢复逻辑
            self.recovery_text.delete(1.0, tk.END)
            self.recovery_text.insert(tk.END, f"开始恢复客户端 {target_id} 的数据...\n")
            
            # 模拟恢复结果
            mse = 0.15
            psnr = 25.5
            
            # 更新恢复指标
            self.update_recovery_metrics(mse, psnr)
            
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def update_metrics(self, accuracy=None, loss=None):
        if accuracy is not None:
            self.accuracy_var.set(f"{accuracy:.2f}")
        if loss is not None:
            self.loss_var.set(f"{loss:.4f}")

    def update_recovery_metrics(self, mse=None, psnr=None):
        if mse is not None:
            self.mse_var.set(f"{mse:.6f}")
        if psnr is not None:
            self.psnr_var.set(f"{psnr:.2f}")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = FederatedLearningGUI(root)
    root.mainloop()
