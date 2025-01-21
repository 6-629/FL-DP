import tkinter as tk
from tkinter import ttk, filedialog

# 创建主窗口
root = tk.Tk()
root.title("联邦学习隐私保护系统")
root.geometry("1100x600")

# 创建左侧框架 - 基本设置
left_frame = tk.LabelFrame(root, text="基本设置", padx=10, pady=10)
left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH)

# 差分隐私方案选择
dp_label = tk.Label(left_frame, text="差分隐私方案:")
dp_label.pack(anchor='w')
dp_var = tk.StringVar(value="Laplace")
dp_modes = ["Laplace", "Gaussian", "Exponential"]
dp_menu = ttk.Combobox(left_frame, textvariable=dp_var, values=dp_modes)
dp_menu.pack(anchor='w', pady=(0,10))

# 数据集选择
dataset_label = tk.Label(left_frame, text="数据集选择:")
dataset_label.pack(anchor='w')
dataset_var = tk.StringVar(value="CIFAR-10")
datasets = ["CIFAR-10", "MNIST"]
dataset_menu = ttk.Combobox(left_frame, textvariable=dataset_var, values=datasets)
dataset_menu.pack(anchor='w', pady=(0,10))

# 客户端数量设置
clients_label = tk.Label(left_frame, text="客户端数量:")
clients_label.pack(anchor='w')
clients_var = tk.StringVar(value="10")
clients_entry = ttk.Spinbox(left_frame, from_=5, to=50, textvariable=clients_var)
clients_entry.pack(anchor='w', pady=(0,10))

# 隐私预算设置
privacy_label = tk.Label(left_frame, text="隐私预算 (ε):")
privacy_label.pack(anchor='w')
privacy_var = tk.StringVar(value="0.01")
privacy_entry = ttk.Entry(left_frame, textvariable=privacy_var, width=10)
privacy_entry.pack(anchor='w', pady=(0,10))


# 创建右侧框架 - 训练控制
right_frame = tk.LabelFrame(root, text="训练控制", padx=10, pady=10)
right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH)

# 保存路径显示框
path_label = tk.Label(right_frame, text="模型保存路径:")
path_label.pack(anchor='w')
path_var = tk.StringVar()
path_entry = tk.Entry(right_frame, textvariable=path_var, width=40, state='readonly')
path_entry.pack(anchor='w', pady=(0,10))

# 启动训练按钮
def start_training():
    save_path = filedialog.askdirectory(title="选择保存路径")
    if save_path:
        path_var.set(save_path)
        print(f"开始训练:\n"
              f"差分隐私方案: {dp_var.get()}\n"
              f"数据集: {dataset_var.get()}\n"
              f"客户端数量: {clients_var.get()}\n"
              f"隐私预算: {privacy_var.get()}")

# 训练控制按钮
start_button = ttk.Button(right_frame, text="开始训练", command=start_training)
start_button.pack(anchor='w', pady=10)

# 参数聚合按钮
def aggregate_params():
    print(f"执行参数聚合，使用{dp_var.get()}差分隐私方案")

aggregate_button = ttk.Button(right_frame, text="参数聚合", command=aggregate_params)
aggregate_button.pack(anchor='w')

# 添加进度显示
progress_label = tk.Label(right_frame, text="训练进度:")
progress_label.pack(anchor='w', pady=(20,0))
progress_bar = ttk.Progressbar(right_frame, length=300, mode='determinate')
progress_bar.pack(anchor='w', pady=(0,10))

# 结果显示区域
result_label = tk.Label(right_frame, text="训练结果:")
result_label.pack(anchor='w')
result_text = tk.Text(right_frame, height=10, width=40)
result_text.pack(anchor='w', pady=(0,10))

# 创建中间框架 - 数据恢复控制
middle_frame = tk.LabelFrame(root, text="数据恢复测试", padx=10, pady=10)
middle_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH)

# 恢复方法选择
recovery_label = tk.Label(middle_frame, text="恢复方法:")
recovery_label.pack(anchor='w')
recovery_var = tk.StringVar(value="梯度恢复")
recovery_modes = ["梯度恢复", "深度泄漏", "模型反演"]
recovery_menu = ttk.Combobox(middle_frame, textvariable=recovery_var, values=recovery_modes)
recovery_menu.pack(anchor='w', pady=(0,10))

# 目标客户端选择
target_label = tk.Label(middle_frame, text="目标客户端ID:")
target_label.pack(anchor='w')
target_var = tk.StringVar(value="1")
target_entry = ttk.Spinbox(middle_frame, from_=1, to=50, textvariable=target_var)
target_entry.pack(anchor='w', pady=(0,10))

# 迭代次数设置
iter_label = tk.Label(middle_frame, text="恢复迭代次数:")
iter_label.pack(anchor='w')
iter_var = tk.StringVar(value="1000")
iter_entry = ttk.Entry(middle_frame, textvariable=iter_var, width=10)
iter_entry.pack(anchor='w', pady=(0,10))

# 开始恢复按钮
def start_recovery():
    print(f"开始数据恢复:\n"
          f"恢复方法: {recovery_var.get()}\n"
          f"目标客户端: {target_var.get()}\n"
          f"迭代次数: {iter_var.get()}")
    # TODO: 实现数据恢复逻辑

recovery_button = ttk.Button(middle_frame, text="开始恢复", command=start_recovery)
recovery_button.pack(anchor='w', pady=10)

# 恢复结果显示
recovery_result_label = tk.Label(middle_frame, text="恢复结果:")
recovery_result_label.pack(anchor='w')
recovery_text = tk.Text(middle_frame, height=8, width=35)
recovery_text.pack(anchor='w', pady=(0,10))

# MSE显示
mse_label = tk.Label(middle_frame, text="恢复误差(MSE):")
mse_label.pack(anchor='w')
mse_var = tk.StringVar(value="0.0")
mse_display = ttk.Entry(middle_frame, textvariable=mse_var, state='readonly', width=10)
mse_display.pack(anchor='w', pady=(0,10))

# 添加主循环
root.mainloop()
