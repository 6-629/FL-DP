import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from client import Client


# 测试案例
def test_client_dp():
    # 1. 创建测试配置
    conf = {
        'no_models': 5,  # 客户端数量
        'batch_size': 32,
        'local_epochs': 2,
        'lr': 0.0001,  # 降低学习率
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'dp_noise_type': 'gaussian',  # 改用高斯噪声
        'dp_noise_scale': 0.001,  # 降低噪声强度
        'epsilon': 1.0,
        'delta': 1e-5,
        'clip_grad': 0.5,  # 降低梯度裁剪阈值
        'max_grad_norm': 0.5  # 添加最大梯度范数限制
    }

    # 2. 准备测试数据（使用MNIST）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 3. 创建更复杂的模型
    class ComplexModel(nn.Module):
        def __init__(self):
            super(ComplexModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x

    global_model = ComplexModel().cuda()  # 确保模型在GPU上

    # 4. 初始化所有客户端
    clients = []
    for i in range(conf['no_models']):
        client = Client(
            conf=conf,
            model=global_model,
            train_dataset=train_dataset,
            id=i
        )
        clients.append(client)

    print("\n=== 测试多个客户端训练 ===")
    
    # 5. 测试每个客户端的训练
    for client in clients:
        print(f"\n测试客户端 {client.client_id}:")
        
        # 记录原始参数
        original_params = copy.deepcopy(global_model.state_dict())
        
        # 在每个批次训练后添加差分隐私
        for epoch in range(conf['local_epochs']):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(client.train_loader):
                data, target = data.cuda(), target.cuda()
                
                # 前向传播和损失计算
                client.optimizer.zero_grad()
                output = client.local_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    client.local_model.parameters(),
                    conf['clip_grad']
                )
                
                # 应用差分隐私
                for name, param in client.local_model.named_parameters():
                    if param.grad is not None:
                        sensitivity = torch.norm(param.grad).item()
                        noised_grad = client.apply_differential_privacy(
                            param.grad,
                            conf['dp_noise_type'],
                            conf['epsilon'],
                            sensitivity * conf['dp_noise_scale'],
                            conf['delta']
                        )
                        param.grad = noised_grad
                
                client.optimizer.step()
                running_loss += loss.item()
                
                if batch_idx % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        # 计算并打印更新的统计信息
        update_norm = 0.0
        noise_norm = 0.0
        for name, param in client.local_model.named_parameters():
            if name in original_params:
                param_diff = param.data - original_params[name].cuda()
                update_norm += torch.norm(param_diff).item()
                
        print(f"更新量总范数: {update_norm:.4f}")
        sensitivity = client.compute_sensitivity(global_model)
        print(f"敏感度: {sensitivity:.4f}")

    print("\n=== 所有客户端测试完成 ===")


if __name__ == "__main__":
    # 运行测试案例
    print("=== 开始差分隐私客户端测试 ===")
    test_client_dp()
    print("\n=== 所有测试通过 ===")
