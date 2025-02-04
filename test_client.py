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
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'dp_noise_type': 'gaussian',  # 测试不同噪声类型
        'dp_noise_scale': 0.5,
        'epsilon': 1.0,
        'delta': 1e-5,
        'clip_grad': 1.0  # 梯度裁剪阈值
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

    # 3. 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return self.fc(x.view(-1, 28 * 28))

    global_model = SimpleModel()

    # 4. 初始化客户端
    client = Client(
        conf=conf,
        model=global_model,
        train_dataset=train_dataset,
        id=0
    )

    # 5. 执行本地训练
    original_params = copy.deepcopy(global_model.state_dict())
    model_update = client.local_train(global_model)

    # 6. 验证结果
    print("\n=== 测试结果验证 ===")

    # 验证1: 更新参数不为空
    assert len(model_update) > 0, "模型更新为空"
    print("✅ 模型更新参数数量:", len(model_update))

    # 验证2: 参数确实发生变化
    param_changed = False
    for name in original_params:
        if not torch.equal(original_params[name], model_update[name]):
            param_changed = True
            break
    assert param_changed, "参数未发生变化"
    print("✅ 检测到参数更新")

    # 验证3: 噪声添加有效（比较更新量范数）
    update_norm = 0.0
    noise_norm = 0.0
    for name in model_update:
        update = model_update[name]
        param_diff = global_model.state_dict()[name] - original_params[name]

        update_norm += torch.norm(update).item()
        noise_norm += torch.norm(update - param_diff).item()  # 计算噪声部分

    print(f"更新量总范数: {update_norm:.4f}")
    print(f"噪声部分范数: {noise_norm:.4f}")
    assert noise_norm > 0, "未检测到噪声添加"
    print("✅ 噪声添加验证通过")

    # 验证4: 敏感度计算合理
    sensitivity = client.compute_sensitivity(global_model)
    print(f"计算敏感度: {sensitivity:.4f}")
    assert sensitivity > 0, "敏感度计算异常"
    print("✅ 敏感度验证通过")


if __name__ == "__main__":
    # 运行测试案例
    print("=== 开始差分隐私客户端测试 ===")
    test_client_dp()
    print("\n=== 所有测试通过 ===")
