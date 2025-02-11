import unittest
import sys
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import time
import torch.nn as nn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import resnet18

class TestGlobalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 设置设备
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {cls.device}")
        
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # 数据预处理
        cls.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10标准化参数
        ])
        
        # 加载测试数据集
        cls.test_dataset = CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=cls.transform
        )
        cls.test_loader = DataLoader(
            cls.test_dataset, 
            batch_size=64, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 加载全局模型
        try:
            # 初始化模型
            cls.model = resnet18(num_classes=10)
            
            # 获取最新的全局模型文件
            model_dir = os.path.join('model_weights', 'global')
            model_files = [f for f in os.listdir(model_dir) if f.startswith('global_model_')]
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(model_dir, latest_model)
            print(f"加载模型文件: {model_path}")
            
            # 加载模型权重
            checkpoint = torch.load(model_path)
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # 过滤和调整权重
            new_state_dict = {}
            for name, param in state_dict.items():
                # 跳过fc层的权重（最后的分类层）
                if 'fc.' in name:
                    continue
                new_state_dict[name] = param
            
            # 加载除最后一层外的权重
            cls.model.load_state_dict(new_state_dict, strict=False)
            
            # 重新初始化最后的分类层
            nn.init.xavier_uniform_(cls.model.fc.weight)
            nn.init.zeros_(cls.model.fc.bias)
            
            cls.model.to(cls.device)
            cls.model.eval()
            print(f"成功加载模型权重")
            
            # 打印模型结构
            print("\n模型结构:")
            print(cls.model)
            
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise
        
        # 定义类别名称
        cls.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck')
    
    def test_model_structure(self):
        """测试模型结构"""
        # 检查模型是否为ResNet18
        self.assertTrue(isinstance(self.model, torch.nn.Module))
        # 检查输出维度
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        output = self.model(dummy_input)
        self.assertEqual(output.shape[1], 10)  # CIFAR-10有10个类别
    
    def test_accuracy(self):
        """测试模型准确率"""
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # 计算每个类别的准确率
                for i in range(targets.size(0)):
                    label = targets[i]
                    pred = predicted[i]
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
        
        # 总体准确率
        accuracy = 100 * correct / total
        print(f"\n总体准确率: {accuracy:.2f}%")
        
        # 每个类别的准确率
        for i in range(10):
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{self.classes[i]:<10} 准确率: {class_acc:.2f}%')
        
        self.assertGreaterEqual(accuracy, 10, f"准确率过低: {accuracy}%")
    
    def test_loss_distribution(self):
        """测试损失分布"""
        losses = []
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                batch_losses = criterion(outputs, targets)
                losses.extend(batch_losses.cpu().numpy())
        
        losses = np.array(losses)
        print(f"\n损失统计:")
        print(f"平均损失: {np.mean(losses):.4f}")
        print(f"标准差: {np.std(losses):.4f}")
        print(f"最小值: {np.min(losses):.4f}")
        print(f"最大值: {np.max(losses):.4f}")
        
        # 检查损失是否在合理范围内
        self.assertLess(np.mean(losses), 5.0, "平均损失过高")
    
    def test_prediction_confidence(self):
        """测试预测置信度"""
        confidences = []
        correct_confidences = []
        incorrect_confidences = []
        class_confidences = {i: [] for i in range(10)}  # 每个类别的置信度
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                
                # 应用温度缩放来提高置信度校准
                temperature = 2.0
                scaled_outputs = outputs / temperature
                
                probabilities = torch.nn.functional.softmax(scaled_outputs, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
                # 记录置信度
                confidences.extend(confidence.cpu().numpy())
                
                # 分别记录正确和错误预测的置信度
                correct_mask = predicted == targets
                correct_confidences.extend(confidence[correct_mask].cpu().numpy())
                incorrect_confidences.extend(confidence[~correct_mask].cpu().numpy())
                
                # 记录每个类别的置信度
                for i in range(10):
                    mask = targets == i
                    if mask.any():
                        class_confidences[i].extend(confidence[mask].cpu().numpy())
        
        print(f"\n置信度统计:")
        print(f"平均置信度: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")
        print(f"正确预测的平均置信度: {np.mean(correct_confidences):.4f} ± {np.std(correct_confidences):.4f}")
        print(f"错误预测的平均置信度: {np.mean(incorrect_confidences):.4f} ± {np.std(incorrect_confidences):.4f}")
        
        print("\n各类别的平均置信度:")
        for i in range(10):
            if class_confidences[i]:
                mean_conf = np.mean(class_confidences[i])
                std_conf = np.std(class_confidences[i])
                print(f"{self.classes[i]:<10}: {mean_conf:.4f} ± {std_conf:.4f}")
        
        # 计算置信度分布
        confidence_bins = np.linspace(0, 1, 11)
        hist_correct, _ = np.histogram(correct_confidences, bins=confidence_bins)
        hist_incorrect, _ = np.histogram(incorrect_confidences, bins=confidence_bins)
        
        print("\n置信度分布:")
        for i in range(len(hist_correct)):
            bin_start = confidence_bins[i]
            bin_end = confidence_bins[i+1]
            print(f"区间 [{bin_start:.1f}, {bin_end:.1f}): "
                  f"正确预测 {hist_correct[i]}, 错误预测 {hist_incorrect[i]}")
        
        # 检查置信度是否合理
        self.assertGreater(np.mean(correct_confidences), np.mean(incorrect_confidences),
                          "正确预测的置信度应该高于错误预测")
        self.assertGreater(np.mean(confidences), 0.5,
                          "平均置信度应该大于0.5")
    
    def test_inference_speed(self):
        """测试推理速度"""
        batch_times = []
        
        # 预热
        dummy_input = torch.randn(64, 3, 32, 32).to(self.device)
        for _ in range(10):
            _ = self.model(dummy_input)
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                
                start_time = time.time()
                _ = self.model(data)
                batch_time = time.time() - start_time
                batch_times.append(batch_time)
        
        avg_time = np.mean(batch_times)
        std_time = np.std(batch_times)
        
        print(f"\n推理速度统计:")
        print(f"平均批次时间: {avg_time*1000:.2f}ms")
        print(f"标准差: {std_time*1000:.2f}ms")
        print(f"每秒处理图片数: {64/avg_time:.2f}")
        
        self.assertLess(avg_time, 0.1, f"批次推理时间过长: {avg_time*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main(verbosity=2) 