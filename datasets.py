import torch 
from torchvision import datasets, transforms
import os
import json
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_dataset(dir, name):
	if name == 'cifar-100':
		# CIFAR100的均值和标准差
		CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
		CIFAR100_STD = (0.2675, 0.2565, 0.2761)
		
		# 训练集转换
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
		])
		
		# 测试集转换
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
		])
		
		train_dataset = datasets.CIFAR100(
			root=dir, 
			train=True, 
			download=True,
			transform=transform_train
		)
		
		eval_dataset = datasets.CIFAR100(
			root=dir, 
			train=False, 
			download=True,
			transform=transform_test
		)
		
	elif name == 'cifar-10':
		# CIFAR10的均值和标准差
		CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
		CIFAR10_STD = (0.2023, 0.1994, 0.2010)
		
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
		])
		
		train_dataset = datasets.CIFAR10(
			root=dir, 
			train=True, 
			download=True,
			transform=transform_train
		)
		eval_dataset = datasets.CIFAR10(
			root=dir, 
			train=False, 
			transform=transform_test
		)
	
	return train_dataset, eval_dataset

class COCO8Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        COCO8数据集加载器
        :param root: 数据集根目录
        :param split: 'train' 或 'val'
        :param transform: 图像转换
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # 使用正确的图像目录路径
        self.img_dir = os.path.join(root, 'images', split)
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"找不到图像目录: {self.img_dir}")
            
        # 标签目录路径
        self.label_dir = os.path.join(root, 'labels', split)
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"找不到标签目录: {self.label_dir}")
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(self.img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.image_files) == 0:
            raise ValueError(f"未找到图像文件在: {self.img_dir}")
            
        print(f"{split}集图片数量: {len(self.image_files)}")
    
    def __getitem__(self, index):
        """获取单个数据样本"""
        try:
            # 获取图像文件名
            img_file = self.image_files[index]
            img_path = os.path.join(self.img_dir, img_file)
            
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            
            # 获取对应的标签文件
            label_file = os.path.join(self.label_dir, 
                                    os.path.splitext(img_file)[0] + '.txt')
            
            # 读取标签（YOLO格式：class x y w h）
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_line = f.readline().strip()
                    if label_line:
                        label = int(label_line.split()[0])  # 只取类别ID
                    else:
                        label = 0
            else:
                label = 0
            
            # 应用转换
            if self.transform is not None:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.image_files)