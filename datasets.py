import torch 
from torchvision import datasets, transforms
import os
import json
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import VOCDetection, VOCSegmentation, CIFAR10, CIFAR100
import xml.etree.ElementTree as ET

def get_dataset(dir, name):
	if name == 'voc2007':
		# VOC2007的均值和标准差
		VOC_MEAN = (0.485, 0.456, 0.406)
		VOC_STD = (0.229, 0.224, 0.225)
		
		# 训练集转换
		transform_train = transforms.Compose([
			transforms.Resize((224, 224)),  # VOC图像尺寸不固定，需要调整
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(VOC_MEAN, VOC_STD)
		])
		
		# 测试集转换
		transform_test = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(VOC_MEAN, VOC_STD)
		])
		
		class VOCClassification(Dataset):
			def __init__(self, root, image_set='train', transform=None):
				self.root = root
				self.image_set = image_set
				self.transform = transform
				
				# 设置图像和标注路径
				self.images_dir = os.path.join(root, 'VOC2007', 'JPEGImages')
				self.annotations_dir = os.path.join(root, 'VOC2007', 'Annotations')
				
				# 读取数据集划分文件
				split_file = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', f'{image_set}.txt')
				with open(split_file, 'r') as f:
					self.ids = [x.strip() for x in f.readlines()]
				
				# VOC类别
				self.classes = [
					'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
					'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
					'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
				]
				self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
				
			def __getitem__(self, index):
				img_id = self.ids[index]
				
				# 加载图像
				img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
				img = Image.open(img_path).convert('RGB')
				
				# 加载标注
				anno_path = os.path.join(self.annotations_dir, f'{img_id}.xml')
				tree = ET.parse(anno_path)
				root = tree.getroot()
				
				# 获取第一个对象的类别作为图像类别
				objects = root.findall('object')
				if objects:
					label = self.class_to_idx[objects[0].find('name').text]
				else:
					label = 0
				
				if self.transform:
					img = self.transform(img)
				
				return img, label
			
			def __len__(self):
				return len(self.ids)
		
		# 创建训练集和测试集
		train_dataset = VOCClassification(
			root=dir,
			image_set='train',
			transform=transform_train
		)
		
		eval_dataset = VOCClassification(
			root=dir,
			image_set='val',
			transform=transform_test
		)
		
	elif name == 'cifar-100':
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

class SerializableVOCDataset(torch.utils.data.Dataset):
    """可序列化的VOC数据集类"""
    def __init__(self, root, year='2007', image_set='train', transform=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        
        # VOC数据集路径
        self.voc_root = os.path.join(root, f'VOCdevkit/VOC{year}')
        self.image_dir = os.path.join(self.voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(self.voc_root, 'Annotations')
        
        # 读取数据集索引
        split_f = os.path.join(self.voc_root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_f, 'r') as f:
            self.images = [x.strip() for x in f.readlines()]
            
        # VOC类别名称
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 获取图像ID
        img_id = self.images[index]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        
        # 加载标注
        anno_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        target = self._parse_voc_xml(ET.parse(anno_path).getroot())
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def _parse_voc_xml(self, node):
        """解析VOC XML文件并返回目标类别索引"""
        objects = node.findall('object')
        if not objects:
            return 0  # 默认背景类
        
        # 获取第一个对象的类别（简化处理）
        obj = objects[0]
        class_name = obj.find('name').text.lower().strip()
        return self.class_to_idx.get(class_name, 0)

def get_dataset(data_dir, dataset_type):
    """获取指定类型的数据集"""
    # 基本的数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_type == 'cifar10':
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_type == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_type == 'voc2007':
        train_dataset = SerializableVOCDataset(root=data_dir, year='2007', image_set='train', transform=transform)
        test_dataset = SerializableVOCDataset(root=data_dir, year='2007', image_set='val', transform=transform)
    else:
        raise ValueError(f'不支持的数据集类型: {dataset_type}')

    return train_dataset, test_dataset