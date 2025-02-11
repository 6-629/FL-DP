import torch
import torch.nn as nn
import torchvision.models as models

def get_model(name="resnet18", num_classes=10):
	"""获取模型实例"""
	if name == "resnet18":
		# 使用weights=None确保不加载预训练权重
		model = models.resnet18(weights=None)
		# 修改最后的全连接层以匹配类别数
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)
		return model
	elif name == "resnet50":
		model = models.resnet50(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	elif name == "densenet121":
		model = models.densenet121(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.classifier = nn.Linear(model.classifier.in_features, num_classes)
		return model
	elif name == "alexnet":
		model = models.alexnet(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
		return model
	elif name == "vgg16":
		model = models.vgg16(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
		return model
	elif name == "vgg19":
		model = models.vgg19(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
		return model
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	elif name == "googlenet":		
		model = models.googlenet(pretrained=False)
		# 修改最后的全连接层以匹配类别数
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	else:
		raise ValueError(f"不支持的模型: {name}")

# 直接导出resnet18函数
def resnet18(num_classes=10):
	return get_model("resnet18", num_classes)

if __name__ == "__main__":
	model = get_model("resnet18", 10)
	print(model) 