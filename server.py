import models, torch
from tqdm import tqdm

class Server(object):

    def __init__(self, conf, eval_dataset):

        self.conf = conf

        # 使用配置中的类别数初始化模型
        num_classes = conf.get("num_classes", 10)  # 默认为10类
        self.global_model = models.get_model(self.conf["model_name"], num_classes=num_classes)

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = self.global_model.to(self.device)

        if eval_dataset is not None:
            self.eval_loader = torch.utils.data.DataLoader(
                eval_dataset, 
                batch_size=self.conf["batch_size"], 
                shuffle=True,
                pin_memory=True if torch.cuda.is_available() else False
            )

    def model_aggregate(self, weight_accumulator):
        """
        聚合模型参数
        """
        try:
            # 检查权重是否包含NaN或Inf
            for name, update in weight_accumulator.items():
                if torch.isnan(update).any() or torch.isinf(update).any():
                    raise ValueError(f"权重{name}包含NaN或Inf值")

            # 使用float32进行聚合计算
            with torch.cuda.amp.autocast(enabled=False):
                for name, data in self.global_model.state_dict().items():
                    if name in weight_accumulator:
                        update = weight_accumulator[name]
                        # 确保数据类型一致
                        if data.dtype != update.dtype:
                            update = update.to(dtype=data.dtype)
                        # 确保设备一致
                        if data.device != update.device:
                            update = update.to(device=data.device)
                        # 检查更新值的范围
                        if torch.isnan(update).any() or torch.isinf(update).any():
                            continue  # 跳过包含NaN或Inf的更新
                        # 应用更新
                        data.copy_(update)
                        
        except Exception as e:
            print(f"聚合时出错: {str(e)}")
            raise

    def model_eval(self):
        """评估全局模型"""
        try:
            self.global_model.eval()
            total_loss = 0.0
            correct = 0
            dataset_size = 0
            batch_count = 0
            
            with torch.no_grad():
                for batch_id, batch in enumerate(self.eval_loader):
                    data, target = batch
                    dataset_size += data.size()[0]

                    # 确保数据在正确的设备上
                    data = data.to(self.device)
                    target = target.to(self.device)

                    try:
                        # 使用float32进行前向传播
                        with torch.cuda.amp.autocast(enabled=False):
                            output = self.global_model(data)
                            
                            # 检查输出是否有效
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                print(f"警告：批次 {batch_id} 的模型输出包含NaN或Inf")
                                continue
                                
                            # 计算损失时添加数值稳定性处理
                            output = torch.clamp(output, min=-1e6, max=1e6)  # 限制输出范围
                            log_softmax = torch.log_softmax(output, dim=1)
                            loss = torch.nn.functional.nll_loss(log_softmax, target, reduction='sum')
                            
                            # 检查损失值
                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f"警告：批次 {batch_id} 的损失计算异常")
                                continue
                                
                            total_loss += loss.item()
                            batch_count += 1

                        pred = output.max(1)[1]
                        correct += pred.eq(target).cpu().sum().item()

                        # 定期打印评估进度和中间结果
                        if batch_id % 20 == 0:
                            avg_loss_so_far = total_loss / dataset_size if dataset_size > 0 else float('inf')
                            acc_so_far = 100.0 * correct / dataset_size if dataset_size > 0 else 0.0
                            print(f"评估进度: {batch_id}/{len(self.eval_loader)}, "
                                  f"当前损失: {avg_loss_so_far:.4f}, "
                                  f"当前准确率: {acc_so_far:.2f}%")

                    except RuntimeError as e:
                        print(f"警告：处理批次 {batch_id} 时出错: {str(e)}")
                        continue

            # 只在有效批次上计算平均值
            if batch_count == 0:
                raise ValueError("没有有效的评估批次")
                
            avg_loss = total_loss / dataset_size if dataset_size > 0 else float('inf')
            accuracy = 100.0 * correct / dataset_size if dataset_size > 0 else 0.0

            # 添加结果验证
            if avg_loss > 1e3 or accuracy < 0 or accuracy > 100:
                raise ValueError(f"评估结果异常: 损失={avg_loss:.4f}, 准确率={accuracy:.2f}%")

            return accuracy, avg_loss

        except Exception as e:
            print(f"评估时出错: {str(e)}")
            raise
