FederatedLearningPrivacy/
├── data/                          # 数据集存放目录
│   ├── dataset1/
│   ├── dataset2/
│   ├── preprocess.py              # 数据预处理脚本
│   ├── loaders.py                 # 数据加载脚本
├── models/                        # 模型相关代码
│   ├── base_model.py              # 模型定义基类
│   ├── resnet.py                  # ResNet 模型
│   ├── yolo.py                    # YOLO 模型
│   ├── __init__.py                # 模块初始化
├── clients/                       # 客户端相关代码
│   ├── client.py                  # 客户端逻辑实现
│   ├── privacy_schemes.py         # 差分隐私保护方案实现
├── server/                        # 服务器端相关代码
│   ├── server.py                  # 服务器端逻辑实现
│   ├── aggregation.py             # 模型聚合实现
├── utils/                         # 工具函数模块
│   ├── metrics.py                 # 评价指标
│   ├── visualization.py           # 数据和实验结果可视化
│   ├── logger.py                  # 日志记录工具
├── experiments/                   # 实验测试相关
│   ├── experiment1.py             # 实验 1：不同隐私保护程度测试
│   ├── experiment2.py             # 实验 2：客户端数量测试
│   ├── experiment3.py             # 实验 3：数据恢复实验
├── configs/                       # 配置文件目录
│   ├── default_config.json        # 默认配置
│   ├── dataset1_config.json       # 数据集1配置
│   ├── dataset2_config.json       # 数据集2配置
├── requirements.txt               # 项目依赖列表
├── README.md                      # 项目说明文档
└── main.py                        # 程序入口
项目结构 仅供参考
差分隐私三种方案
![alt text](image.png)