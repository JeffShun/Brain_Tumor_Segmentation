## 一、基于级联U-Net的脑肿瘤分割模型
**论文链接（MICCAI2019 workshop）：[CU-Net: Cascaded U-Net with Loss Weighted Sampling for Brain Tumor Segmentation](https://springer.dosf.top/chapter/10.1007/978-3-030-33226-6_12)** 


## 二、主要文件说明

- train.py: 单卡训练代码入口
- train_multi_gpu.py: 分布式训练代码入口
- ./custom/dataset/dataset.py: dataset类
- ./custom/model/model_network.py: 模型head文件
- ./custom/model/model_head.py: 整个网络部分，训练阶段构建模型，forward方法输出loss的dict
- ./custom/utils/generate_dataset.py: 从原始数据生成输入到模型的数据，供custom/dataset/dataset.py使用
- ./custom/utils/save_torchscript.py: 生成模型对应的静态图
- ./custom/utils/common_tools.py: 工具函数包
- ./custom/utils/distributed_utils.py: 分布式训练函数包
- ./config/seg_braintumor_config.py: 训练的配置文件
