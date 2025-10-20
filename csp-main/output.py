import torch

# 加载检查点
checkpoint = torch.load('/home/wjx/paper/csp-main/data/model/c-fashion/sample_model/soft_embeddings_epoch_20.pt')

# 查看内容
print("检查点键名:", checkpoint.keys())

# 如果是模型权重
if 'model_state_dict' in checkpoint:
    print("模型参数键名:", checkpoint['model_state_dict'].keys())

# 直接查看所有内容
for key, value in checkpoint.items():
    print(f"{key}: {type(value)}")