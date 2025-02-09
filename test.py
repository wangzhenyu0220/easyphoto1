import torch
print(torch.__version__)  # 1.11.0
print(torch.version.cuda)  # 应显示11.3（对应PyTorch 1.11的官方编译版本）
print(torch.cuda.is_available())  # 必须返回True