import os
import logging

# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

# 下载Lora模型 gym.safetensors （这是一个训练好的人物Lora模型）
os.system('rm -rf ./gym.safetensors && rm -rf ./lora_weights/user_weights/gym/gym.safetensors')
os.system('wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/dsw/gym.safetensors')
os.makedirs("./lora_weights/user_weights/gym", exist_ok=True)
os.system('cp -rf gym.safetensors ./lora_weights/user_weights/gym')

# 下载模板照片 template.tar.gz
os.system('rm -rf ./datasets/template && rm -rf template.tar.gz')
os.system('wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/dsw/template.tar.gz && tar -xf template.tar.gz')
os.makedirs('./datasets/', exist_ok=True)
os.system('mv template ./datasets/template')

# 下载Lora训练数据示例 gym.tar.gz
os.system('rm -rf ./datasets/inputs/gym && rm -rf gym.tar.gz')
os.system('wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/dsw/gym.tar.gz && tar -xf gym.tar.gz')
os.makedirs('./datasets/inputs', exist_ok=True)
os.system('mv gym ./datasets/inputs')

logging.info("----------------------------INFO----------------------------")
logging.info(f" Ohhhhh : all demo data ready !!!")
