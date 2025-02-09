import os
import logging

# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

USER_ID = "gym"
GENDER = "girl"
INPUTS_DATA_DIR = os.path.join(os.getcwd(), "datasets/inputs/")
OUTPUTS_DATA_DIR = os.path.join(os.getcwd(), "datasets/outputs/")
TEMPLATE_DIR = os.path.join(os.getcwd(), "datasets/template/")
VISIBLE_DIR = os.path.join(os.getcwd(), "datasets/visible/")
LORA_OUTPUT_DIR = os.path.join(os.getcwd(), "lora_weights")
CACHE_MODEL_PATH = os.path.join(os.getcwd(), 'model_data')

logging.info("----------------------------INFO----------------------------")
logging.info(f" Ohhhhh : end of pip install !!!")