import sys
import logging
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 设置ModelScope管道
# pipeline 函数用于初始化不同的任务管道
# Tasks.face_detection 等常量用于指定任务类型，如人脸检测、质量评估、皮肤修饰、图像融合和人脸识别
# 每个 pipeline 调用中，第二个参数指定了要使用的具体模型。
pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')
pipeline(Tasks.skin_retouching, model='damo/cv_unet_skin-retouching')
pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')

# 导入扩展模块
sys.path.append("/root/photog_dsw/extention/CodeFormer")

# 设置日志记录器
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logging.info('----------------------all load finished-----------------')
