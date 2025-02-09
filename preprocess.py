import argparse
import json
import os
import sys
sys.path.append("~/photog_dsw/extension/CodeFormer")
import codeformer_helper

import numpy as np
from tqdm import tqdm
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from utils.sod_predictor import SODPredictor
from utils.face_process_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        required=True,
        help="The user_id of the training data.",
    )
    parser.add_argument(
        "--sex",
        type=str,
        default=None,
        help=(
            "The sex of the user."
        ),
    )
    parser.add_argument(
        "--inputs_dir",
        type=str,
        default=None,
        help=(
            "The inputs dir of the data for preprocessing."
        ),
    )
    parser.add_argument(
        "--middle_dir",
        type=str,
        default=None,
        help=(
            "The middle_dir after the preprocess."
        ),
    )
    parser.add_argument(
        "--visible_dir",
        type=str,
        default=None,
        help=(
            "The visilble_dir in the preprocess."
        ),
    )
    parser.add_argument(
        "--crop_ratio",
        type=int,
        default=3,
        help=(
            "The expand ratio for input data to crop."
        ),
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default='../model_data/',
        help=(
            "The expand ratio for input data to crop."
        ),
    )
    args = parser.parse_args()
    return args

def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # 然后对真人图片取mean，获取真人图片的平均特征
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # 计算一个文件夹中，和中位值最接近的图片排序
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores

if __name__ == "__main__":
    args        = parse_args()

    user_id     = args.user_id      # "zhoumo"
    sex         = args.sex          # "girl"
    inputs_dir  = args.inputs_dir   # "datasets/inputs"
    middle_dir  = args.middle_dir   # "mnt"
    visible_dir = args.visible_dir

    validation_prompt = f"{user_id}_face, {user_id}, 1{sex}"
    print(validation_prompt)

    # 人脸评分
    face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')
    # embedding
    face_recognition       = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
    # 人脸检测
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    # 显著性检测
    salient_detect          = SODPredictor(model_name='u2netp', model_path=os.path.join(args.model_cache_dir, 'u2netp.pth'))
    # 
    skin_retouching = pipeline(Tasks.skin_retouching,model='damo/cv_unet_skin-retouching')
    # 
    codeFormer_net, bg_upsampler, face_helper = codeformer_helper.get_nets()
    
    # 创建输出文件夹
    cache_save_path     = os.path.join(middle_dir, "user_images", user_id)
    images_save_path    = os.path.join(middle_dir, "user_images", user_id, "train")
    json_save_path      = os.path.join(middle_dir, "user_images", user_id, "metadata.jsonl")
    os.system(f"rm -rf {cache_save_path}")
    os.makedirs(middle_dir, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    
    if visible_dir is not None:
        os.system(f"rm -rf {visible_dir}")
        os.makedirs(visible_dir, exist_ok=True)

    # 获得jpg列表，并且裁剪+显著性检测
    jpgs    = os.listdir(os.path.join(args.inputs_dir, user_id))
    
    copy_jpgs = []
    embeddings = []
    scores = []
    selected_paths = []
    for index, jpg in tqdm(enumerate(jpgs)):
        if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            continue

        try:
            _image_path = os.path.join(args.inputs_dir, user_id, jpg)
            image       = Image.open(_image_path)
            h, w, c     = np.shape(image)
            retinaface_box, _, _ = call_face_crop(retinaface_detection, image, args.crop_ratio, prefix="tmp")
        
            face_width = (retinaface_box[2] - retinaface_box[0]) / (args.crop_ratio - 1)
            face_height = (retinaface_box[3] - retinaface_box[1]) / (args.crop_ratio - 1)
            if face_width / w < 1/8 or face_height / h < 1/8:
                continue
        
            sub_image   = image.crop(retinaface_box)
            sub_image = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
        
            embedding = np.array(face_recognition(sub_image)[OutputKeys.IMG_EMBEDDING])
            score = face_quality_func(sub_image)[OutputKeys.SCORES]
            score = 0 if score is None else score[0]
        except:
            print(f'faild to preprocess  {jpg}')
            continue
        
        copy_jpgs.append(jpg)
        embeddings.append(embedding)
        scores.append(score)
        selected_paths.append(_image_path)
    
    embeddings = compare_jpg_with_face_id(embeddings)
    print("scores :", scores)
    print("embeddings :", embeddings)
    
    total_scores = np.array(embeddings) # np.array(embeddings) * np.array(scores)
    indexes = np.argsort(total_scores)[::-1][:15]
    
    selected_jpgs = []
    selected_scores = []
    for index in indexes:
        selected_jpgs.append(copy_jpgs[index])
        selected_scores.append(scores[index])
        print("jpg:", copy_jpgs[index], "embeddings", embeddings[index])
                             
    images  = []
    max_codeformer_num = len(selected_jpgs) // 2
    codeformer_num = 0
    for index, jpg in tqdm(enumerate(selected_jpgs[::-1])):
        if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            continue
        _image_path = os.path.join(args.inputs_dir, user_id, jpg)
        image       = Image.open(_image_path)
        retinaface_box, _, _ = call_face_crop(retinaface_detection, image, args.crop_ratio, prefix="tmp")
        sub_image   = image.crop(retinaface_box)
        sub_image   = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
        if (selected_scores[index] < 0.60 or np.shape(sub_image)[0] < 512 or np.shape(sub_image)[1] < 512) and codeformer_num < max_codeformer_num:
            sub_image = Image.fromarray(codeformer_helper.infer(codeFormer_net, face_helper, bg_upsampler, np.array(sub_image)))
            codeformer_num += 1
        
        sub_box, _, sub_mask = call_face_crop(retinaface_detection, sub_image, 1, prefix="tmp")
        h, w, c      = np.shape(sub_mask)
        face_width   = sub_box[2] - sub_box[0]
        face_height   = sub_box[3] - sub_box[1]
        sub_box[0]   = np.clip(np.array(sub_box[0], np.int32) - face_width * 0.3, 1, w - 1)
        sub_box[2]   = np.clip(np.array(sub_box[2], np.int32) + face_width * 0.3, 1, w - 1)
        sub_box[1]   = np.clip(np.array(sub_box[1], np.int32) + face_height * 0.15, 1, h - 1)
        sub_box[3]   = np.clip(np.array(sub_box[3], np.int32) + face_height * 0.15, 1, h - 1)

        sub_mask = np.zeros_like(np.array(sub_mask, np.uint8))
        sub_mask[sub_box[1]:sub_box[3], sub_box[0]:sub_box[2]] = 1
        result      = salient_detect.predict([sub_image])[0]['mask']
        # mask        = cv2.resize(result, [np.shape(sub_image)[1], np.shape(sub_image)[0]])
        mask        = np.float32(np.expand_dims(result > 128, -1)) * sub_mask
        mask_sub_image = np.array(sub_image) * np.array(mask) + np.ones_like(sub_image) * 255 * (1 - np.array(mask))
        mask_sub_image = Image.fromarray(np.uint8(mask_sub_image))
        if visible_dir is not None:
            image.save(os.path.join(visible_dir, str(index) + ".jpg"))
            sub_image.save(os.path.join(visible_dir, str(index) + "_crop.jpg"))
            mask_sub_image.save(os.path.join(visible_dir, str(index) + "_crop_mask.jpg"))
        images.append(mask_sub_image)

    # 写入结果
    for index, base64_pilimage in enumerate(images):
        image = base64_pilimage.convert("RGB")
        image.save(os.path.join(images_save_path, str(index) + ".jpg"))
        print("save processed image to "+ os.path.join(images_save_path, str(index) + ".jpg"))
        with open(os.path.join(images_save_path, str(index) + ".txt"), "w") as f:
            f.write(validation_prompt)

    with open(json_save_path, 'w', encoding="utf-8") as f:
        for root, dirs, files in os.walk(images_save_path, topdown=False):
            for file in files:
                path = os.path.join(root, file)
                if not file.endswith('txt'):
                    txt_path = ".".join(path.split(".")[:-1]) + ".txt"
                    if os.path.exists(txt_path):
                        prompt          = open(txt_path, 'r').readline().strip()
                        jpg_path_split  = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {
                            "file_name": file_name, 
                            "text": prompt
                        }
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")