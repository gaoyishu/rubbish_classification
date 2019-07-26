import IPython
import json, os, re, sys, time
import numpy as np
import keras
from page import app
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


FILE_PAHT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(FILE_PAHT,'resnet50_best.h5')
DATA_DIR = os.path.join(FILE_PAHT,'data2')    # 图片数据根目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')    # 训练数据文件夹目录
VALID_DIR = os.path.join(DATA_DIR, 'val')    # 验证数据文件夹目录
SIZE = (224, 224)    # 图片大小
BATCH_SIZE = 16    # 训练数据的batch size

# 以下是定义的一个预测函数，函数的参数时图片路径和训练好的模型路径
def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))    # 加载图片，resize图片大小到224*224
    x = image.img_to_array(img)    # 将图片数据转成数组
    x = np.expand_dims(x, axis=0)    # 为图片数组增加一个维度
    preds = model.predict(x)    # 预测图片
    return preds    # 返回预测的结果

def predict_API(file_location):
    # 0: 湿垃圾
	# 1：可回收
	# 2：干垃圾
	# 3：有害垃圾
 #    cate_dict = {"Epipremnum_Aureum": 0, "bag": 1, "cylindrical_battery": 3, "eggshells": 0,
	# "eraser": 2, "gel_pen": 2, "keyboard": 1, "led_tube": 3}
    cate_dict = {"Epipremnum_Aureum": "湿垃圾", "bag": "可回收", "cylindrical_battery": "有害垃圾", "eggshells": "湿垃圾", "eraser": "干垃圾", "gel_pen": "干垃圾", "keyboard": "可回收", "led_tube": "有害垃圾"}
    model_path = MODEL_PATH    # 训练好的模型路径
    print('Loading model:', model_path)
    t0 = time.time()    # 设置计时器，记录开始时间
    model = load_model(model_path)    # 加载训练好的模型
    t1 = time.time()    # 记录结束时间
    print('Loaded in:', t1-t0)   # 打印出加载模型花费的时间
    gen = keras.preprocessing.image.ImageD ataGenerator()   # 创建训练集图片读取器
    # 按batch_size为单位，获取所有的训练数据
    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    # 获取数据集的标签名称即类别名称
    labels = list(batches.class_indices.keys())
    # 获取待预测的图片路径
    file_url = file_location
    print('Generating predictions on image:', file_url)
    # 调用上面定义的predict函数，获取预测值
    preds = predict(file_url, model)
    # 输出该图片属于各个类别的预测分数，分数最高的类别即该图片所属类别
    for label, score in zip(labels, preds[0]):
        print("{}: {}".format(label, score))
	
    max_id  = np.argmax(preds[0])
    label_str = labels[max_id]
    print("{} is {}".format(label_str, cate_dict[label_str]))

    return preds



