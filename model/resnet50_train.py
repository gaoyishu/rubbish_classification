import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image


DATA_DIR = 'C:\\Users\\Administrator\\Desktop\\garbage\\data2'    # 图片数据根目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')    # 训练数据文件夹目录
VALID_DIR = os.path.join(DATA_DIR, 'val')    # 验证数据文件夹目录
SIZE = (224, 224)    # 图片大小
BATCH_SIZE = 16    # 训练数据的batch size


if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])    # 读取训练数据，看一下总共有多少训练样本
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])    # 读取验证数据，看一下总共有多少验证样本

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)   # 计算学习完一次所有的训练样本需要的步数
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)   # 计算验证一次所有的验证数据集需要的步数

    gen = keras.preprocessing.image.ImageDataGenerator()    # 创建训练集图片读取器
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)    # 创建验证集图片读取器

    # 按batch_size为单位，获取所有的训练数据
    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    # 按batch_size为单位，获取所有的验证数据
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    # 加载resnet50模型
    model = keras.applications.resnet50.ResNet50()
    # 获取图片数据的类别标签和对应的编号（即train文件夹下的类别文件夹名称）
    classes = list(iter(batches.class_indices))
    # 删除resnet50的最后一层，因为预训练好的resent50最后一层是为1000个分类类别设计的
    model.layers.pop()
    # 固定resent50的每层参数，不让其参与训练，这样针对自己的图片分类问题可以更快完成训练
    for layer in model.layers:
        layer.trainable=False
    # 获取resent50的最后一层输出
    last = model.layers[-1].output
    # 在resent50的最后一层输出后接一个神经元个数为类别数量的全连接
    x = Dense(len(classes), activation="softmax")(last)
    # 初始化修正后的模型
    finetuned_model = Model(model.input, x)
    # 设置训练使用的学习方法，损失函数以及评价指标
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 获取类别编号到类别名称的映射
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    # 设置训练的early_stoping，即当验证集的准确率连续两个epoch都不再提高时，自动终止训练
    early_stopping = EarlyStopping(patience=2)
    # 设置保存训练好的模型的名字
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    # 开始训练模型
    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=1000, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    # 保存训练好的模型
    finetuned_model.save('resnet50_final.h5')
