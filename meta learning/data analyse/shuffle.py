import os
import glob
import random
import shutil


def mkdirs_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 划分数据集
def split_data(rate=0.25):
    src = './data final/train'

    # 1.检查并创建文件夹
    val_src = './data final/test'
    mkdirs_if_missing(val_src)

    # 2.抽样
    img_pths = glob.glob(src + os.sep + '*.jpg')
    random.shuffle(img_pths)  # 打乱顺序
    val_count = int(len(img_pths) * rate)
    val_pths = random.sample(img_pths, val_count)  # 抽样
    # 3.移动抽样的图片到验证集
    for pth in val_pths:
        name = pth.split(os.sep)[-1]
        shutil.move(pth, os.path.join(val_src, name))
    print(f'test amount：{val_count}')


split_data()
