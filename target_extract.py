# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2022/6/2 19:29
# ------------------------------------------------
# 导入需要用到的库

import random
import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
import os.path as osp
import time

# 随机数生成器种子
RNG_SEED = 56961
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9

# 分割类别
CLASSES = (
    'background',
    'road',
)
# 定义全局变量
SEED = 56961

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

eval_transforms = T.Compose([
    T.Resize(target_size=300),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 构建DeepLab V3+模型，使用ResNet-50作为backbone
model = pdrs.tasks.DeepLabV3P(
    input_channel=3,
    num_classes=2,
    backbone='ResNet50_vd'
)
model.net_initialize(
    # pretrain_weights='CITYSCAPES',
    resume_checkpoint=None,
    is_backbone_weights=False
)
state_dict = paddle.load('model/target_extract/model.pdparams')
model.net.set_state_dict(state_dict)


out_dir = r'static/imgBase'


def mainFunc_te(pic_path):
    model.net.eval()
    img = cv2.imread(pic_path)
    temp = cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_CUBIC)
    output = model.predict(temp, eval_transforms)
    end = output['score_map'][:, :, 1] * 255
    path = str(time.time()) + 'te.png'
    cv2.imwrite(osp.join(out_dir, path), end)
    return 'imgBase/' + path
    pass


if __name__ == '__main__':
    mainFunc_te(r"tempPic/te/input/img-6.png")
