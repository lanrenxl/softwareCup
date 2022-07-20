# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2022/5/25 16:59
# ------------------------------------------------
# 解压数据集
# 划分训练集/验证集/测试集，并生成文件名列表
# 所有样本从RSOD数据集的playground子集中选取

import random
import os.path as osp
import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from paddlers.tasks.utils.visualize import visualize_detection
import time

# 随机数生成器种子
RNG_SEED = 52980
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 调节此参数控制验证集数据的占比
VAL_RATIO = 0.05
# 数据集路径
DATA_DIR = 'tempPic/td'

# 目标类别
CLASS = 'playground'

# 定义全局变量

# 随机种子
SEED = 52980
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = 'tempPic/td/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = 'tempPic/td/label.txt'
# 目标类别
CLASS = 'playground'
# 模型验证阶段输入影像尺寸
INPUT_SIZE = 608

# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

# 构建数据集
eval_transforms = T.Compose([
    # 使用双三次插值将输入影像缩放到固定大小
    T.Resize(
        target_size=INPUT_SIZE, interp='CUBIC'
    ),
    # 验证阶段与训练阶段的归一化方式必须相同
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])
# 构建PP-YOLO模型
model = pdrs.tasks.PPYOLO()
model.net_initialize(
    pretrain_weights='COCO',
    resume_checkpoint=None,
    is_backbone_weights=False
)


# 为模型加载历史最佳权重
# state_dict = paddle.load("model/Target_detection/model.pdparams")
# model.net.set_state_dict(state_dict)


# 预测结果可视化
# 重复运行本单元可以查看不同结果

def read_rgb(path):
    im = cv2.imread(path)
    im = im[..., ::-1]
    return im


# 绘制目标框
# with paddle.no_grad():
#     model.labels = test_dataset.labels
#     for idx, im in zip(chosen_indices, ims):
#
#         im = cv2.resize(im[..., ::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
#         pred = model.predict(im, eval_transforms)
#
#         vis = im
#         # 用绿色画出预测目标框
#         if len(pred) > 0:
#             vis = visualize_detection(
#                 np.array(vis), pred,
#                 color=np.asarray([[0, 255, 0]], dtype=np.uint8),
#                 threshold=0.2, save_dir=None
#             )

out_dir = r'static/imgBase'


def mainFunc_td(pic_path):
    model.labels = ['playground']
    img = cv2.imread(pic_path)
    img = cv2.resize(img[..., ::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    # model.
    predict = model.predict(img, eval_transforms)

    visualize = img
    # 用绿色画出预测目标框
    if len(predict) > 0:
        visualize = visualize_detection(
            np.array(visualize), predict,
            color=np.asarray([[0, 255, 0]], dtype=np.uint8),
            threshold=0.2, save_dir=None
        )

    path = str(time.time()) + 'td.png'
    cv2.imwrite(osp.join(out_dir, str(time.time()) + 'td.png'), visualize)

    return 'imgBase/' + path
    pass


if __name__ == '__main__':
    mainFunc_td(r"..\tempPic\td\playground\JPEGImages\playground_165.jpg")
