# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2022/4/23 14:22
# ------------------------------------------------
import random
import os
import os.path as osp
from copy import deepcopy
from functools import partial
import time

import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from skimage.io import imsave

# 定义全局变量
# 可在此处调整实验所用超参数
# 随机种子
SEED = 1919810
# 保存最佳模型的路径
BEST_CKP_PATH = r'model/change_detection/model.pdparams'

# 训练的epoch数
NUM_EPOCHS = 300
# 每多少个epoch保存一次模型权重参数
SAVE_INTERVAL_EPOCHS = 10
# 初始学习率
LR = 0.001
# 学习率衰减步长（注意，单位为迭代次数而非epoch数），即每多少次迭代将学习率衰减一半
DECAY_STEP = 1000
# 训练阶段 batch size
TRAIN_BATCH_SIZE = 16
# 推理阶段 batch size
INFER_BATCH_SIZE = 4
# 加载数据所使用的进程数
NUM_WORKERS = 4
# 裁块大小
CROP_SIZE = 256
# 模型推理阶段使用的滑窗步长
STRIDE = 64
# 影像原始大小
ORIGINAL_SIZE = (1024, 1024)
# 固定随机种子，尽可能使实验结果可复现

# 随机数生成器种子
RNG_SEED = 114514
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.95

random.seed(RNG_SEED)

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

model = pdrs.tasks.BIT()


# print(model.net)


# 定义一些辅助函数

def info(msg, **kwargs):
    print(msg, **kwargs)


def warn(msg, **kwargs):
    print('\033[0;31m' + msg, **kwargs)


def quantize(arr):
    return (arr * 255).astype('uint8')


# 定义推理阶段使用的数据集
class InferDataset(paddle.io.Dataset):
    """
    变化检测推理数据集。
    Args:
        pic1 输入图片1
        pic2 输入图片2
        transforms (paddlers.transforms.Compose): 需要执行的数据变换操作。
    """

    def __init__(
            self,
            pic1path,
            pic2path,
            transforms
    ):
        super().__init__()

        self.transforms = deepcopy(transforms)
        self.pic1Path = pic1path
        self.pic2Path = pic2path

        pdrs.transforms.arrange_transforms(
            model_type='changedetector',
            transforms=self.transforms,
            mode='test'
        )

        self.samples = [{'image_t1': pic1path, 'image_t2': pic2path}]
        self.names = [osp.basename(pic1path)]

    def __getitem__(self, idx):
        sample = deepcopy(self.samples[idx])
        output = self.transforms(sample)
        return paddle.to_tensor(output[0]), paddle.to_tensor(output[1])

    def __len__(self):
        return len(self.samples)


# 考虑到原始影像尺寸较大，以下类和函数与影像裁块-拼接有关。
class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        if self.h < self.ch or self.w < self.cw:
            raise NotImplementedError
        self.si = si
        self.sj = sj
        self._i, self._j = 0, 0

    def __next__(self):
        # 列优先移动（C-order）
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i + self.ch, self.h)
        right = min(self._j + self.cw, self.w)
        top = max(0, bottom - self.ch)
        left = max(0, right - self.cw)

        if self._j >= self.w - self.cw:
            if self._i >= self.h - self.ch:
                # 设置一个非法值，使得迭代可以early stop
                self._i = self.h + 1
            self._goto_next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._goto_next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def __iter__(self):
        return self

    def _goto_next_row(self):
        self._i += self.si
        self._j = 0


def crop_patches(dataLoader, ori_size, window_size, stride):
    """
    将`dataLoader`中的数据裁块。
    Args:
        dataLoader (paddle.io.DataLoader): 可迭代对象，能够产生原始样本（每个样本中包含任意数量影像）。
        ori_size (tuple): 原始影像的长和宽，表示为二元组形式(h,w)。
        window_size (int): 裁块大小。
        stride (int): 裁块使用的滑窗每次在水平或垂直方向上移动的像素数。
    Returns:
        一个生成器，能够产生iter(`dataLoader`)中每一项的裁块结果。一幅图像产生的块在batch维度拼接。例如，当`ori_size`为1024，而
            `window_size`和`stride`均为512时，`crop_patches`返回的每一项的batch_size都将是iter(`dataLoader`)中对应项的4倍。
    """

    for ims in dataLoader:
        ims = list(ims)
        h, w = ori_size
        win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
        all_patches = []
        for rows, cols in win_gen:
            # NOTE: 此处不能使用生成器，否则因为lazy evaluation的缘故会导致结果不是预期的
            patches = [im[..., rows, cols] for im in ims]
            all_patches.append(patches)
        yield tuple(map(partial(paddle.concat, axis=0), zip(*all_patches)))


def recons_prob_map(patches, ori_size, window_size, stride):
    """从裁块结果重建原始尺寸影像，与`crop_patches`相对应"""
    # NOTE: 目前只能处理batch size为1的情况
    h, w = ori_size
    win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
    prob_map = np.zeros((h, w), dtype=np.float)
    cnt = np.zeros((h, w), dtype=np.float)
    # XXX: 需要保证win_gen与patches具有相同长度。此处未做检查
    for (rows, cols), patch in zip(win_gen, patches):
        prob_map[rows, cols] += patch
        cnt[rows, cols] += 1
    prob_map /= cnt
    return prob_map


# 若输出目录不存在，则新建之（递归创建目录）
out_dir = r'static/imgBase'
if not osp.exists(out_dir):
    os.makedirs(out_dir)
# 为模型加载历史最佳权重
state_dict = paddle.load(BEST_CKP_PATH)
# 同样通过net属性访问组网对象
model.net.set_state_dict(state_dict)


def mainFunc_cd(pic1Path, pic2Path):
    # 实例化测试集
    test_dataset = InferDataset(
        pic1Path,
        pic2Path,
        # 注意，测试阶段使用的归一化方式需与训练时相同
        T.Compose([
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    )
    # 创建DataLoader
    test_dataLoader = paddle.io.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        return_list=True
    )
    test_dataLoader = crop_patches(
        test_dataLoader,
        ORIGINAL_SIZE,
        CROP_SIZE,
        STRIDE
    )
    # 推理过程主循环
    info("start")
    model.net.eval()
    with paddle.no_grad():
        for name, (t1, t2) in zip(test_dataset.names, test_dataLoader):
            shape = paddle.shape(t1)
            pred = paddle.zeros(shape=(shape[0], 2, *shape[2:]))
            for i in range(0, shape[0], INFER_BATCH_SIZE):
                pred[i:i + INFER_BATCH_SIZE] = model.net(t1[i:i + INFER_BATCH_SIZE], t2[i:i + INFER_BATCH_SIZE])[0]
            # 取softmax结果的第1（从0开始计数）个通道的输出作为变化概率
            prob = paddle.nn.functional.softmax(pred, axis=1)[:, 1]
            # 由patch重建完整概率图
            prob = recons_prob_map(prob.numpy(), ORIGINAL_SIZE, CROP_SIZE, STRIDE)
            # 默认将阈值设置为0.5，即，将变化概率大于0.5的像素点分为变化类
            out = quantize(prob > 0.5)
            path = str(time.time())+'cd.png'
            imsave(osp.join(out_dir, str(time.time())+'cd.png'), out, check_contrast=False)
    info("end")
    return 'imgBase/'+path


if __name__ == '__main__':
    pass
    # mainFunc_cd(r'static/imgBase/test_1.png',
    #          r'static/imgBase/test_2.png')
