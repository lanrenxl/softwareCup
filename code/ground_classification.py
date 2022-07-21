# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2022/5/25 14:46
# ------------------------------------------------
# 导入需要用到的库

import random
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
import cv2
import os.path as osp
import copy
from paddle.io import Dataset
from paddlers.utils import logging, get_num_workers, get_encoding, path_normalization, is_pic
import time

# 随机种子
SEED = 77571
# 数据集存放目录
DATA_DIR = '../../tempPic/gc'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '../../tempPic/gc/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '../../tempPic/gc/labels.txt'
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)
# 构建数据集
eval_transforms = T.Compose([
    T.Resize(target_size=256),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 构建DeepLab V3+模型，使用ResNet-50作为backbone
model = pdrs.tasks.DeepLabV3P()
# print(model)
# print(paddle.summary(model))
# 为模型加载历史最佳权重
state_dict = paddle.load("model/Ground_classification/model.pdparams")
model.net.set_state_dict(state_dict)


class SegDataset(Dataset):
    """读取语义分割任务数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径。默认值为None。
        transforms (paddlers.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list=None,
                 transforms=None,
                 num_workers='auto',
                 shuffle=False):
        super(SegDataset, self).__init__()
        self.transforms = copy.deepcopy(transforms)
        # TODO batch padding
        self.batch_transforms = None
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        self.file_list = list()
        self.labels = list()

        # TODO：非None时，让用户跳转数据集分析生成label_list
        # 不要在此处分析label file
        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)
        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 2:
                    raise Exception(
                        "A space is defined as the delimiter to separate the image and label path, " \
                        "so the space cannot be in the image or label path, but the line[{}] of " \
                        " file_list[{}] has a space in the image or label path.".format(line, file_list))
                items[0] = path_normalization(items[0])
                items[1] = path_normalization(items[1])
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not is_pic(full_path_im) or not is_pic(full_path_label):
                    continue
                if not osp.exists(full_path_im):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im))
                if not osp.exists(full_path_label):
                    raise IOError('Label file {} does not exist!'.format(
                        full_path_label))
                self.file_list.append({
                    'image': full_path_im,
                    'mask': full_path_label
                })
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.file_list[idx])
        outputs = self.transforms(sample)
        return outputs

    def __len__(self):
        return len(self.file_list)


# 构建测试集
test_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=TEST_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False
)

out_dir = r'static/imgBase'


def mainFunc_gc(path):
    # out = model.predict(path, eval_transforms)
    img = cv2.imread(path)
    # img = cv2.resize(img, (256, 256))
    model.net.eval()
    # result = np.zeros(img.shape, dtype=np.float32)
    inputImg = (img - np.min(img)) / (np.max(img) - np.min(img))
    inputImg = inputImg.astype(np.float32)
    inputImg = inputImg.transpose((2, 0, 1))
    inputImg = paddle.to_tensor(inputImg).unsqueeze(0)
    output, *_ = model.net(inputImg)
    output = output.numpy()
    output1 = output[0][0]
    output1 = ((output1 + 1.0) / 2 * 255).astype(np.uint8)

    output1 = output1 < 110
    output1 = output1.astype(np.uint8) * 255

    # pred = paddle.argmax(output[0], axis=0)
    # pred = pred.numpy().astype(np.uint8)

    path = str(time.time()) + 'gc.png'
    cv2.imwrite(osp.join(out_dir, path), output1)
    return 'imgBase/' + path
    pass


if __name__ == '__main__':
    mainFunc_gc("tempPic/gc/img_testA/3.jpg")
    pass
