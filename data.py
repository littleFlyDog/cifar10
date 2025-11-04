import torch
import os
import math
import collections
import shutil
import torchvision
#csv标签读取成字典
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))
#返回了一个字典，标签为1，2，3，4.....，值为字符
labels = read_csv_labels('./data/trainLabels.csv')

def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#复制分类文件
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = collections.defaultdict(int)
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]#文件名（数字）找字典中的字符串
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] += 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

#函数整合       
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


#图像增广
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形，如果时小变大会用插值法
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

#加载dataset
def load_dataloader_2(data_dir, batch_size):

    train_ds = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform_train) 
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, folder),
        transform=transform_test) for folder in ['valid', 'test']]

    #加载dataloader
    train_iter = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True, drop_last=True)
        

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                            drop_last=True)

    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)
    return train_iter, valid_iter, test_iter,train_ds



def load_dataloader_1(data_dir, batch_size):

    train_ds  = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform_train) 

    valid_ds = torchvision.datasets.ImageFolder(
        os.path.join(data_dir,  'valid'),
        transform=transform_test) 

    #加载dataloader
    train_iter = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True, drop_last=True,num_workers=8)

    valid_iter = torch.utils.data.DataLoader(
        valid_ds, batch_size, shuffle=False, drop_last=True,num_workers=8)
    print("jiazaiwanbi")
    return train_iter, valid_iter