#加载resnet18
from model import resnet18
import torch
from torch import nn
import torchvision
from data import load_dataloader_1,load_dataloader_2, reorg_cifar10_data
from train import train
from predict import predict_kaggle,predict_myimg
#超参数设置
num_epochs=100 #30
batch_size=64 #128
valid_ratio=0.1

#创建模型
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=resnet18(num_classes=10,in_channels=3)
model.to(device)

#损失函数
loss = nn.CrossEntropyLoss(reduction="none")
loss=loss.to(device)

#优化器设置
lr=0.001
wd=5e-4
lr_period=4
lr_decay=0.9
trainer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                            weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

#超参数设置

#加载数据集
data_dir='./train_valid_test'
# reorg_cifar10_data(data_dir, valid_ratio)
if __name__ == "__main__":


    # # train_iter, valid_iter, train_valid_iter,test_iter=load_dataloader(data_dir, batch_size)
    # train_iter, valid_iter=load_dataloader_1(data_dir, batch_size)
    # #--------------------开始训练--------------------
    # train(
    #     model=model, 
    #     train_iter=train_iter, 
    #     valid_iter=valid_iter, 
    #     num_epochs=num_epochs, 
    #     device=device, 
    #     loss=loss, 
    #     trainer=trainer, 
    #     scheduler=scheduler
    # )










    #--------------------开始预测kaggle--------------------
#     model.load_state_dict(torch.load('./best_model.pth'))
# #net, train_iter, valid_iter, num_epochs, device, loss, trainer, scheduler
#     train_iter, valid_iter,test_iter,train_ds=load_dataloader_2(data_dir, batch_size)
#     predict_kaggle(
#         model=model,
#         test_iter=test_iter,
#         train_ds=train_ds,
#         len=300000,
#         device=device
#     )





    #--------------------开始预测单张图片--------------------
    model.load_state_dict(torch.load('./best_model.pth'))
    img_path='./myimg/test6.jpg'
    predict_myimg(
        img_path=img_path,
        model=model,
        device=device
    )