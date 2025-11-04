import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
   
def train_epoch(model, features, labels, loss, trainer, device):
    features = features.to(device)
    labels = labels.to(device)
    model.train()
    trainer.zero_grad()
    pred = model(features)
    l = loss(pred, labels)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    return train_loss_sum

def accuracy(model, valid_iter, device):
    """计算在指定数据集上模型的精度"""
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时不需要计算梯度
        for features, labels in valid_iter:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def train(model, train_iter, valid_iter, num_epochs, device, loss, trainer, scheduler):
    writer = SummaryWriter('logs')
    best_acc = 0.0
    patience = 10
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}...')
        model.train()
        running_loss = 0.0
        sample_count = 0
        with tqdm.tqdm(train_iter,desc="开始训练......",position=1,leave=False) as data_iterater:
            for features, labels in data_iterater:
                l = train_epoch(model, features, labels, loss, trainer, device)
                running_loss += l.item()
                sample_count += labels.shape[0]
        avg_loss = running_loss / sample_count
        valid_acc = accuracy(model, valid_iter,device)
        if valid_acc >= best_acc:
            torch.save(model.state_dict(),'best_model.pth')
            best_acc = valid_acc
            patience=10
        else:
            patience-=1
        print(f'Epoch {epoch + 1}, 'f'loss {avg_loss:.4f}')
        print(f'Validation accuracy {valid_acc:.4f}')
        if patience==0:
            print("Early stopping triggered")
            break
        if epoch % 10 == 0:
            writer.add_scalars('Loss Accuracy', {'accuracy': valid_acc,'Train Loss': avg_loss}, epoch)

        scheduler.step()

    writer.close()
    model.load_state_dict(torch.load('best_model.pth'))#使当前模型加载为最佳模型

