#用于预测
from numpy import argmax
import pandas as pd
import torch
import tqdm
from PIL import Image
from torchvision import transforms


def predict_kaggle(model,test_iter,train_ds, len,device):
    preds=[]
    with tqdm.tqdm(test_iter,desc="开始预测......",position=1,leave=True) as data_iterater:
        for X, _ in data_iterater:
            y_hat = model(X.to(device))
            preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

    sorted_ids = list(range(1, len + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_ds.classes[x])
    df.to_csv('submission.csv', index=False)

def predict_myimg(img_path, model, device):
    cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
    #单个图片预处理
    img = Image.open(img_path)
    img= img.convert('RGB')
    img.show()
    transform_img=transforms.Compose([transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0),ratio=(1.0, 1.0)),transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])
    img=transform_img(img)
    img=torch.reshape(img,(1,3,32,32))

    #进行训练
    model.eval()
    img=img.to(device)
    with torch.no_grad():
        output=model(img)
    index=output.argmax(-1).item()
    print(f'该图片为:{cifar10_classes[index]}')
    

