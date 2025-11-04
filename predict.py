#用于预测
import pandas as pd
import torch
import tqdm




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
