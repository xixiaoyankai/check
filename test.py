import os

from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #防止报错

import numpy as np
import pandas as pd
import torch
import torch.nn
import matplotlib.pyplot as plt
import d2l
#
# import sys
# sys.path.append("..")

#--------------数据预处理-------------#
train_data=pd.read_csv('house_data/train.csv')
test_data=pd.read_csv('house_data/test.csv')
#train_data=train_data.iloc[:, 1:]
#test_data=test_data.iloc[:, 1:]  #都去除前面的1

m=train_data.shape[0]

alltrain=pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_idx=alltrain.dtypes[alltrain.dtypes !='object'].index #获取类
alltrain[numeric_idx]=alltrain[numeric_idx].apply(
    lambda x: (x-x.mean())/(x.std()))
alltrain[numeric_idx]=alltrain[numeric_idx]
alltrain=alltrain.fillna(0)
alltrain=pd.get_dummies(alltrain, dummy_na=True)
print(alltrain.shape)



xtrain= torch.tensor(alltrain.iloc[:m,:].values)
xtest = torch.tensor(alltrain.iloc[m,:].values)
y= torch.tensor(train_data.iloc[:, -1].values.reshape(-1,1))


#alltrain = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

#-------------建立模型-----------#


loss = torch.nn.MSELoss()
def get_net(feature_num):
    net = nn.Linear(feature_num, 1) #
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

get_net(alltrain.shape[1])
def log_rmse(net, x, y):
    with torch.no_grad():
        # 将⼩于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(x.float()), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), y.log()).mean())
        return rmse.item()

#--------------训练模型------------#

def train(net, xtrain, ytrain, xtest, ytest, epoch, learning_rate, weight, batch_size):
    #返回训练中每次得到的误差
    train_ls, test_ls=[], []
    dataset=torch.utils.data.TensorDataset(xtrain, ytrain)
    train_iter=torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer=torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight)

    net=net.float()
    for epoch in range(epoch):
        for x, y in train_iter:
            l=loss(net(x.float()), y.float()) #mse和rmse成正比
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, xtrain, ytrain))
        if ytest is not None:
            test_ls.append(log_rmse(net, xtest, ytest))
    return train_ls, test_ls

def get_k_fold_data(k, i, x, y):
    assert k>1
    fold_size=x.shape[0]//k #整除
    x_train, y_train=None, None
    for j in range(k):
        idx=slice(j*fold_size, (j+1)*fold_size)
        x_part, y_part=x[idx, :], y[idx]
        if j==i:
            x_valid, y_valid=x_part, y_part
        elif x_train is None: x_train, y_train=x_part, y_part
        else:
            x_train=torch.cat((x_train, x_part), dim=0)
            y_train=torch.cat((y_train, y_part), dim=0)

    return x_train, y_train, x_valid, y_valid
#---------------测试模型-------------#

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小。"""
    plt.rcParams['figure.figsize'] = figsize
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """画对数y的图形。"""
    set_figsize(figsize)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum=0, 0
    for i in range(k):
        data=get_k_fold_data(k, i, x_train, y_train)
        net=get_net(x_train.shape[1])
        train_ls, valid_ls=train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum+=train_ls[-1]
        valid_l_sum+=valid_ls[-1]
        if i==0:
            #semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse', range(1, num_epochs+1), valid_ls, ['train', 'valid'])
            pass
        print('fold %d, train rmse %f, valid rmse %f'%(i, train_ls[-1], valid_ls[-1]))
    return train_l_sum/k, valid_l_sum/k #返回k次平均误差

k, epochs, lr, weight, batch_size=5, 100, 5, 0, 64

train_l, valid_l=k_fold(k, xtrain, y, epochs, lr, weight, batch_size)
print('avg train rmse %f, valid rmse %f'%(train_l, valid_l))






#---------------预测模型--------------#

def train_and_pred(xtrain, xtest, ytrain, test_data, epochs, lr, w, bat ):
    net=get_net(xtrain.shape[1])
    train_ls,_=train(net, xtrain, ytrain, None, None, epochs, lr, w, bat)
    semilogy(range(1, epochs+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f'%train_ls[-1])
    print(xtest)
    preds=net(xtest.float()).detach().numpy()
    test_data['SalePrice']=pd.Series(preds.reshape(1, -1)[0])
    print(preds.shape)
    submission=pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


train_and_pred(xtrain, xtest, y, test_data, epochs, lr, weight, batch_size)