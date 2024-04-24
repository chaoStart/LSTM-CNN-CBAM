# 卞庆朝
# 开发时间：2023/3/2 9:52
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from AllModel import LSTM
plt.rcParams['font.sans-serif'] = 'SimHei'
EPOCH = 5
BATCH_SIZE = 32
LR = 0.002
# 这些输入data_x1等都是归一化处理过的
data_x1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
data_x4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
data_x6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
print('data_x1的形状和大小:',data_x1.shape)
data_y1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y1.npy")
data_y4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y4.npy")
data_y6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y6.npy")
print('data_y1的形状:',data_y1.shape)
print(data_y1[:5,:])
train_x = np.append(data_x1,data_x4, axis=0)
train_x = np.append(train_x,data_x6, axis=0)
train_y = np.append(data_y1,data_y4, axis=0)
train_y = np.append(train_y,data_y6, axis=0)
print('train_x:',train_x.shape)
# 归一化
scaler = MinMaxScaler()
train_y=scaler.fit_transform(train_y)
print('train_y的形状:',train_y.shape)
print('归一化后train_y的数据\n',train_y[:5,:])
test_x = data_x1
test_y = train_y[:315,:]
'''
# 反归一化
train_y=scaler.inverse_transform(train_y)
print('train_y的形状:',train_y.shape)
print('反归一化后train_y的数据\n',train_y[:5,:])

test_y=scaler.inverse_transform(test_y)
print('反归一化后test_y的数据\n',test_y[:5,:])
'''
# torch.from_numpy()用来将数组array转换为张量Tensor
train_x,train_y= torch.from_numpy(train_x),torch.from_numpy(train_y)
# train_y = torch.from_numpy(train_y)
test_x,test_y= torch.from_numpy(test_x),torch.from_numpy(test_y)
# test_y = torch.from_numpy(test_y)
print('torch.from_numpy将数组array转化为tensor的数据类型:',type(train_x),train_x.shape)
print('torch.from_numpy将数组array转化为tensor的数据类型:',type(train_y),train_y.shape)
# 对给定的 tensor 数据，将他们包装成 dataset
train_dataset = Data.TensorDataset(train_x, train_y)
print('train_dataset是什么:',type(train_dataset),len(train_dataset))
all_num = train_x.shape[0]
print('all_num的数据类型:',train_x.shape[0])
train_num = int(all_num * 0.8)
print('train_num的数据形状',train_num)
train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])
print('train_data的数据类型和形状',type(train_data),len(train_data))
print('val_data的数据类型和形状',type(val_data),len(val_data))
# torch.utils.data.DataLoader(): 构建可迭代的数据装载器, 我们在训练的时候，每一个for循环，
# 每一次iteration，就是从DataLoader中获取一个batch_size大小的数据的。
# 把训练数据集756个放在train_loader,把测试数据集189个放在val_loader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )
test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )
print('test_loader的数据类型和形状',type(test_loader),len(test_loader))
# 加载保存好的模型
# model=torch.load('LSTMmodel.pkl')
model=torch.load('CNNmodel.pkl')
print('已经加载好模型')
# /////////////////下面是计算测试集的损失函数////////////////////
print('/////////////////下面是计算测试test集的损失函数////////////////////')
pred = torch.empty(1)#创建1个空数组tensor
model.eval()
# model.eval()的作用是不启用梯度计算，用保存好的模型进行推断。

with torch.no_grad():
# 这里的test_loader其实是C1工况的数据集，前序内容train_loader、val_loader是将C4、C6数据集作为训练集和验证集
    for i, (tx, ty) in enumerate(test_loader):
        tx = tx.float()
        ty = ty.float()
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()
        test_output = model(tx)
        # print('test_output.shape:',test_output.shape)
        out = model(tx).squeeze(-1)
        # .item() 将一个Tensor变量转换为python标量（int float等）常用于用于深度学习训练时，
        # 将loss值转换为标量并加，以及进行分类任务，计算准确值值时需要
        pred = torch.cat((pred, out.cpu()))
        # detach有什么用?
        # 如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，
        # 而不更新A中的参数值，这时候就可以使用detach()
        # numpy.delete(arr, obj, axis=None)
        # arr: 输入向量
        # obj: 表明哪一个子向量应该被移除。可以为整数或一个int型的向量
        # axis: 表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量,0表示按照行删除
# y_pred表示预测的值
y_pred = np.delete(pred.detach().numpy(), 0, axis=0)#去掉了第316个数字中第一个为0的数值
print('y_pred是什么样子的',type(y_pred),y_pred.shape)
y_pred=y_pred[:,np.newaxis]
print('y_pred是unsqueeze之后的',type(y_pred),y_pred.shape)
y_pred=scaler.inverse_transform(y_pred)
print('y_actual的形状:',y_pred.shape)
print('反归一化后y_pred(C1工况的磨损值)的数据\n',y_pred[:5,:])

# y_actual表示实际的y值
y_actual = test_y.cpu().detach().numpy()
y_actual=scaler.inverse_transform(y_actual)
print('y_actual的形状:',y_actual.shape)
print('反归一化后y_actual(C1工况的磨损值)的数据\n',y_actual[:5,:])
# 计算均方根误差
rmse = math.sqrt(mean_squared_error(y_pred, y_actual))
print('Test RMSE: %.3f' % rmse)
# 下面是保存LSTM的预测结果
# np.save('Y_PRED', y_pred)
print('预测的pred已经保存好了~')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
xx1 = range(0, 315)
# plt.title('No.1 tool wear')
plt.title('LSTM model')
plt.plot(xx1, y_actual,label='actual')
plt.plot(xx1, y_pred,label='predict')
plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
# 手动设置标签位置为左上角
plt.legend(loc='upper left')
plt.savefig("LSTM_pred.svg")
plt.show()