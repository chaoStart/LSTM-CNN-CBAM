# 卞庆朝
# 开发时间：2023/9/8 21:44
import pandas as pd
import numpy as np
# 把.csv文件转化为numpy格式的数据文件
'''
# csv_file_path1 = 'G:\pycharm\chap06_刀具磨损预测\c1_wear.csv'  # 替换为你的CSV文件路径
# csv_file_path2 = 'G:\pycharm\chap06_刀具磨损预测\c4_wear.csv'  # 替换为你的CSV文件路径
# csv_file_path3 = 'G:\pycharm\chap06_刀具磨损预测\c6_wear.csv'  # 替换为你的CSV文件路径
# data_frame1 = pd.read_csv(csv_file_path1)
# data_frame2 = pd.read_csv(csv_file_path2)
# data_frame3 = pd.read_csv(csv_file_path3)
# data_array1 = data_frame1.to_numpy()
# data_array1=data_array1[:,1:]
# data_array4 = data_frame2.to_numpy()
# data_array4=data_array4[:,1:]
# data_array6 = data_frame3.to_numpy()
# data_array6=data_array6[:,1:]
# # 保存实际c1,c4,c6的磨损值
# np.save('c1_actual.npy', data_array1)
# np.save('c4_actual.npy', data_array4)
# np.save('c6_actual.npy', data_array6)
# print(data_array6.shape)
# print(data_array6)
# print("********************************")
'''
#读取保存的输入数据（已经归一化完毕！！！）315 7 5000
'''
# data_x1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
# data_x4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
# data_x6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
# data_y1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y1.npy")
# data_y4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y4.npy")
# data_y6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y6.npy")
# print(data_x1.shape)
# print(data_x1[0,:,:10].shape)
# print(data_x1[0,:,:10])
#
# print(data_y1.shape)
# print(data_y1[:10,:].shape)
# print(data_y1[:10,:])
# # np.append[axis=0]表示两个数组上下合并
# train_x = np.append(data_x1,data_x4, axis=0)
# train_x = np.append(train_x,data_x6, axis=0)
# train_y = np.append(data_y1,data_y4, axis=0)
# train_y = np.append(train_y,data_y6, axis=0)
# test_x = data_x1
# test_y = data_y1
# print('train_x的数据类型:',type(train_x),train_x.shape)
# print('train_y的数据类型:',type(train_y),train_y.shape)
# print('test_x的数据类型:',type(test_x),test_x.shape)
# print('test_y的数据类型:',type(test_y),test_y.shape)
'''
#测试squeeze(-1)来去掉维数为1的多余维度，以及创建新的空的empty数组来接受数据
'''下面的代码表示:创建一个空的数组用来接受网络模型返回来的数据
# print('---------------------------')
# import torch
# # 创建一个形状为(3, 1, 4, 2)的张量
# x = torch.rand(3, 1, 4, 2)
# print(x.size())
# # 使用.squeeze(-2)去除倒数第二个维度的大小为1的维度
# y = x.squeeze(dim=1)
# #输出y的形状，将变为(3, 4, 2)
# print(y.size())
# y=y.unsqueeze(1)
# print(y.size())
# print('/////////////////////')
# import torch
# # 创建一个空的2x3的二维数组
# empty_tensor = torch.zeros(2, 3)
# # 输出创建的空数组
# print(empty_tensor)
'''
#下面的代码验证了：归一化只能使用一次，而反归一化可以反复使用
'''下面的代码验证了：归一化只能使用一次，而反归一化可以反复使用
print('$$$$$$$$$$$$$$$$$$$$$$$$$')
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
EPOCH = 1000
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
test_x = data_x1
test_y = data_y1
print('train_y:',train_y.shape)
print('test_y的形状:',test_y.shape)
print(test_y[:5,:])
print('train_y的形状:',train_y.shape)
print(train_y[:5,:])
# 归一化
print('进行归一化')
scaler = MinMaxScaler()
# train_y=scaler.fit_transform(train_y)
# print('train_y的形状:',train_y.shape)
# print(train_y[:5,:])

test_y=scaler.fit_transform(data_y1)
print('test_y的形状:',test_y.shape)
print(test_y[:5,:])
# 反归一化
print('进行反归一化')
# train_y=scaler.inverse_transform(train_y)
# print('train_y的形状:',train_y.shape)
# print(train_y[:5,:])

test_y=scaler.inverse_transform(test_y)
print('test_y的形状:',test_y.shape)
print(test_y[:5,:])
print('$$$$$$$$$$$$$$$$$$$$$$$$$')
'''
# 读取保存好的预测数据，并绘图
y_pred = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\Y_PRED.npy")
# y_pred = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\CNN_c1_PRED.npy")
y_actual = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y1.npy")
print(y_pred.shape)
print(y_actual.shape)
# print(y_actual[:10])
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
xx1 = range(0, 315)
# plt.plot(xx1, y_pred, color='black', label='Predicted value')
# plt.plot(xx1, y_actual, color='red', label='Actual value')
plt.title('LSTM model')
plt.plot(xx1, y_actual,label='actual')
plt.plot(xx1, y_pred,label='predict')
plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
# 手动设置标签位置为左上角
plt.legend(loc='upper left')
plt.savefig("LSTM_pred.svg")
plt.show()