# -*- coding = utf-8 -*-
# @Time :2022-11-28 18:54
# @Author :宁坡
# @File : GA-LSTM.py
# @Software : PyCharm
import keras
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dropout, Flatten, Dense
from keras.metrics import metrics
from keras.saving.save import load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras import backend as K, Sequential, Input
from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.core.problem import ElementwiseProblem
# from pymoo.factory import get_termination
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from sklearn.model_selection import train_test_split
##设置GPU
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 限制GPU占用率
sess = tf.compat.v1.Session(config=config)
K.set_session(sess) # 设置session
'''
log:
上证：22,5，60
创业板：22，5,60
'''

# 设置一些参数,需要进行调整
lookback = 60  # 回望历史天数，分别为 5、22、60、240
data_file = "上证_teh_cpi_index.csv"  # 设置读取数据文件地址和名称，跑自己负责的
mode_file = "./model/SH_60LSTM_model"  # 设置模型保存地址和名称，设置成自己的，同一文件夹下直接输入文件名就行
err_file = "./error/SH_60error_LSTM.csv"  # 设置评价指标保存地址和名称，设置成自己的，同一文件夹下直接输入文件名就行
pred_graph_file = "./预测图/SH-60GA-LSTM" # 预测图的保存地址，设置成自己的，同一文件夹下直接输入文件名就行
loss_graph_file = "./预测图/SH-60GA-LSTM模型训练损失图"  # 训练损失图的保存地址，设置成自己的，同一文件夹下直接输入文件名就行

## 定义一些函数
# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 设置支持负号显示
def get_loss_visualize(loss, val_loss, name=""):
    """loss图"""
    plt.figure(0)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    # plt.title('{}'.format(name))
    plt.xlabel('Epoch times')
    plt.ylabel('Loss')
    plt.savefig(name, dpi=1200, bbox_inches='tight')
    # plt.clf()
    plt.show()
def get_predict_visualize(Y_test_predict, Y_test_real, name=""):
    """预测的结果对比图"""
    plt.figure(1)
    plt.figure(figsize=(20, 4))
    plt.plot(Y_test_predict, "b", label="Predict")
    plt.plot(Y_test_real, "r", linestyle=':', label="Real")
    # plt.title('{}预测曲线'.format(name),fontsize=18)
    plt.xlabel('时间',fontsize=18)
    plt.ylabel('收盘价',fontsize=18)
    plt.tick_params(axis='both', labelsize=18)  # 设置刻度字体大小
    plt.legend(fontsize=20)
    plt.savefig(name, dpi=1200, bbox_inches='tight')
    # plt.clf()
    plt.show()

# 滑动时间窗口函数
def sliding_windows(data, lookback, pre_time=1, jump=1):
    # 获取数据并分割输入X和输出Y，滑动窗口设置，# (Sample,Timestep,Features)
    X, Y = [], []
    for i in range(0, len(data) - lookback - pre_time + 1, jump):
        X.append(data[i:(i + lookback)])
        Y.append(data[(i + lookback):(i + lookback + pre_time)])
    X = np.array(X)
    Y = np.array(Y)
    # print('X.shape, Y.shape:\n', X.shape,Y.shape)
    return X, Y
# 划分测试集 训练集函数
def data_split(x, y, split_ratio):

    # from sklearn.model_selection import train_test_split
    y_train_true, y_test_true = train_test_split(y, shuffle = False, test_size = split_ratio)
    X_train_true, X_test_true = train_test_split(x, shuffle = False, test_size = split_ratio)
    # # 再把训练集的10%划分为验证集
    X_train_true,X_validate_true = train_test_split(X_train_true,test_size = .1, shuffle = False)
    y_train_true,y_validate_true = train_test_split(y_train_true,test_size = .1, shuffle = False)
    print("===============训练集验证集和测试集划分完毕===================")
    print('X_train_true',X_train_true.shape)
    print("X_validate_true", X_validate_true.shape)
    print("X_test_true", X_test_true.shape)
    print('y_train_true',y_train_true.shape)
    print('y_validate_true',y_validate_true.shape)
    print('y_test_true',y_test_true.shape)

    return X_train_true, X_validate_true, X_test_true, y_train_true, y_validate_true, y_test_true
# 标准化函数
def normal_roll( X_train_true, X_validate_true, X_test_true, y_train_true, y_validate_true, y_test_true):
    """
    标准化和滑动窗口设置
    """
    # 时间序列，不能打乱，需要分别标准化
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    mm_X = MinMaxScaler()
    # mm_X_validate = MinMaxScaler()
    # mm_X_test = MinMaxScaler()

    mm_y = MinMaxScaler()
    # mm_y_validate = MinMaxScaler()
    # mm_y_test = MinMaxScaler()
    X_scaler_param = mm_X.fit(X_train_true)
    y_scaler_param = mm_y.fit(np.array(y_train_true).reshape(-1, 1))

    X_train_s = X_scaler_param.transform(X_train_true)
    X_validate_s = X_scaler_param.transform(X_validate_true)
    X_test_s = X_scaler_param.transform(X_test_true)

    y_train_s = y_scaler_param.transform(np.array(y_train_true).reshape(-1, 1))
    y_validate_s = y_scaler_param.transform(np.array(y_validate_true).reshape(-1, 1))
    y_test_s = y_scaler_param.transform(np.array(y_test_true).reshape(-1, 1))  # [0,1]

    # 滑动窗口设置
    X_test, void = sliding_windows(X_test_s,lookback)
    void, y_test = sliding_windows(y_test_s,lookback)
    y_test = np.array([list(a.ravel()) for a in y_test])
    # X_test = np.array([list(a.ravel()) for a in X_test])

    X_validate, void = sliding_windows(X_validate_s,lookback)
    void, y_validate = sliding_windows(y_validate_s,lookback)
    y_validate = np.array([list(x.ravel()) for x in y_validate])
    # X_validate = np.array([list(a.ravel()) for a in X_validate])

    X_train, void = sliding_windows(X_train_s,lookback)
    void, y_train = sliding_windows(y_train_s,lookback)
    y_train = np.array([list(x.ravel()) for x in y_train])
    # X_train = np.array([list(a.ravel()) for a in X_train])
    print("===============所有数据形状处理完毕===================")
    print('X_train   ：', X_train.shape)
    print('X_validate：', X_validate.shape)
    print('X_test    ：', X_test.shape)
    print('y_train   ：', y_train.shape)
    print('y_validate：', y_validate.shape)
    print('y_test    ：', y_test.shape)
    return X_train, X_validate, X_test, y_train, y_validate, y_test, mm_y
# 计算R2函数
def r_square(y_true, y_pred):
    # 计算R2的函数
    SSR = K.sum(K.square(y_pred - K.mean(y_true)))
    SST = K.sum(K.square(y_true - K.mean(y_true)))
    return SSR/SST
# 计算出预测值函数,用于画图
def Predict(model, X_test, y_test, mm_y_test):
    '''
    传入参数：模型，X_test,y_test 归一化y的参数
    '''
    predict_y = model.predict(X_test)
    # 反归一化预测结果得到预测值
    predict = mm_y_test.inverse_transform(predict_y)
    # 反归一化y_test得到真实值
    real = mm_y_test.inverse_transform(y_test)
    return predict, real


# LSTM基础模型
def lstm_model(param, X_train, y_train, X_validate, y_validate):

    dense_units = round(param["dense_units"]) # 全连接输出维度,四舍五入取整数
    dropout_rate = param["dropout_rate"] # dropout_rate
    lstm_units=param["lstm_units"] # lstm输出维度列表
    lstm_layer_num=round(param["lstm_layer_num"]) # lstm层数
    learning_rate = param["learning_rate"] # 学习率
    assert len(lstm_units)>=lstm_layer_num

# 模型基本结构
    keras.utils.set_random_seed(66)

    model = Sequential()
    model.add(Input((input_dim, feature_size))), # 输入层
    for i in range(lstm_layer_num):
        model.add(LSTM(units = round(lstm_units[i]), activation = 'tanh', return_sequences = True)), # return_sequences=True表示将结果传到下一步
        model.add(Dropout(dropout_rate))
    # model.add(Dropout(dropout_rate)),  # 正则化，表示删除一些神经元
    model.add(Flatten()), # 拉平
    model.add(Dense(units = dense_units, activation = "tanh"))  # 全连接层
    model.add(Dense(output_dim))  # 输出层

    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'mse', metrics = [
        metrics.mean_squared_error,
        # metrics.RootMeanSquaredError,
        metrics.mean_absolute_error,
        metrics.mean_absolute_percentage_error,
        # metrics.mean_squared_logarithmic_error
        r_square
    ]
                  )
    history= model.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_data = (X_validate, y_validate), shuffle = False,
                # validation_split=0.1,
                # callbacks=[checkpoint],
                verbose = 0)

    return model, history

# 1、数据读取与处理（特征工程）
data = pd.read_csv(data_file)
df = data
# df = data.drop(columns=['index'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.set_index('date')
df.dropna(inplace=True)  # 删除数据集中含有空值的行
df.sort_index(inplace=True)  # 按时间对时间序列排序
closing_prices = df['收盘价']  # 需要预测的指标

full_feature_dataset = df.to_numpy()  # 只取数值
print("===============数据划分sample&label===================")
print("整个数据集X大小:", full_feature_dataset.shape)
print("需要预测的Y大小:", closing_prices.shape)
X_train_true, X_validate_true, X_test_true, y_train, y_validate_true, y_test_true = data_split(full_feature_dataset, closing_prices, split_ratio = 0.2)
X_train, X_validate, X_test, y_train, y_validate, y_test, mm_y = normal_roll(X_train_true, X_validate_true,
                                                                                      X_test_true, y_train,
                                                                                      y_validate_true,
                                                                                      y_test_true)  # 数据集标准化
# 设置LSTM参数
batch_size = 64
input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]


# GA优化的一些参数设置
down = [2, 0.05, 1, 1e-4, 10] # 下界 依次是:dense_units, dropout_rate, lstm_layer_num, learning_rate, lstm_units,
up = [100, 0.5, 4, 1e-2, 256] # 上界 依次是:dense_units, dropout_rate, lstm_layer_num, learning_rate, lstm_units,
lstm_layer_max_num = 5 # LSTM层的最大数量
lstm_units_max = 100 # LSTM输出维度的最大值
down.append(1)
up.append(lstm_layer_max_num)
for _ in range(lstm_layer_max_num):
    down.append(2)
    up.append(lstm_units_max)
print(down)
print(up)
# GA-class
class MyProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        # self.cost_matrix = cost_matrix
        super().__init__(n_var = len(up),  # 变量数
                         n_obj = 1,  # 目标数
                         n_constr = 0,  # 约束数
                         xl = down,  # 变量下界
                         xu = up,  # 变量上界
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        dense_units, dropout_rate, lstm_layer_num, learning_rate, lstm_units  = x[0], x[1], x[2], x[3], x[4:]
        params = {"dense_units": int(dense_units),  "lstm_units": lstm_units, "dropout_rate": dropout_rate, "lstm_layer_num":lstm_layer_num, "learning_rate":learning_rate}
        model_temp, history_temp = lstm_model(params, X_train, y_train, X_validate, y_validate) # 传入参数字典
        c = model_temp.evaluate(X_validate, y_validate, verbose=0)
        # pre = model_temp(X_validate)
         # 反归一化预测结果得到预测值
        # predict = mm_y_test.inverse_transform(pre)
        # 反归一化y_test得到真实值
        # real = mm_y_test.inverse_transform(y_validate)
        # mape = np.mean(np.abs((predict - real) / real))
        out["F"] = np.array([c[2]]) # 定义目标函数以键“F”添加到字典中,mape

termination = get_termination("n_gen", 15)# 迭代10次
# 定义遗传算法
algorithm = GA(
    pop_size = 30,  # 种群数量
    eliminate_duplicates = True,
    n_offsprings = 10, # 新生种群数量
    crossover = SBX(), # 交叉方法
    mutation = PM(), # 变异方法，多项式突变(PM, Polynomial Mutation)算子
    selection = TournamentSelection(func_comp=comp_by_cv_and_fitness), # 选择方法
)
res = minimize(MyProblem(), algorithm, termination, seed = 66, verbose = True, save_history = True)
print("优化用时:{}分钟".format((res.exec_time)/60))


# 优化后的超参数
x=res.X
dense_units, dropout_rate, lstm_layer_num, learning_rate, lstm_units  = x[0], x[1], x[2], x[3], x[4:]
params = {"dense_units": int(dense_units), "dropout_rate": dropout_rate, "lstm_layer_num":lstm_layer_num, "learning_rate":learning_rate, "lstm_units": lstm_units}
print("优化后的参数",params)

model_lstm, history = lstm_model(params, X_train, y_train, X_validate, y_validate)
model_lstm.summary()

# 模型评估
err = model_lstm.evaluate(X_test, y_test,verbose=1)
error = pd.DataFrame(err,index = ['MAE','MSE','MAPE','SDAPE','R2'])


loss = history.history['loss']
val_loss = history.history['val_loss']
predict, real = Predict(model_lstm, X_test, y_test, mm_y)




error.to_csv(err_file)
model_lstm.save(mode_file)  # 保存模型
get_loss_visualize(loss, val_loss, name=loss_graph_file)
get_predict_visualize(predict, real, name=pred_graph_file)

print("运行完毕")
