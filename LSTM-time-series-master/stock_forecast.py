# coding: utf-8
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置相关参数
days_for_train = 25    # 时间戳
epoch = 100             # 训练次数

# 使用gpu训练
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# 加载数据
df = pd.read_csv('data/SH600519.csv')
# print(df.head())
# print(df.info())
# 不要列の削除
df = df.drop(columns=df.columns[[0]])
df = df.drop(['open', 'low', 'high', 'volume', 'code'], axis=1)
# データの並び替え
df.sort_values(by='date', ascending=True, inplace=True)
# 終値の25日移動平均(25MA)を算出
df['25MA'] = df['close'].rolling(window=25, min_periods=0).mean()
# print(df.head())
# 終値と25日移動平均を見る
plt.figure()
plt.title('Z_Holdings')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(df['date'], df['close'], color='black',
         linestyle='-', label='close')
plt.plot(df['date'], df['25MA'], color='red',
         linestyle='--', label='25MA')
plt.legend()  # 凡例
# plt.savefig('Z_Holdings.png')  # 図の保存
plt.show()

# 数据标准化
ma = df['25MA'].values.reshape(-1, 1)
scaler = StandardScaler()
ma_std = scaler.fit_transform(ma)


# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
def data_split(data, days_for_train):
    x, y = [], []
    for i in range(len(data) - days_for_train):
        end_idx = i + days_for_train
        seq_x, seq_y = data[i:end_idx], data[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# 前(2426-300=2126)天的开盘价作为训练集,后300天的开盘价作为测试集
x_train, y_train = data_split(ma_std[:2126], days_for_train)
x_test, y_test = data_split(ma_std[2126:], days_for_train)

# データの形状を確認
print("x_train size: {}".format(x_train.shape))
print("y_train size: {}".format(y_train.shape))
print("x_test size: {}".format(x_test.shape))
print("y_test size: {}".format(y_test.shape))

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_batch = DataLoader(
    dataset=train_dataset,  # データセットの指定
    batch_size=128,  # バッチサイズの指定
    shuffle=True  # シャッフルするかどうかの指定
    )  # コアの数
test_batch = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False)

# ミニバッチデータセットの確認
for data, label in train_batch:
    print("batch data size: {}".format(data.size()))  # バッチの入力データサイズ
    print("batch label size: {}".format(label.size()))  # バッチのラベルサイズ
    break


# 建构 LSTM模型
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 100, batch_first=True, num_layers=1)
        self.fc = nn.Linear(100, 1)
        
    def forward(self, x):
        # x is input, size (seq_size, batch_size, feature_size)
        out, _ = self.lstm(x)
        # out is output, size (seq_size, batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        return out
    

model = LSTM()
model.to(device)
# print(model)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss_list = []  # 学習損失
test_loss_list = []  # 評価損失

for i in range(epoch):
    # エポックの進行状況を表示
    print('---------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))
    
    # 損失の初期化
    train_loss = 0  # 学習損失
    test_loss = 0  # 評価損失
    
    model.train()
    for data, label in train_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)
        # 勾配を初期化
        optimizer.zero_grad()
        # データを入力して予測値を計算（順伝播）
        y_pred = model(data)
        # 損失（誤差）を計算
        loss = loss_function(y_pred, label)
        # 勾配の計算（逆伝搬）
        loss.backward()
        # パラメータ（重み）の更新
        optimizer.step()
        # ミニバッチごとの損失を蓄積
        train_loss += loss.item()
        
    # ミニバッチの平均の損失を計算
    batch_train_loss = train_loss / len(train_batch)
    # ---------学習パートはここまで--------- #
    
    # ---------評価パート--------- #
    model.eval()
    # 評価時の計算で自動微分機能をオフにする
    with torch.no_grad():
        for data, label in test_batch:
            # GPUにTensorを転送
            data = data.to(device)
            label = label.to(device)
            # データを入力して予測値を計算（順伝播）
            y_pred = model(data)
            # 損失（誤差）を計算
            loss = loss_function(y_pred, label)
            # ミニバッチごとの損失を蓄積
            test_loss += loss.item()
            
    # ミニバッチの平均の損失を計算
    batch_test_loss = test_loss / len(test_batch)
    # ---------評価パートはここまで--------- #
    
    # エポックごとに損失を表示
    print("Train_Loss: {:.6f} Test_Loss: {:.6f}".format(batch_train_loss, batch_test_loss))
    # 損失をリスト化して保存
    train_loss_list.append(batch_train_loss)
    test_loss_list.append(batch_test_loss)
   
# 損失
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()  # 凡例
plt.show()  # 表示

model.eval()
# 推定時の計算で自動微分機能をオフにする
with torch.no_grad():
    # 初期化
    pred_ma = []
    true_ma = []
    for data, label in test_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)
        # 予測値を計算：順伝播
        y_pred = model(data)
        pred_ma.append(y_pred.view(-1).tolist())
        true_ma.append(label.view(-1).tolist())
        
pred_ma = [elem for lst in pred_ma for elem in lst] 
true_ma = [elem for lst in true_ma for elem in lst]

pred_ma = scaler.inverse_transform(np.reshape(pred_ma, (-1, 1)))
true_ma = scaler.inverse_transform(np.reshape(true_ma, (-1, 1)))
pred_ma = pred_ma.reshape(-1)
true_ma = true_ma.reshape(-1)
# 平均絶対誤差を計算します
mae = mean_absolute_error(true_ma, pred_ma)
print("MAE: {:.6f}".format(mae))

date = df['date'][-1 * 275:].values.reshape(-1)  # テストデータの日付
test_close = df['close'][-1 * 275:].values.reshape(-1)  # テストデータの終値
plt.figure()
plt.title('Stock Price Prediction')
plt.xlabel('date')
plt.ylabel('Stock Price')
plt.plot(date, test_close, color='green',
         linestyle='-', label='close')
plt.plot(date, true_ma, color='dodgerblue',
         linestyle='--', label='true_25MA')
plt.plot(date, pred_ma, color='red',
         linestyle=':', label='predicted_25MA')
plt.legend()  # 凡例
plt.xticks(date[0::30])
plt.xticks(rotation=30)  
plt.show()
