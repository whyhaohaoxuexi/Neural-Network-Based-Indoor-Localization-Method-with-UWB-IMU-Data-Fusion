import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib  # For saving and loading scalers
from XLSTMModel import xlstm  # 引用你自定义的模型
from parser_my import args  # 导入 parser_my.py 中的 args
from tqdm import tqdm  # 导入tqdm
import matplotlib
import matplotlib.font_manager as fm

# 从命令行参数中获取文件路径
input_data_path = args.train_path
target_data_path = args.train_coords_path

# 加载包含NaN值的输入数据，并指定编码方式
input_df = pd.read_csv(input_data_path, encoding='utf-8')
target_df = pd.read_csv(target_data_path, encoding='utf-8')

# 打印列名以检查是否与预期一致
print("输入数据的列名:", input_df.columns)

# 检查输入数据和目标数据中的 NaN 值
input_nan_present = input_df.isnull().values.any()
target_nan_present = target_df.isnull().values.any()

if input_nan_present or target_nan_present:
    if input_nan_present:
        print("输入数据包含 NaN 值。正在进行前向填充...")
        input_df.ffill(inplace=True)  # 使用前向填充

        # 检查前向填充后是否仍存在 NaN 值
        if input_df.isnull().values.any():
            print("前向填充后仍有 NaN 值，尝试后向填充...")
            input_df.bfill(inplace=True)  # 使用后向填充

        # 显示 NaN 值所在的行和列
        nan_positions = input_df[input_df.isnull().any(axis=1)]
        print("填充后仍有 NaN 值，以下是包含 NaN 的行：")
        print(nan_positions)

        # 最终使用均值或中位数填充
        if input_df.isnull().values.any():
            print("后向填充后仍有 NaN 值，使用列的均值进行填充...")
            input_df.fillna(input_df.mean(), inplace=True)  # 或使用 input_df.median()

        # 显式将填充后的数据转换为数值类型
        input_df[['uwb1', 'uwb2', 'uwb3', 'uwb4', 'acc1', 'acc2', 'acc3', 'angle']] = \
            input_df[['uwb1', 'uwb2', 'uwb3', 'uwb4', 'acc1', 'acc2', 'acc3', 'angle']].apply(pd.to_numeric,
                                                                                              errors='coerce')

        # 检查转换后的数据类型
        print("转换后的输入数据类型：\n", input_df.dtypes)

        # 最终检查是否还有 NaN 值
        if input_df.isnull().values.any():
            print("警告：填充后仍然存在 NaN 值，请检查数据来源和处理方法。")
        else:
            print("NaN 值处理完成，输入数据已无 NaN 值。")

    if target_nan_present:
        print("目标数据包含 NaN 值。正在进行前向填充...")
        target_df.ffill(inplace=True)  # 使用前向填充

        # 检查前向填充后是否仍存在 NaN 值
        if target_df.isnull().values.any():
            print("前向填充后仍有 NaN 值，尝试后向填充...")
            target_df.bfill(inplace=True)  # 使用后向填充

        # 显示 NaN 值所在的行和列
        nan_positions_target = target_df[target_df.isnull().any(axis=1)]
        print("填充后仍有 NaN 值，以下是包含 NaN 的行：")
        print(nan_positions_target)

        # 最终使用均值填充
        if target_df.isnull().values.any():
            print("后向填充后仍有 NaN 值，使用列的均值进行填充...")
            target_df.fillna(target_df.mean(), inplace=True)

        # 显式将填充后的目标数据转换为数值类型
        target_df = target_df.apply(pd.to_numeric, errors='coerce')

        # 检查转换后的数据类型
        print("转换后的目标数据类型：\n", target_df.dtypes)

        # 最终检查是否还有 NaN 值
        if target_df.isnull().values.any():
            print("警告：填充后仍然存在 NaN 值，请检查数据来源和处理方法。")
        else:
            print("NaN 值处理完成，目标数据已无 NaN 值。")
else:
    print("输入数据和目标数据中均未发现 NaN 值。")


# 定义计算特征的函数
def compute_features(data, window_size=10):
    features = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size].astype(float)  # 确保窗口中的数据是浮点数
        feature = []
        for col in range(window.shape[1]):
            feature.extend([
                np.min(window[:, col]),
                np.max(window[:, col]),
                np.quantile(window[:, col], 0.25),
                np.quantile(window[:, col], 0.50)
            ])
        features.append(feature)
    return np.array(features)


# 提取UWB数据的特征
uwb_features = compute_features(input_df[['uwb1', 'uwb2', 'uwb3', 'uwb4']].values)

# 提取加速度数据特征
acc_features = compute_features(input_df[['acc1', 'acc2', 'acc3']].values)

# 对角度数据进行特征提取
angle_features = compute_features(input_df[['angle']].values)  # 这里使用和UWB、加速度相同的窗口

# 对齐目标数据与由于窗口减少的输入特征数量
target_coordinates = target_df[['x_coord', 'y_coord']].values[9:]  # 假设窗口大小为10

# 数据缩放
scaler_x_uwb = StandardScaler()
scaler_x_acc = StandardScaler()
scaler_x_angle = StandardScaler()
scaler_y = StandardScaler()

x_uwb = scaler_x_uwb.fit_transform(uwb_features)
x_acc = scaler_x_acc.fit_transform(acc_features)
x_angle = scaler_x_angle.fit_transform(angle_features)
y_data = scaler_y.fit_transform(target_coordinates)

# 保存缩放器
joblib.dump(scaler_x_uwb, '/root/pythonProject/data/scaler_x_uwb.pkl')
joblib.dump(scaler_x_acc, '/root/pythonProject/data/scaler_x_acc.pkl')
joblib.dump(scaler_x_angle, '/root/pythonProject/data/scaler_x_angle.pkl')
joblib.dump(scaler_y, '/root/pythonProject/data/scaler_y.pkl')
print("缩放器已保存。")

# 划分训练集、验证集和测试集，64% 训练，16% 验证，20% 测试
# 首先，划分出 20% 测试集
x_uwb_temp, x_uwb_test, x_acc_temp, x_acc_test, x_angle_temp, x_angle_test, y_temp, y_test = train_test_split(
    x_uwb, x_acc, x_angle, y_data, test_size=0.2, random_state=42
)

# 接着，从剩下的 80% 数据中划分出 64% 作为训练集，16% 作为验证集
x_uwb_train, x_uwb_val, x_acc_train, x_acc_val, x_angle_train, x_angle_val, y_train, y_val = train_test_split(
    x_uwb_temp, x_acc_temp, x_angle_temp, y_temp, test_size=0.2, random_state=42
)


# 自定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, uwb, acc, angle, target):
        self.uwb = uwb
        self.acc = acc
        self.angle = angle
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.target is not None:
            return {
                'uwb': torch.tensor(self.uwb[idx], dtype=torch.float32).unsqueeze(0),
                'acc': torch.tensor(self.acc[idx], dtype=torch.float32).unsqueeze(0),
                'angle': torch.tensor(self.angle[idx], dtype=torch.float32).unsqueeze(0),
                'target': torch.tensor(self.target[idx], dtype=torch.float32)
            }
        else:
            return {
                'uwb': torch.tensor(self.uwb[idx], dtype=torch.float32).unsqueeze(0),
                'acc': torch.tensor(self.acc[idx], dtype=torch.float32).unsqueeze(0),
                'angle': torch.tensor(self.angle[idx], dtype=torch.float32).unsqueeze(0)
            }


# 创建 DataLoader
train_dataset = TrajectoryDataset(x_uwb_train, x_acc_train, x_angle_train, y_train)
val_dataset = TrajectoryDataset(x_uwb_val, x_acc_val, x_angle_val, y_val)
test_dataset = TrajectoryDataset(x_uwb_test, x_acc_test, x_angle_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 实例化你的XLSTM模型并定义损失函数和优化器
model = xlstm(
    uwb_input_size=args.uwb_input_size,
    acc_input_size=args.acc_input_size,
    angle_input_size=args.angle_input_size,
    hidden_size=args.hidden_size,
    num_layers=args.layers,
    output_size=2,
    dropout=args.dropout,
    batch_first=args.batch_first
).to(args.device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 训练模型并在验证集上保存最佳模型
num_epochs = args.epochs
best_val_loss = float('inf')
best_model_path = '/root/pythonProject/duibishiyanmodel/best_model_xlstm.pth'

# 创建列表保存每个 epoch 的损失
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="white"):
        optimizer.zero_grad()
        outputs = model(
            batch['uwb'].to(args.device),
            batch['acc'].to(args.device),
            batch['angle'].to(args.device)
        )
        main_loss = criterion(outputs, batch['target'].to(args.device))
        main_loss.backward()
        optimizer.step()
        epoch_loss += main_loss.item()

    # 计算平均训练损失
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证集评估
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:  # 使用验证集
            outputs = model(
                batch['uwb'].to(args.device),
                batch['acc'].to(args.device),
                batch['angle'].to(args.device)
            )
            val_loss += criterion(outputs, batch['target'].to(args.device)).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # 打印训练和验证损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'新最佳模型已保存，验证集损失: {best_val_loss:.4f}')

import time  # 引入时间模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
print(f'最佳模型已加载。')

# 绘制训练集和验证集损失对比图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 保存图像到指定路径
save_path = '/root/pythonProject/loss_duibi/loss_comparison_zhuzi.png'
plt.savefig(save_path)
plt.show()

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import matplotlib

# 开始记录测试时间
start_time = time.time()

# 可视化实际坐标与预测坐标
model.eval()  # 设置模型为评估模式
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:  # 使用测试集
        outputs = model(
            batch['uwb'].to(args.device),
            batch['acc'].to(args.device),
            batch['angle'].to(args.device)
        )
        predictions.append(outputs.cpu().numpy())
        actuals.append(batch['target'].to('cpu').numpy())  # 将实际值移动到CPU

# 结束测试时间记录
end_time = time.time()
test_time = end_time - start_time
print(f'测试时间: {test_time:.4f} 秒')

# 将预测和实际值拼接
predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# 对预测值和真实值进行逆标准化处理
predictions_rescaled = scaler_y.inverse_transform(predictions)
actuals_rescaled = scaler_y.inverse_transform(actuals)

# 将数据保留到小数点后两位
predictions_rescaled = np.round(predictions_rescaled, 2)
actuals_rescaled = np.round(actuals_rescaled, 2)

# 保存实际值与预测值到文件
comparison_df = pd.DataFrame({
    'x_actual': actuals_rescaled[:, 0],
    'y_actual': actuals_rescaled[:, 1],
    'x_pred': predictions_rescaled[:, 0],
    'y_pred': predictions_rescaled[:, 1]
})
comparison_file = '/root/pythonProject/zuizhongcomparisonszhuzi.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f'对比数据已保存到 {comparison_file}')

# 从文件加载对比数据
loaded_df = pd.read_csv(comparison_file)

# 提取实际值和预测值
actuals_rescaled_loaded = loaded_df[['x_actual', 'y_actual']].values
predictions_rescaled_loaded = loaded_df[['x_pred', 'y_pred']].values

# 计算欧几里得误差
errors_loaded = np.sqrt(np.sum((predictions_rescaled_loaded - actuals_rescaled_loaded) ** 2, axis=1))

# 统计误差
max_error_loaded = np.max(errors_loaded)
min_error_loaded = np.min(errors_loaded)
mean_error_loaded = np.mean(errors_loaded)
count_02m_loaded = np.sum(errors_loaded <= 0.2)
percentage_02m_loaded = (count_02m_loaded / len(errors_loaded)) * 100
count_1m_loaded = np.sum(errors_loaded <= 1.0)
percentage_1m_loaded = (count_1m_loaded / len(errors_loaded)) * 100

# 打印加载后计算的误差结果
print(f'最大误差: {max_error_loaded:.4f} m')
print(f'最小误差: {min_error_loaded:.4f} m')
print(f'平均误差: {mean_error_loaded:.4f} m')
print(f'0.2m以内的坐标数量: {count_02m_loaded}，百分比: {percentage_02m_loaded:.2f}%')
print(f'1m以内的坐标数量: {count_1m_loaded}，百分比: {percentage_1m_loaded:.2f}%')

# 绘制实际值与预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(loaded_df['x_actual'], loaded_df['y_actual'], color='blue', label='Actual')
plt.scatter(loaded_df['x_pred'], loaded_df['y_pred'], color='red', label='Predicted')
plt.title('Actual vs Predicted Coordinates (From File)')
plt.xlabel('x_coord')
plt.ylabel('y_coord')
plt.legend()

# 保存对比图
comparison_plot_file = '/root/pythonProject/duibishiyantu/trajectory_comparison_from_file_zhuzi.png'
plt.savefig(comparison_plot_file)
print(f'对比图已保存到 {comparison_plot_file}')
plt.show()

# 绘制误差分布图
plt.figure(figsize=(10, 6))
plt.hist(errors_loaded, bins=50, edgecolor='k', alpha=0.7)
plt.title('预测误差分布')
plt.xlabel('误差 (欧几里得距离)')
plt.ylabel('频率')

# 保存误差分布图
error_distribution_file = '/root/pythonProject/error_show/error_distribution_zuizhuzi.png'
plt.savefig(error_distribution_file)
print(f'误差分布图已保存到 {error_distribution_file}')
plt.show()
