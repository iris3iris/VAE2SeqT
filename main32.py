# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from data_test import *
from transformer3 import Transformer
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
# 步骤 1: 读取文件
def read_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            # 步骤 2: 将每行转换为浮点数列表
            sequence = [float(num) for num in line.strip().split()]
            sequences.append(sequence)
    return sequences

# 步骤 3: 转换为PyTorch张量
def convert_to_tensor(sequences):
    return torch.tensor(sequences, dtype=torch.float32)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # self.quantile_loss = QuantileLoss(quantile)
        # self.huberLoss = nn.HuberLoss(delta=delta)
        # self.emdloss = EMDLoss()
        # self.alpha = alpha
        # self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_true, y_pred):
        # trend_loss_value = trend_preservation_loss(y_pred, y_true)
        # huberLoss_value = self.huberLoss(y_true, y_pred)
        # mean_loss = self.mse_loss(y_pred, y_true)
        # emdloss_value = self.emdloss(y_pred, y_true)
        L1loss_value = self.l1_loss(y_pred, y_true)
        # print("y_pred shape:", y_pred.shape)
        # print("y_true shape:", y_true.shape)
        d_t_pred = torch.zeros_like(y_true)
        d_t_true = torch.zeros_like(y_true)

        # 正确计算 d_t_pred 和 d_t_true
        d_t_pred = y_pred[:, 1:] - y_pred[:, :-1]
        d_t_true = y_true[:, 1:] - y_true[:, :-1]

        # L_D 损失
        L_D = torch.mean((d_t_pred - d_t_true)**2 )
        # L_D = self.l1_loss(d_t_pred,d_t_true)

        # 符号差异计算
        sign_diff = (torch.sign(d_t_pred) != torch.sign(d_t_true)).float()
        rho = torch.mean(sign_diff)

        # MSE 损失
        mse_loss_value = self.mse_loss(y_pred, y_true)

        return rho * mse_loss_value + (1 - rho) * L_D

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=3000, patience=10):
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for enc_inputs, dec_inputs, dec_outputs in train_loader:  # enc_inputs : [batch_size, src_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            dec_outputs = dec_outputs.view_as(outputs)
            loss = criterion(outputs[:, -20:], dec_outputs[:, -20:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        val_loss,_ = validate_model(model, criterion, val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')

    #     # 保存最佳模型权重
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         best_model_wts = model.state_dict()
    #         early_stop_counter = 0
    #         torch.save(model.state_dict(), 'best_model32_2.pth')
    #     else:
    #         early_stop_counter += 1
    #
    #     # 检查早停
    #     if early_stop_counter >= patience:
    #         print(f"Early stopping at epoch {epoch + 1}")
    #         break
    #
    #     scheduler.step()
    #
    # model.load_state_dict(best_model_wts)


    # 保存损失到txt文件
    with open('losses32_3.txt', 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    return model

def validate_model(model, criterion, val_loader):
    model.eval()
    total_loss = 0
    all_predictions = []

    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in val_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            dec_outputs = dec_outputs.view_as(outputs)
            loss = criterion(outputs[:, -20:], dec_outputs[:, -20:])
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    return total_loss / len(val_loader), all_predictions


def calculate_mse(model, val_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in val_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(dec_outputs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse_loss = mean_squared_error(all_targets, all_predictions)

    with open('best_val_mse_loss8.txt', 'w') as f:
        f.write(f"Best Val MSE Loss: {mse_loss:.6f}\n")

    return mse_loss

if __name__ == "__main__":

    # enc_inputs, dec_inputs, dec_outputs = make_data()
    file_path = 'scaled_input32_all_max_min.txt'  # 这里替换成你的文件路径
    file_path2 = 'dec_input_32.txt'  # 这里替换成你的文件路径
    file_path3 = 'dec_output_32.txt'  # 这里替换成你的文件路径
    sequences = read_sequences(file_path)
    sequences2 = read_sequences(file_path2)
    sequences3 = read_sequences(file_path3)
    enc_inputs = convert_to_tensor(sequences)
    dec_inputs = convert_to_tensor(sequences2)
    dec_outputs = convert_to_tensor(sequences3)
    enc_train, enc_val, dec_train, dec_val, out_train, out_val = train_test_split(
        enc_inputs, dec_inputs, dec_outputs, test_size=0.15, random_state=42)
    # 创建训练集和验证集的数据集对象
    train_dataset = MyDataSet(enc_train, dec_train, out_train)
    val_dataset = MyDataSet(enc_val, dec_val, out_val)

    # 创建数据加载器
    train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    # loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 64, True)

    model = Transformer().cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    criterion = CombinedLoss()
    criterion1 = nn.MSELoss()         # 忽略 占位符 索引为0.
    criterion2 = nn.L1Loss()




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # relative_eigenvalues = relative_eigenvalues.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9,weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)  # T_max 是周期的长度
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=1000, patience=10)
    torch.save(model.state_dict(), 'best_model32_3.pth')
    # 加载最佳模型权重
    model.load_state_dict(torch.load('best_model32_3.pth'))

    # 计算验证集的MSE损失
    best_val_mse_loss = calculate_mse(model, val_loader)
    print(f"Best Val MSE Loss: {best_val_mse_loss:.6f}")


    # # 调用 validate_model 并保存预测值
    validation_loss, all_predictions = validate_model(model, criterion, val_loader)
    all_predictions = all_predictions[:, -20:]
    np.savetxt('predictions_train.txt', all_predictions, fmt='%.6f', delimiter=' ')
    print(all_predictions.shape)
    # np.savetxt('predictions_train.txt', all_predictions, fmt='%.6f', delimiter=' ')

    # 加载真实值数据并进行反向缩放
    data = np.loadtxt('results_all_nonzero.txt')
    print(data.shape)
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    restored_predictions = (all_predictions * (data_max - data_min) + data_min)
    np.savetxt('predictions.txt', restored_predictions, fmt='%.6f', delimiter=' ')

    # 创建用于保存图像的文件夹
    output_dir = 'prediction_plots_32_mse'
    os.makedirs(output_dir, exist_ok=True)

    # 加载验证集的真实值并进行反向缩放
    true_values = (out_val[:, -20:] * (data_max - data_min) + data_min)

    # 绘制和保存图像
    x = np.arange(1, restored_predictions.shape[1] + 1)
    x = np.insert(x, 0, 0)

    for i in range(100):
        extended_predictions = np.insert(restored_predictions[i], 0, 0)
        extended_true_values = np.insert(true_values[i], 0, 0)

        plt.figure()
        plt.scatter(x, extended_true_values, label='Simulation', marker='o')
        plt.plot(x, extended_true_values, linestyle='-', alpha=0.7)
        plt.scatter(x, extended_predictions, label='Prediction', marker='o')
        plt.plot(x, extended_predictions, linestyle='-', alpha=0.7)

        plt.xlabel('Strain')
        plt.ylabel('Stress/MPa')
        plt.legend()

        plt.savefig(os.path.join(output_dir, f'predictions_vs_true_values_{i + 1}.png'))
        plt.close()

