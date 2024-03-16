import torch.nn as nn
import torch.optim as optim
from models.simple_lstm import SimpleLSTMForecast
from dataset.data_sets import prepare_data

import sys , os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def test_train_model():

    window_size = 15
    forecast_length = 5
    batch_size = 128
    input_size = 54
    output_size = 10

    model = SimpleLSTMForecast(
        input_size=input_size,
        output_size=output_size,
        hidden_size=batch_size,
        forecast_length=forecast_length
    )

    dataset, dataloader = prepare_data(window_size=window_size, forecast_length=forecast_length, batch_size=batch_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 假设我们使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    # 设置训练的轮数
    num_epochs = 100

    # 开始训练
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.transpose(1, 2).transpose(0, 1)
            targets = targets.transpose(1, 2).transpose(0, 1)
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
        
        # 打印每轮的平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    print('Training finished.')

test_train_model()