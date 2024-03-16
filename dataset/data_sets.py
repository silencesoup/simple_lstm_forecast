from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

def read_data():
    data = loadmat('data/TrainData1.mat')
    return data['XTrain'], data['YTrain']

def prepare_data(window_size, forecast_length, batch_size):

    input_X = []
    obs_Y = []
    data_x, data_y = read_data()

    for i in range(len(data_x[0]) - window_size - forecast_length + 1):
        input_X.append(data_x[:, i:i + window_size])
        obs_Y.append(data_y[:, i + window_size:i + window_size + forecast_length])

    input_tensor = torch.tensor(np.array(input_X)).float()
    output_tensor = torch.tensor(np.array(obs_Y)).float()

    # 创建TensorDataset和DataLoader
    dataset = TensorDataset(input_tensor, output_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader