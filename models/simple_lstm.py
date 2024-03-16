import torch.nn as nn
from torch.nn import functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(SimpleLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        return self.linearOut(out_lstm)
    
class SimpleLSTMForecast(SimpleLSTM):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(SimpleLSTMForecast, self).__init__(input_size, output_size, hidden_size, dr)
        self.forecast_length = forecast_length
        self.output_size = output_size

    def forward(self, x):
        # 调用父类的forward方法获取完整的输出
        full_output = super(SimpleLSTMForecast, self).forward(x)

        # 重塑输出以匹配目标形状
        batch_size = full_output.shape[1]
        forecast_output = full_output[-self.forecast_length:, :, :]
        # forecast_output = forecast_output.view(self.forecast_length, batch_size, self.output_size)
        return forecast_output