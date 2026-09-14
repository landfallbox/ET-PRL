"""
@Author      : landfallbox
@Date        : 2025/11/18 星期二 15:18
@Description : LSTM 模型定义（通用可复用实现）
"""

import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM 神经网络模型

    网络结构：
        - 多层 LSTM 层（层数和每层神经元数量可配置）
        - 一层全连接输出层
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        """
        初始化 LSTM 模型

        参数：
            input_size: 输入特征维度
            hidden_sizes: 每层 LSTM 的隐藏层神经元数量列表，列表长度即为 LSTM 层数
                          例如：[64, 32] 表示 2 层 LSTM，第一层 64 个神经元，第二层 32 个神经元
            output_size: 输出层神经元数量
            batch_first: 是否将 batch 维度放在第一维，默认为 True (batch, seq, feature)
            dropout: LSTM 层之间的 dropout 比例，默认为 0.0（仅当层数 > 1 时有效）
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes)
        self.batch_first = batch_first
        self.dropout = dropout  # 保存用于重建的 dropout 配置

        # 构建多层 LSTM
        self.lstm_layers = nn.ModuleList()

        for i in range(self.num_layers):
            # 确定当前层的输入维度
            current_input_size = input_size if i == 0 else hidden_sizes[i - 1]
            current_hidden_size = hidden_sizes[i]

            # 只在非最后一层且层数大于1时应用 dropout
            current_dropout = dropout if (i < self.num_layers - 1 and self.num_layers > 1) else 0.0

            lstm_layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=current_hidden_size,
                num_layers=1,  # 每个 LSTM 模块只包含 1 层
                batch_first=batch_first,
                dropout=0.0,  # 在单层 LSTM 中不使用内置 dropout
            )
            self.lstm_layers.append(lstm_layer)

            # 如果需要 dropout，在层之间添加
            if current_dropout > 0:
                self.lstm_layers.append(nn.Dropout(current_dropout))

        # 输出层（全连接层）
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置模型参数（使用Xavier初始化）"""
        for layer in self.lstm_layers:
            if isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

        # 初始化全连接层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量
               如果 batch_first=True: shape (batch_size, seq_length, input_size)
               如果 batch_first=False: shape (seq_length, batch_size, input_size)

        返回：
            output: 输出张量，shape (batch_size, output_size)
        """
        # 依次通过每层 LSTM 和 Dropout
        out = x
        for layer in self.lstm_layers:
            if isinstance(layer, nn.LSTM):
                out, _ = layer(out)  # 忽略隐藏状态
            elif isinstance(layer, nn.Dropout):
                out = layer(out)

        # 取最后一个时间步的输出
        if self.batch_first:
            out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        else:
            out = out[-1, :, :]  # shape: (batch_size, hidden_size)

        # 通过全连接层得到最终输出
        out = self.fc(out)  # shape: (batch_size, output_size)

        return out

    def get_model_info(self):
        """
        获取模型信息

        返回：
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            "model_name": "LSTM",
            "num_lstm_layers": self.num_layers,
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        return info
