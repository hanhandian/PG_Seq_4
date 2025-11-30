import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, n_dropout, device):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, dropout=n_dropout,
                            batch_first=True)  # utilize the GRU model in torch.nn
        self.linear = nn.Linear(hidden_size, output_size)  # 全连接层
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, _x):
        batch_size, seq_len = _x.shape[0], _x.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # h_0 = torch.randn(self.num_directions * num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        x, _ = self.LSTM(_x, (h_0, c_0))  # _x is input, size (seq_len, batch, input_size)
        x = self.linear(x)
        x = x[:, -1, :]
        return x


class GRU(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, n_dropout, device):
        super(GRU, self).__init__()
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, dropout=n_dropout,
                          batch_first=True)  # utilize the GRU model in torch.nn
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 全连接层
        self.linear = nn.Linear(hidden_size, output_size)  # 全连接层
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, _x):
        batch_size, seq_len = _x.shape[0], _x.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        x, _ = self.GRU(_x, h_0)  # _x is input, size (seq_len, batch, input_size)
        x = self.fc1(x)
        x = self.linear(x)
        x = x[:, -1, :]
        return x


class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            device='cuda:0'
        )

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device='cuda:0')
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:  # 双向BI-LSTM
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        return gru_out, hidden.squeeze(0)


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)

    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, pre_len, seq_len, teacher_forcing=0.3):
        super().__init__()
        self.encoder = RNNEncoder(num_layers, input_size, seq_len, hidden_size)
        self.decoder_cell = AttentionDecoderCell(input_size, output_size, seq_len, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pre_len
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size, output_size)

    def __call__(self, xb, yb=None):
        input_seq = xb  # (b,s,input)(batch,5,8)
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda:0')
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = input_seq[:, -1, :]
        for i in range(self.pred_len):
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i].unsqueeze(1)
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output
        outputs = outputs.permute(1, 0, 2)
        outputs = self.linear(outputs)
        outputs = outputs.squeeze(-1)
        # print(outputs.shape)
        return outputs
