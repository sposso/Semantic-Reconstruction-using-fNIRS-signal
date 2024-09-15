
import torch
import torch.nn as nn
import torch.nn.functional as F 



class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=50,dropout=0.5):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,nonlinearity, output_size=50, dropout=0.5):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity=nonlinearity, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,nonlinearity, output_size=50, dropout=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity=nonlinearity,bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=50,dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=50,dropout=0.5):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)  # 2 for bidirection

    
    def attention(self, lstm_output):
        
        E = torch.bmm(lstm_output, lstm_output.transpose(1,2))
        A = F.softmax(E, dim=2)
        C = torch.bmm(A, lstm_output)
        
        return C
            
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        
        weighted_out = self.attention(out)
        
        att_out = self.fc(weighted_out[:, -1, :])
        return att_out
    
    
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(x.device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size1).to(x.device) 
        out, _ = self.lstm1(x, (h0, c0))
        
        h1 = torch.zeros(1, x.size(0), self.hidden_size2).to(x.device) 
        c1 = torch.zeros(1, x.size(0), self.hidden_size2).to(x.device) 
        out, _ = self.lstm2(out, (h1, c1))
        
        out = self.fc(out[:, -1, :])
        return out
    
    
class BiCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BiCustomLSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm1 = nn.LSTM(input_size, hidden_size1,bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1*2, hidden_size2,bidirectional= True, batch_first=True)
        self.fc = nn.Linear(hidden_size2*2, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size1).to(x.device) 
        c0 = torch.zeros(2, x.size(0), self.hidden_size1).to(x.device) 
        out, _ = self.lstm1(x, (h0, c0))
        
        h1 = torch.zeros(2, x.size(0), self.hidden_size2).to(x.device) 
        c1 = torch.zeros(2, x.size(0), self.hidden_size2).to(x.device) 
        out, _ = self.lstm2(out, (h1, c1))
        
        out = self.fc(out[:, -1, :])
        return out
    
    
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 nonlinearity):
        super(CustomRNN, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.rnn1 = nn.RNN(input_size, hidden_size1, batch_first=True,nonlinearity=nonlinearity)
        self.rnn2 = nn.RNN(hidden_size1, hidden_size2, batch_first=True,nonlinearity=nonlinearity)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(x.device) 
        out, _ = self.rnn1(x, h0)
        
        h1 = torch.zeros(1, x.size(0), self.hidden_size2).to(x.device)  
        out, _ = self.rnn2(out,h1)
        
        out = self.fc(out[:, -1, :])
        return out
    
class BiCustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 nonlinearity):
        super(BiCustomRNN, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.rnn1 = nn.RNN(input_size, hidden_size1, batch_first=True,bidirectional= True,nonlinearity=nonlinearity)
        self.rnn2 = nn.RNN(hidden_size1*2, hidden_size2, batch_first=True,bidirectional= True,nonlinearity=nonlinearity)
        self.fc = nn.Linear(hidden_size2*2, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size1).to(x.device) 
        out, _ = self.rnn1(x, h0)
        
        h1 = torch.zeros(2, x.size(0), self.hidden_size2).to(x.device)  
        out, _ = self.rnn2(out,h1)
        
        out = self.fc(out[:, -1, :])
        return out

    
############################################CNN######################################################
    
    
class DWSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DWSConv, self).__init__()
        self.depth_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels)
        self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.point_conv(x)
        return x
    
    



class regression_fNIRSNet(torch.nn.Module):
    """
    fNIRSNet model

    Args:
        num_class: Number of classes.
        DHRConv_width: Width of DHRConv = width of fNIRS signals.
        DWConv_height: Height of DWConv = height of 2 * fNIRS channels, and '2' means HbO and HbR.
        num_DHRConv: Number of channels for DHRConv.
        num_DWConv: number of channels for DWConv.
    """
    def __init__(self, out=50, DHRConv_width=10, DWConv_height=22, num_DHRConv=4, num_DWConv=8):
        super(regression_fNIRSNet, self).__init__()
        # DHR Module
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=num_DHRConv, kernel_size=(1, DHRConv_width), stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(num_DHRConv)

        # Global Module
        self.conv2 = DWSConv(in_channels=num_DHRConv, out_channels=num_DWConv, kernel_size=(DWConv_height, 1))
        self.bn2 = torch.nn.BatchNorm2d(num_DWConv)

        self.fc = torch.nn.Linear(num_DWConv, out)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x