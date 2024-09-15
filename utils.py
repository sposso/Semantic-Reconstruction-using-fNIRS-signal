import numpy as np 
import pandas as pd 
import os
import torchtext 
import torch 
import torch.nn as nn


from deep_models import (LSTMRegressor,BiLSTM,regression_fNIRSNet,RNN,CustomLSTM,
                         CustomRNN,BiRNN,BiCustomLSTM,BiCustomRNN,BiLSTM_Attention)

from torch.optim.lr_scheduler import ReduceLROnPlateau

class Regression_Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        
        #self.feature = torch.tensor(self.feature, dtype=torch.float)
        #self.label = torch.tensor(self.label, dtype=torch.float)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        return self.feature[item], self.label[item]

def conditions_to_categories(labels):
    
    ca1 = [0, 10, 15, 16, 35]
    ca2 = [1, 23, 44, 45, 47]
    ca3 = [2, 3, 24, 29, 48]
    ca4 = [4, 9, 12, 25, 49]
    ca5 = [5, 30, 31, 32, 34]
    ca6 = [6, 7, 8, 27, 28]
    ca7 = [11, 17, 19, 36, 37]
    ca8 = [13, 20, 21, 22, 39]
    ca9 = [14, 18, 33, 38, 46]
    ca10 = [26, 40, 41, 42, 43]
    
    ca_labels = [ca1, ca2, ca3,ca4,ca5,ca6,ca7,ca8,ca9,ca10]
    
    for idx,label in enumerate(labels):
        for i, ca in enumerate(ca_labels):
            if label in ca:
                labels[idx] = i 
                
    return labels


def experiment(data,stimuli_embedding,name,categories= True):
    
    #input :data -> size [n_channels, n_conditions, n_trials,sections, n_timepoints]
    #                    [22,50,7,8,16] 
    
    
    if name == 'avg_window':
        
        avg_window = np.mean(data, axis=3)
        
        labels = []

        d =[]
        int_labels=[]
    

        for i in range(avg_window.shape[1]):
            for j in range(avg_window.shape[2]):
                
                d.append(avg_window[:,i,j])
                
                stimuli_vec= stimuli_embedding[i] 
                
                labels.append(stimuli_vec)
                int_labels.append(i)
            
                
                
        d= np.array(d)

        labels = np.array(labels)
        int_labels = np.array(int_labels)
        
        if categories:
            
            int_labels = conditions_to_categories(int_labels)  
            
        return d, labels,int_labels 
    
    elif name == 'avg_trial':
        
        avg_trial = np.mean(data, axis=2)
        
        labels = []
        d =[]
        int_labels=[]

        for i in range(avg_trial.shape[1]):
            for j in range(avg_trial.shape[2]):
                
                d.append(avg_trial[:,i,j]) 
                stimuli_vec= stimuli_embedding[i]
                
                labels.append(stimuli_vec)
                int_labels.append(i)
                
                
        d= np.array(d)
        labels = np.array(labels)
        int_labels = np.array(int_labels)
        
        if categories:
            
            int_labels = conditions_to_categories(int_labels)

        
        return  d, labels,int_labels
    
    
    elif name== 'no_avg':
        
        labels = []
        d =[]
        int_labels =[]

        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    
                
                    
                    d.append(data[:,i,j,k,:])
                    stimuli_vec= stimuli_embedding[i]
                    labels.append(stimuli_vec)
                    
                    int_labels.append(i)
            
        d= np.array(d)
        labels = np.array(labels)
        int_labels = np.array(int_labels)
        
        if categories:
                
            labels = conditions_to_categories(labels)
            
        return d, labels,int_labels
    
def regression_experiment_data(dir_path,subject,sections,delay,wind_lenght,step_size,word_vec_dim):
    
    """
    This function loads the fNIRS data and the word embedding vectors of the stimuli for a given subject.

    Parameters:
    dir_path (str): The directory path where the data is located.
    select_scan_num (int): The number of samples to select from each stimuli response.
    delay (int): It has been reported that the hemodynamic response peaks 6 seconds after the neurons's immediate 
    activation in a region.Therefore, this parameters depends on the sampling frequency of the fNIRS device.
    
    subject (int): The index of the subject in the 'folders' list.
    word_vec_dim (int): The dimension of the word embedding vectors.

    Returns:
    chn =   stimulus responses along the 22 channels during 7 trials. 
    chnavg = average stimulus responses along the 22 channels.
    stimuli_embedding = word embedding vectors of the stimuli.
    """
    
    folders = ['4001', '4002', '4003', '4004', '4005', '4006', '4007']
    
    folder = folders[subject]
    
    channels = 22
    trial=7
    
    end = (sections-1)*step_size + delay 
    
    
    stimuli = ['lettuce','T-shirt', 'cat', 'cow', 'tower', 
             'eye', 'sofa', 'chair', 'table', 'pyramid', 
             'tomato', 'ant', 'pool', 'chopsticks', 
             'hammer', 'corn', 'carrot', 'housefly', 'saw', 
             'butterfly', 'pan', 'spoon', 'glass', 'jeans', 'dog', 
             'stadium', 'bicycle', 'bed', 'bookcase', 'horse',
             'arm', 'foot', 'palm', 'scissors', 'leg',
             'celery', 'dragonfly','bee', 'pliers', 'knife',
             'car', 'train', 'aircraft', 'truck','sweater',
             'skirt', 'screwdriver', 'dress','panda','tiananmen']
    
    vec = torchtext.vocab.GloVe(name='6B',dim= word_vec_dim)
    stimuli_embedding = vec.get_vecs_by_tokens(stimuli,lower_case_backup=True)
             
    
    
    #vec = torchtext.vocab.GloVe(name='6B',dim= word_vec_dim)
    #stimuli_embedding = vec.get_vecs_by_tokens(stimuli,lower_case_backup=True)
    

    data = np.zeros((channels,len(stimuli),trial,sections,wind_lenght))
    #chn= np.zeros((len(stimuli), channels,trial, select_scan_num))
    #chnAvg = np.zeros((len(stimuli), channels, select_scan_num))
    
    individual= []
    individual_label =[]
    
    for ch in range(0,channels):
        
        
        df1 = pd.read_csv(os.path.join(dir_path, str(folder) + '_1', 
                                        str(folder) + '_1_' + str(ch + 1) +'.txt'), sep='\t', header=0,
                            dtype={'value': np.float64})
        df2 = pd.read_csv(os.path.join(dir_path, str(folder) + '_2', 
                                        str(folder) + '_2_' + str(ch + 1) +'.txt'), sep='\t', header=0,
                            dtype={'value': np.float64})
        frames = [df1, df2]
        df = pd.concat(frames)
        df.fillna(0)
        value = df[['value']].to_numpy()
        
        #Empty dict
        conditions ={i: 0 for i in range(1,len(stimuli)+1)}
        
        #Fill dict with index of each condition
        for i in conditions.keys():
            
            conditions[i] =  df[df['condition'] == i].index.tolist()
            
            
        for c in conditions.keys():
            
            if len(conditions[c])==6:
              
                conditions[c].append(conditions[c][4])
                
              
            
            
        # creating  vectors from fnirs data according to the  condition and delay   
        for idx,c  in enumerate(conditions.values()):
            
            for tr, i in enumerate(c):
                
             
                start = delay 
                run = 0
                
                while(start<= end):
                    
                    individual.append(value[i+start:i+start+wind_lenght].squeeze())
                    individual_label.append(idx)
                    
                    data[ch,idx,tr,run]= value[i+start:i+start+wind_lenght].squeeze()
                    start+=step_size
                    run+=1
           
                    
            

    
    return data,stimuli_embedding 

def pilot_regression_experiment_data(dir_path,subject,sections,delay,wind_lenght,step_size,word_vec_dim):
    
    """
    This function loads the fNIRS data and the word embedding vectors of the stimuli for a given subject.

    Parameters:
    dir_path (str): The directory path where the data is located.
    select_scan_num (int): The number of samples to select from each stimuli response.
    delay (int): It has been reported that the hemodynamic response peaks 6 seconds after the neurons's immediate 
    activation in a region.Therefore, this parameters depends on the sampling frequency of the fNIRS device.
    
    subject (int): The index of the subject in the 'folders' list.
    word_vec_dim (int): The dimension of the word embedding vectors.

    Returns:
    chn =   stimulus responses along the 22 channels during 7 trials. 
    chnavg = average stimulus responses along the 22 channels.
    stimuli_embedding = word embedding vectors of the stimuli.
    """
    
    folders = [1001, 1003, 1004, 1005]
    
    folder = folders[subject]
    
    channels = 46
    trial=12
    
    end = (sections-1)*step_size + delay 
    
    
    stimuli = ['bunny', 'kitty', 'bear', 'dog', 'hand', 'mouth', 'nose', 'foot']
    
    vec = torchtext.vocab.GloVe(name='6B',dim= word_vec_dim)
    stimuli_embedding = vec.get_vecs_by_tokens(stimuli,lower_case_backup=True)
             
    
    
    #vec = torchtext.vocab.GloVe(name='6B',dim= word_vec_dim)
    #stimuli_embedding = vec.get_vecs_by_tokens(stimuli,lower_case_backup=True)
    

    data = np.zeros((channels,len(stimuli),trial,sections,wind_lenght))
    #chn= np.zeros((len(stimuli), channels,trial, select_scan_num))
    #chnAvg = np.zeros((len(stimuli), channels, select_scan_num))
    
    individual= []
    individual_label =[]
    
    for ch in range(0,channels):
        
        
        df = pd.read_csv(os.path.join(dir_path, str(folder), 
                                            str(folder) + '_' + str(ch + 1) +'.txt'), sep='\t', header=0,
                             dtype={'value': np.float64})
        
        
        df= df.fillna(0)
        value = df[['value']].to_numpy()
        
        #Empty dict
        conditions ={i: 0 for i in [110,120,130,140,150,160,170,180]}
        
        #Fill dict with index of each condition
        for i in conditions.keys():
            
            conditions[i] =  df[df['condition'] == i].index.tolist()
            
                
            
            
        # creating  vectors from fnirs data according to the  condition and delay   
        for idx,c  in enumerate(conditions.values()):
            
            for tr, i in enumerate(c):
                
             
                start = delay 
                run = 0
                
                while(start<= end):
                    
                
                            
                    individual.append(value[i+start:i+start+wind_lenght].squeeze())
                    individual_label.append(idx)
                    
                    data[ch,idx,tr,run]= value[i+start:i+start+wind_lenght].squeeze()
                    start+=step_size
                    run+=1
                    
            
           
                    
            

    
    return data,stimuli_embedding

def initialize_model(model,device, input_size, hidden_size, num_layers,nonlinearity, output_size,dropout):
    
    if model == "LSTM":
        net = LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,dropout=dropout).to(device)
        return net
        
    elif model == "BiLSTM":
        net = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,dropout=dropout).to(device)
        return net
        
    elif model == "CNN":
        net = regression_fNIRSNet(out=output_size,DHRConv_width=hidden_size,DWConv_height=num_layers).to(device)
        
    elif model == "RNN":
        net = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,nonlinearity= nonlinearity,
                  output_size=output_size,dropout=dropout).to(device)
        return net
        
    elif model == "BiRNN":
        
        net = BiRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,nonlinearity= nonlinearity,
                  output_size=output_size,dropout=dropout).to(device)
        return net
    elif model == "BiLSTM_Attention":
        
        net = BiLSTM_Attention(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,dropout=dropout).to(device)
        
        return net
        
   

def initialize_wise_model(model,device, input_size, hidden_size1,hidden_size2,nonlinearity,
                          output_size):
    
    
    
    if model == "LSTM":
        net = CustomLSTM(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, 
                         output_size=output_size).to(device)
        
        return net
        
    elif model == "RNN":
        net = CustomRNN(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2,nonlinearity= nonlinearity,
                  output_size=output_size).to(device)
        return net
        
    elif model == "BiLSTM":
        net = BiCustomLSTM(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, 
                         output_size=output_size).to(device)
        return net
        
    elif model == "BiRNN":
        net = BiCustomRNN(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2,nonlinearity= nonlinearity,
                  output_size=output_size).to(device)
        return net
    
    

def dataloaders(x_train,y_train,x_test,y_test):
        
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
    
        
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
    
        x_test = x_test.permute(0,2,1)
        x_train = x_train.permute(0,2,1)
        
        
        train_set = Regression_Dataset(x_train, y_train)
        test_set = Regression_Dataset(x_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=x_train.shape[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=x_test.shape[0], shuffle=True)
        
        return train_loader,test_loader 
 
 
def data_load(x_train,y_train):
    
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        
        
        x_train = x_train.permute(0,2,1)
        
        train_set = Regression_Dataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=x_train.shape[0], shuffle=False)
        
        return train_loader   
    

def full_dataloaders(x_train,y_train,x_test,y_test,x_val,y_val):
        
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        
    
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        
        
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).float()
            
    
        x_test = x_test.permute(0,2,1)
        x_train = x_train.permute(0,2,1)
        x_val = x_val.permute(0,2,1)

        
        train_set = Regression_Dataset(x_train, y_train)
        val_set = Regression_Dataset(x_val, y_val)
        test_set = Regression_Dataset(x_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=x_train.shape[0], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=x_val.shape[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=x_test.shape[0], shuffle=False)
        
        return train_loader,val_loader,test_loader
    
    
def scale_data(x,y,mode):
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    
    if mode == 'normalization':
        
        #print('You are using Normalization technique to scale your data')
    
    
        
        x = nn.functional.normalize(x, p=2, dim=2)
        y = nn.functional.normalize(y, p=2, dim=1)
        
        x= x.numpy()
        y= y.numpy()
        
    elif mode == 'standardization':
        
        print('You are using Standardization technique to scale your data')
    
        x_mean = torch.mean(x, dim=2, keepdim=True)
        x_std = torch.std(x, dim=2, keepdim=True)
        
        x = (x - x_mean) / x_std
        
        y_mean = torch.mean(y, dim=1, keepdim=True)
        y_std = torch.std(y, dim=1, keepdim=True)
        
        y = (y - y_mean) / y_std
        
        x = x.numpy()
        y = y.numpy()
        
  
        
    return x,y
    
    
    
def full_dataloaders_5fold(x_train,y_train,x_test,y_test,x_val,y_val,batch_size):
        
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        
      
 
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        
       
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).float()
        
    
        x_test = x_test.permute(0,2,1)
        x_train = x_train.permute(0,2,1)
        x_val = x_val.permute(0,2,1)

        
        train_set = Regression_Dataset(x_train, y_train)
        val_set = Regression_Dataset(x_val, y_val)
        test_set = Regression_Dataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=x_val.shape[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=x_test.shape[0], shuffle=False)
        
        return train_loader,val_loader,test_loader

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,model_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,model_path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

def validation(net,test_loader,device,criterion,loss):
    
    #Model evaluation
    net.eval()
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            target = torch.ones(labels.shape[0]).to(device)
            if loss == 'cosine':
                valid_loss = criterion(output, labels,target)
                
            else:
            
                valid_loss = criterion(output, labels)
      
            
    return valid_loss

def prediction(net,test_loader,device):
    
    #Model evaluation
    net.eval()
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            
    return output



def train_val(train_loader,test_loader, device, optimizer, criterion, epochs,net, loss,file,norm):
    
    if loss == 'MSE':
        criterion = torch.nn.MSELoss()
        
    elif loss == 'cosine': 
        criterion = torch.nn.CosineEmbeddingLoss()
        
        
    cos = torch.nn.CosineEmbeddingLoss()
    early_stopping = EarlyStopping(patience=20, verbose=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(epochs):
            net.train()
            loss_steps = []
            for data in train_loader:
                inputs, labels = data
                targets = torch.ones(labels.shape[0]).to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                output = net(inputs)
            
                output = torch.nn.functional.normalize(output, p=2, dim=1)
                
                #norm_ = torch.linalg.norm(output,ord=2,dim=1)
                #print("output norm", norm_)
                if loss == 'MSE':

                    
                    cov = (1/output.shape[0])*torch.matmul(output.T,output)
                    cov = cov/torch.trace(cov)
                    fro_norm = torch.linalg.matrix_norm(cov)
                    H = -2*torch.log(fro_norm)
                    
                    #print("Entropy: ", H)
                    
                    #print("MSE ", criterion(output,labels))
                
                    train_loss = criterion(output,labels) - norm*H
                    
                    
                elif loss == "cosine":
                    
                    cov = (1/output.shape[0])*torch.matmul(output.T,output)
                    cov = cov/torch.trace(cov)
                    fro_norm = torch.linalg.matrix_norm(cov)
                    H = -1*torch.log(fro_norm**2)
                    
                    print("Entropy: ", H)
                    print("Cosine Loss ", criterion(output,labels,targets))
                    train_loss = criterion(output,labels,targets)
                    
                loss_steps.append(train_loss.item())
                optimizer.zero_grad()
                
                train_loss.backward()
                
                optimizer.step()
                #lrStep.step()
                
            
                
            train_running_loss = float(np.mean(loss_steps))
            #print('[%d] Train loss: %0.5f' % ( epoch, train_running_loss))
                
            valid_loss = validation(net,test_loader,device,criterion,loss)
            
            scheduler.step(valid_loss)
                
            early_stopping(valid_loss,net,file)
            
            
                
            if early_stopping.early_stop:
                
                #print("Best validation loss achived at epoch : ",epoch)
                #print("Early stopping")
                #last_lr = scheduler.get_last_lr()
                #print("Last learning rate: ", last_lr)
                break       
            
    
            
            
