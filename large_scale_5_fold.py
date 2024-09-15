import torch 
import os
import numpy as np
import argparse
import wandb 
import random
import matplotlib.pyplot as plt

import itertools 


from sklearn.model_selection import StratifiedKFold
from evaluation import leave_two_out
from sklearn.metrics import mean_squared_error




from utils import (experiment, full_dataloaders_5fold, train_val, 
                   initialize_wise_model,initialize_model,
                   prediction,regression_experiment_data,scale_data)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    print(f"Random seed set as {seed}")

        


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train a LSTM  model on fNIRS data')
    
    parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--layers', type=int, default=5, help='Number of layers of basic LSTM')
    parser.add_argument('--hidden_size', type=int, default=6, help='hidden_size of LSTM')
    parser.add_argument('--hidden_size1', type=int, default=44, help='hidden_size of wise LSTM')
    parser.add_argument('--hidden_size2', type=int, default=10, help='hidden_size of second LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--subject', type=int, default=0, help='subject')
    parser.add_argument('--window_length', type=int, default=10, help='Number of samples to select from each stimuli response')
    parser.add_argument('--delay', type=int, default=48, help='Delay in the stimuli response')
    parser.add_argument('--word_vec_dim', type=int, default=50, help='Dimension of the word vectors')
    parser.add_argument('--sections', type=int, default=1, help='Number of sections to divide the stimuli response')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for the window')
    parser.add_argument('--modality', type=str, default='avg_window', help='Modality of the data')
    parser.add_argument('--dropout',type=float,default=0.25,help= 'dropout in LSTM' )
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='L2 regularization')
    parser.add_argument('--model', type=str, default='LSTM', help='Model to use')
    parser.add_argument('--norm',type=float,default=0.0,help= 'Control prediction norm')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use')
    parser.add_argument('--nonlinearity', type=str, default='tanh', help='activation function for RNN')
    parser.add_argument('--scale', type=str, default='normalization', help='Scaling method for the data')
    parser.add_argument('--architecture', type=str, default='wise', help='Choose  LSTM architecture for regression')
    
    
    return parser.parse_args()


def main(args):
    
    set_seed()
    
    
    categories_ = {
    "ca1": [0, 10, 15, 16, 35],
    "ca2": [1, 23, 44, 45, 47],
    "ca3": [2, 3, 24, 29, 48],
    "ca4": [4, 9, 12, 25, 49],
    "ca5": [5, 30, 31, 32, 34],
    "ca6": [6, 7, 8, 27, 28],
    "ca7": [11, 17, 19, 36, 37],
    "ca8": [13, 20, 21, 22, 39],
    "ca9": [14, 18, 33, 38, 46],
    "ca10": [26, 40, 41, 42, 43]
    }
    
    ###Wandb####
    
    wdb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="LSTM_finetuning",
        name = f"model_{args.model}_hs_{args.hidden_size}_hs_2_{args.hidden_size2}_lr_{args.learning_rate}_dropout1_{args.dropout}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "delay": args.delay,
        "window_length": args.window_length,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "norm": args.norm,
        "layer": args.layers,
        }
        )


    #Hyperparameters 
    #############################
    
    args= parse_args()
    data_path = os.path.join(args.data_path ,'data/preprocessed')
    
    
        
    #fNIRS Parameters
    
    
    wind_length = args.window_length
    delay = args.delay
    word_vec_dim = args.word_vec_dim
    sections= args.sections
    step_size= args.step_size
    subject = args.subject
    modality= args.modality
    weight_decay = args.weight_decay
    model = args.model
    norm = args.norm
    loss = args.loss
    nonlinearity = args.nonlinearity
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_layers = args.layers
    dropout = args.dropout
    
    print("subject number is : ", subject)
    

    
    #Model parameters  

    learning_rate = args.learning_rate
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    epochs = args.epochs
    
    architecture = args.architecture
    
    #Scaling data
    mode = args.scale
    
    #Leave 2 out cross validation
    
    p= 5
    
    #Normalize data
    
    

    #Preparing data for the experiment

    data,stimuli_embedding = regression_experiment_data(data_path,subject,sections,delay,wind_length,step_size,word_vec_dim)
    

    X_train, Y_train,Y_int_labels = experiment(data,stimuli_embedding.numpy(),modality,categories= False)
    
    X_train,Y_train = scale_data(X_train,Y_train,mode)
    

   
    #Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Metrics
    correct =0
    within_total = 0
    within_correct = 0
    outside_total = 0
    outside_correct = 0

    total_count = 0
    
    mse_list =[]

        
    #leave two out cross validation
    kf = StratifiedKFold(n_splits = p,shuffle = True, random_state=42)

    run = 0
    
    for train_index, test_index in kf.split(X_train,Y_int_labels):
        
        run+=1
        
        
        x_train,x_test = X_train[train_index], X_train[test_index]
        
        
        y_train,y_test = np.array(Y_train[train_index]),np.array(Y_train[test_index])
        
        val_size = int(0.10*len(x_train))
        
        #print("val_size:", val_size)
        val_index = np.random.choice(len(x_train),val_size,replace=False)
        
        x_val,y_val = x_train[val_index],y_train[val_index]
        
        
        x_train = np.delete(x_train, val_index, axis=0)
        y_train = np.delete(y_train, val_index, axis=0)


        y_int_test = np.array(Y_int_labels[test_index])
        
        
        
        
        #####Train LSTM model #######
        
        #model 
        
        if architecture == 'wise':
            
            net = initialize_wise_model(model,device, input_size=22, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                                        nonlinearity = nonlinearity,output_size=word_vec_dim)
            
            
            save_model = "finetuning_models_large_scale/model" + f"_{model}_hid1_{hidden_size1}_hid2_{hidden_size2}_lr_{learning_rate}_wd_{weight_decay}_window__{wind_length}_d_{delay}_subject_{subject}_norm{norm}_mode{mode}.pt"
            
        elif architecture == 'basic':
            
            net = initialize_model(model,device, input_size=22, hidden_size=hidden_size, num_layers=num_layers, 
                                       nonlinearity = nonlinearity,output_size=word_vec_dim,dropout=dropout)
            
            save_model = "finetuning_models_large_scale/model" + f"_{model}_Lyr_{num_layers}_hs_{hidden_size}_lr_{learning_rate}_d_{dropout}_wd_{weight_decay}_wl_{wind_length}_d_{delay}_subject_{subject}_norm{norm}_mode_{mode}.pt"
        
        net = net.to(device)      
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss().to(device)
        
        train_loader,val_loader,test_loader  = full_dataloaders_5fold(x_train,y_train,x_test,y_test,x_val,y_val,batch_size)
        
        train_val(train_loader,val_loader,device, optimizer, criterion, epochs,net, loss = loss,file=save_model,norm= norm)
        
        state_dict = torch.load(save_model)

        # Load the trained  state dict  into the model
        net.load_state_dict(state_dict)
        
        #prediction 
        
        predictions = prediction(net,test_loader,device)
        
        
        
        predictions = predictions.cpu().detach().numpy()
        
        #print('predicitons mean', np.mean(predictions[0]))
        #print('True mean', np.mean(y_test[0]))
        
        
        #random_int = random.randint(1, len(predictions))
        
        #plt.figure()
        
        #plt.plot(predictions[random_int], label='Predicted')
        #plt.plot(y_test[random_int], label='True')
        #plt.legend()
        #plt.savefig("my_plot.png")
        #wdb_logger.log({"predictions":wandb.Image("my_plot.png")})
    
        
        numbers =list(range(len(predictions)))
        
        pairs = list(itertools.combinations(numbers,2))
        
        
        
        for i in pairs: 
            
            pair_index = np.asarray(i)
            pair_int_labels = y_int_test[pair_index]
            
            
            if len(pair_int_labels)==len(set(pair_int_labels)):
                
                pair_predictions = predictions[pair_index]
                pair_labels = y_test[pair_index]
                
               
                predict_pair =[]
                label_pair = []
                
               
                
             
            
                for k in range(2):
                    predict_pair.append(pair_predictions[k].reshape(1,-1))
                    label_pair.append(pair_labels[k].reshape(1,-1))
                    
                    mse = mean_squared_error(predict_pair[k], label_pair[k])
            
                    mse_list.append(mse)
                    
                
                correct+= leave_two_out(predict_pair,label_pair) 
                total_count+=1 
                    
                if all (not set(pair_int_labels).issubset(category_list) for category_list in categories_.values()):
                    
                    
                    #print('outside pair:',pair_int_labels)
                    
                    
                    
                    outside_total+=1
                    outside_correct += leave_two_out(predict_pair, label_pair)
                    
                for category_list in categories_.values(): 
                    
                    if set(pair_int_labels).issubset(set(category_list)):
                        
                    
                        
                        #print('within pair:',pair_int_labels)
                        
                
                        within_total+=1
                        within_correct+=leave_two_out(predict_pair,label_pair)
                        
               
                                
                    
                        
            
        
                
        os.remove(save_model)
                
                
    accuracy = round(1.0 * correct / total_count, 2 )
    total_mse = round(sum(mse_list)/len(mse_list),2)
    within_accuracy = round(1.0 * within_correct/within_total, 2)
    outside_accuracy = round(1.0 * outside_correct/outside_total, 2)
    
    
    
    
    wdb_logger.log({"val/matching_score":accuracy,
                    "val/mse":total_mse,
                    "val/within_acc": within_accuracy, 
                    "val/out_acc":outside_accuracy
                
                    })
    
    
    
    wdb_logger.finish()
        

        
 
if __name__=='__main__':
    parser = parse_args()
    main(parser)