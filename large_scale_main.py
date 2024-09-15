import torch 
import os
import numpy as np
import argparse
import wandb 
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import LeavePOut 
from evaluation import leave_two_out
from sklearn.metrics import mean_squared_error



from utils import (experiment, full_dataloaders, train_val, initialize_model,
                   initialize_wise_model,prediction,regression_experiment_data,
                   scale_data)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    #print(f"Random seed set as {seed}")

        


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train a LSTM  model on fNIRS data')
    
    parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val_size', type=int, default=5, help='validation samples to stop model from training')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers of LSTM')
    parser.add_argument('--hidden_size', type=int, default=5, help='hidden_size of LSTM')
    parser.add_argument('--hidden_size1', type=int, default=44, help='hidden_size of wise LSTM')
    parser.add_argument('--hidden_size2', type=int, default=10, help='hidden_size of second wise LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--subject', type=int, default=1, help='subject')
    parser.add_argument('--window_length', type=int, default=10, help='Number of samples to select from each stimuli response')
    parser.add_argument('--delay', type=int, default=48, help='Delay in the stimuli response')
    parser.add_argument('--word_vec_dim', type=int, default=50, help='Dimension of the word vectors')
    parser.add_argument('--sections', type=int, default=1, help='Number of sections to divide the stimuli response')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for the window')
    parser.add_argument('--modality', type=str, default='avg_trial', help='Modality of the data')
    parser.add_argument('--dropout',type=float,default=0.0,help= 'dropout in LSTM' ) 
    parser.add_argument('--weight_decay', type=float, default=0.05, help='L2 regularization')
    parser.add_argument('--model', type=str, default='BiRNN', help='Model to use')
    parser.add_argument('--norm',type=float,default=0.5,help= 'Control prediction norm')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use')
    parser.add_argument('--nonlinearity', type=str, default='tanh', help='activation function for RNN')
    parser.add_argument('--scale', type=str, default='normalization', help='Scaling method for the data')
    parser.add_argument('--architecture', type=str, default='basic', help='Choose  LSTM architecture for regression')
    parser.add_argument('--save_logs', type=str, default='/mnt/gpfs2_4m/scratch/spo230', help='Path to save logs')
    
    
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
        name = f"model_{args.layers}_hidden_{args.hidden_size}_lr_{args.learning_rate}_dropout_{args.dropout}",
        # track hyperparameters and run metadata
        dir = args.save_logs,
        config={
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout":  args.dropout,
        "delay": args.delay,
        "window_length": args.window_length,
        'subject': args.subject
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
    dropout =args.dropout
    val_size = args.val_size
    weight_decay = args.weight_decay
    model = args.model
    norm = args.norm
    loss = args.loss
    nonlinearity = args.nonlinearity

    
    #Model parameters  

    learning_rate = args.learning_rate
    hidden_size = args.hidden_size
    num_layers = args.layers
    epochs = args.epochs
    
    ##Wise architecture
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    
    architecture = args.architecture
    
    #Scaling data
    mode = args.scale
   

    #Training parameters 
    
    #Leave 2 out cross validation
    
    p= 2

    #Preparing data for the experiment

    data,stimuli_embedding = regression_experiment_data(data_path,subject,sections,delay,wind_length,step_size,word_vec_dim)
    

    X_train, Y_train,Y_int_labels = experiment(data,stimuli_embedding.numpy(),modality,categories= False)
    
    #Scaling data
    
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
    lpo = LeavePOut(p)

    run = 0
    
    

    for train_index, test_index in lpo.split(X_train):   
        
        set_seed()
        
        run+=1
        
        
        predict_list = []
        label_list = []
        
        x_train = X_train[train_index]
        x_test = X_train[test_index]
        
        y_train = np.array(Y_train[train_index])
        y_test = np.array(Y_train[test_index])
        
        val_index = np.random.choice(range(len(x_train)),val_size,replace= False)
        x_val,y_val = x_train[val_index],y_train[val_index]
        
    
        x_train = np.delete(x_train, val_index, axis=0)
        y_train = np.delete(y_train, val_index, axis=0)
        
        y_int_test = np.array(Y_int_labels[test_index])
        
        #####Train LSTM model #######
        
        #model 
        
        if architecture == 'wise':
            
            net = initialize_wise_model(model,device, input_size=22, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                                nonlinearity = nonlinearity,output_size=word_vec_dim)
        
        elif architecture == 'basic':
            
            net = initialize_model(model,device, input_size=22, hidden_size=hidden_size, num_layers=num_layers, 
                                    nonlinearity = nonlinearity,output_size=word_vec_dim,dropout=dropout)
            
        net = net.to(device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss().to(device)
        
        train_loader,val_loader,test_loader  = full_dataloaders(x_train,y_train,x_test,y_test,x_val,y_val)
        
        if architecture == 'wise':
            
            save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_hd1_{hidden_size1}_hd2_{hidden_size2}_lr_{learning_rate}_wd_{weight_decay}_ws_{wind_length}_d_{delay}_subject_{subject}_norm{norm}_scale_{mode}.pt"
        
        elif architecture == 'basic':
            
            save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_Ly_{num_layers}_hs_{hidden_size}_lr_{learning_rate}_d_{dropout}_wd_{weight_decay}_window__{wind_length}_d_{delay}_subject_{subject}_scale_{mode}.pt"

        train_val(train_loader,val_loader,device, optimizer, criterion, epochs,net, loss = loss,file=save_model,norm= norm)
        
        state_dict = torch.load(save_model)

        # Load the trained  state dict  into the model
        net.load_state_dict(state_dict)
        
        #prediction 
        
        predictions = prediction(net,test_loader,device)
    
        predictions = predictions.cpu().detach().numpy()
        
        #print('predicitons mean', np.mean(predictions[0]))
        #print('True mean', np.mean(y_test[0]))
        
      
        
        #plt.figure()
        
        #plt.plot(predictions[0], label='Predicted')
        #plt.plot(y_test[0], label='True')
        #plt.legend()
        #plt.savefig("my_plot.png")
                
        
       

        for k in range(len(x_test)):
            
            predict_list.append(predictions[k].reshape(1, -1))
            label_list.append(y_test[k].reshape(1, -1))
            
            mse = mean_squared_error(predict_list[k],label_list[k])
            
            mse_list.append(mse)
            
            print('mse: ',mse)
            
            
        

        
        correct+= leave_two_out(predict_list,label_list)
        
        total_count+=1

        
        #Evaluate perfomance of the model within  and outside condition
        
        if all (not set(y_int_test).issubset(category_list) for category_list in categories_.values()):
            
            outside_total+=1
            outside_correct += leave_two_out(predict_list, label_list)
            
           
                
        for category_list in categories_.values(): 
                    
            if set(y_int_test).issubset(set(category_list)):
                
                
                #print('within pair:',y_int_test)
                
        
                within_total+=1
                within_correct+=leave_two_out(predict_list,label_list)
    
            
                
        os.remove(save_model)
                
                
    accuracy = round(1.0 * correct / total_count, 2 )
    total_mse = round(sum(mse_list)/len(mse_list),4)
    within_accuracy = round(1.0 * within_correct/within_total, 2)
    outside_accuracy = round(1.0 * outside_correct/outside_total, 2)
    
    
    print('within correct:', within_correct)
    print('within total:', within_total)
    print('outside correct:', outside_correct)
    print('outside total:', outside_total)
    #print('Total count:', total_count )
    
    
    
    wdb_logger.log({"val/matching_score":accuracy,
                    "val/mse":total_mse,
                    "val/within_acc": within_accuracy, 
                    "val/out_acc":outside_accuracy
                
                    })
    
    
    wdb_logger.finish()
    
    

        
 
if __name__=='__main__':
    parser = parse_args()
    main(parser)
