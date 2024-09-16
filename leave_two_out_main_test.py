import torch 
import os
import numpy as np
import argparse
import wandb 
import random
import csv
import itertools

from sklearn.model_selection import LeavePOut 
from evaluation import leave_two_out




from utils import (experiment, train_load, dataloaders,train_val, initialize_model,prediction,
                   regression_experiment_data,scale_data,initialize_wise_model)

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
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--val_size', type=int, default=4, help='validation samples to stop model from training')
    parser.add_argument('--layers', type=int, default=5, help='Number of layers of BI-LSTM')
    parser.add_argument('--hidden_size', type=int, default=4, help='hidden_size of BI-LSTM')
    parser.add_argument('--hidden_size1', type=int, default=44, help='hidden_size of wise LSTM')
    parser.add_argument('--hidden_size2', type=int, default=10, help='hidden_size of second wise LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--subject', type=int, default=0, help='subject')
    parser.add_argument('--window_length', type=int, default=10, help='Number of samples to select from each stimuli response')
    parser.add_argument('--delay', type=int, default=48, help='Delay in the stimuli response')
    parser.add_argument('--word_vec_dim', type=int, default=50, help='Dimension of the word vectors')
    parser.add_argument('--sections', type=int, default=1, help='Number of sections to divide the stimuli response')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for the window')
    parser.add_argument('--modality', type=str, default='avg_trial', help='Modality of the data')
    parser.add_argument('--dropout',type=float,default=0.0,help= 'dropout in LSTM' )
    parser.add_argument('--weight_decay', type=float, default=0.05, help='L2 regularization')
    parser.add_argument('--model', type=str, default='BiLSTM', help='Model to use')
    parser.add_argument('--norm',type=float,default=0.5,help= 'Control prediction norm')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use')
    parser.add_argument('--nonlinearity', type=str, default='tanh', help='activation function for RNN')
    parser.add_argument('--scale', type=str, default='normalization', help='Scaling method for the data')
    parser.add_argument('--architecture', type=str, default='basic', help='Choose  LSTM architecture for regression')
    parser.add_argument('--save_logs', type=str, default='/mnt/gpfs2_4m/scratch/spo230', help='Path to save logs')
   
    
    return parser.parse_args()


def main(args):
    
    set_seed()
    
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

    categories = [ca1, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9, ca10]   
    
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
    
    architecture = args.architecture
    
    print("architecture:",architecture)
    mode = args.scale
    
    
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
    

    X_train, Y_train,_ = experiment(data,stimuli_embedding.numpy(),modality,categories= False)
    
    #Scaling data
    
    X_train,Y_train = scale_data(X_train,Y_train,mode)



   
    #Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Metrics
    correct =0
    total_count = 0

        
    #leave two out cross validation
    lpo = LeavePOut(p)

    run = 0
    
    categories = np.array(categories)
    # Create the array from 0 to 9
    array = np.arange(10)


    for train_index, test_index in lpo.split(array):  
        
        set_seed() 
        
        run+=1
        
        
        predict_list = []
        label_list = []
        
        ca_train = categories[train_index].reshape(-1)
        
        x_train= X_train[ca_train]
        y_train = np.array(Y_train[ca_train])
        
        val_index = np.random.choice(range(len(x_train)),val_size,replace= False)
        
    
        
        x_val,y_val = x_train[val_index],y_train[val_index]
        
        x_train = np.delete(x_train, val_index, axis=0)
        y_train = np.delete(y_train, val_index, axis=0)
        
        
        test_1 = categories[test_index[0]]
        test_2 = categories[test_index[1]]
        
        pairs = list(itertools.product(test_1, test_2))

        
        #####Train LSTM model #######
        
        #model 
        
        if architecture == 'wise':
            
            net = initialize_wise_model(model,device, input_size=22, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                                        nonlinearity = nonlinearity,output_size=word_vec_dim)
        
        elif architecture == 'basic':
            
            print("using basic architecture")
            
            net = initialize_model(model,device, input_size=22, hidden_size=hidden_size, num_layers=num_layers, 
                                    nonlinearity = nonlinearity,output_size=word_vec_dim,dropout=dropout)
            
        net = net.to(device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss().to(device)
        
        train_loader,val_loader= dataloaders(x_train,y_train,x_val,y_val)
        
        if architecture == 'wise':
            
            save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_hd1_{hidden_size1}_hd2_{hidden_size2}_lr_{learning_rate}_wd_{weight_decay}_ws_{wind_length}_d_{delay}_subject_{subject}_norm{norm}_scale_{mode}.pt"
        
        elif architecture == 'basic':
            
            print("using basic architecture")
        
            
            save_model = save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_Ly_{num_layers}_hs_{hidden_size}_lr_{learning_rate}_d_{dropout}_wd_{weight_decay}_window__{wind_length}_d_{delay}_subject_{subject}_scale_{mode}.pt"
           

        train_val(train_loader,val_loader,device, optimizer, criterion, epochs,net, loss = loss,file=save_model,norm= norm)
        
        for i in pairs:
            
        
            x_test = X_train[np.asarray(i)]
            y_test = np.array(Y_train[np.asarray(i)])

            test_loader= train_load(x_test,y_test)
            
            

            state_dict = torch.load(save_model)

            # Load the trained  state dict  into the model
            net.load_state_dict(state_dict)
            
            #prediction 
            
            predictions = prediction(net,test_loader,device)
            
            predictions = predictions.cpu().detach().numpy()
            
            
            for k in range(len(x_test)):
                
                predict_list.append(predictions[k].reshape(1, -1))
                label_list.append(y_test[k].reshape(1, -1))
                
            
        
            correct+= leave_two_out(predict_list,label_list)
            
            #if leave_two_out(predict_list,label_list) == 1:
                
                #torch.save(trained_model.state_dict(), "models/LSTM_avg_trial_bidi_train_model_subject" + f"_{subject+1}_MSE.pt")
                
            total_count+=1
            
            #Evaluate perfomance of the model within  and outside condition
            
        os.remove(save_model)
                
                
    accuracy = round(1.0 * correct / total_count, 2 )
    
    print("Matching Score:",accuracy)
   
    
    #save_file = "/home/sposso22/paper_fNIRS/all_subjects_results.csv"
    
    #acc_row = [learning_rate,num_layers,hidden_size,dropout,within_accuracy,outside_accuracy]
    
    #with open(save_file, 'a') as f:
        #csvwriter = csv.writer(f, delimiter=',')
        #csvwriter.writerow(acc_row)
    
    wdb_logger.log({"val/matching_score":accuracy
                    })
    
    

        
 
if __name__=='__main__':
    parser = parse_args()
    main(parser)