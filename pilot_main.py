import torch 
import os
import numpy as np
import argparse
import wandb 
import random
import csv

from sklearn.model_selection import LeavePOut 
from evaluation import leave_two_out
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



from utils import (experiment, train_val, 
                   initialize_model,initialize_wise_model,prediction,
                   pilot_regression_experiment_data,scale_data, full_dataloaders)

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
    parser.add_argument('--val_size', type=int, default=1, help='validation samples to stop model from training')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers of basic LSTM')
    parser.add_argument('--hidden_size', type=int, default=40, help='hidden_size of LSTM')
    parser.add_argument('--hidden_size1', type=int, default=44, help='hidden_size of wise LSTM')
    parser.add_argument('--hidden_size2', type=int, default=10, help='hidden_size of second wise LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--window_length', type=int, default=2, help='Number of samples to select from each stimuli response')
    parser.add_argument('--delay', type=int, default=22, help='Delay in the stimuli response')
    parser.add_argument('--word_vec_dim', type=int, default=50, help='Dimension of the word vectors')
    parser.add_argument('--sections', type=int, default=1, help='Number of sections to divide the stimuli response')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for the window')
    parser.add_argument('--modality', type=str, default='avg_trial', help='Modality of the data')
    parser.add_argument('--dropout',type=float,default=0.0,help= 'dropout in LSTM' )
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='L2 regularization')
    parser.add_argument('--model', type=str, default='BiLSTM', help='Model to use')
    parser.add_argument('--norm',type=float,default=0.0,help= 'Control prediction norm')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use')
    parser.add_argument('--nonlinearity', type=str, default='tanh', help='activation function for RNN')
    parser.add_argument('--scale', type=str, default='normalization', help='Scaling method for the data')
    parser.add_argument('--architecture', type=str, default='basic', help='Choose  LSTM architecture for regression')
    parser.add_argument('--save_logs', type=str, default='/mnt/gpfs2_4m/scratch/spo230', help='Path to save logs')
   
    
    return parser.parse_args()


def main(args):
    
    set_seed()
    
    
    categories ={"ca1":[0,1,2,3],
                "ca2":[4,5,6,7]}
    
    
    ###Wandb####
    
    wdb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="LSTM_finetuning",
        name = f"lstm_{args.layers}_hidden_{args.hidden_size}_lr_{args.learning_rate}_dropout_{args.dropout}",
        dir = args.save_logs,
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout":  args.dropout, 
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
    modality= args.modality
    dropout =args.dropout
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
    val_size = args.val_size
    
    ##Wise architecture
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    
    architecture = args.architecture
    
    print('The architecture chosen is: ', architecture)
    
    #Scaling data
    mode = args.scale
   

    #Training parameters 
    
    #Leave 2 out cross validation
    
    p= 2
    
    avg_accuracy = []
    avg_mse = []
    avg_within_accuracy = []
    avg_outside_accuracy = []
   

    #Preparing data for the experiment
    
    for subject in range(4):

        data,stimuli_embedding = pilot_regression_experiment_data(data_path,subject,sections,delay,wind_length,step_size,word_vec_dim)
        

        X_train, Y_train,Y_int_labels = experiment(data,stimuli_embedding.numpy(),modality,categories= False)
        
        #Scaling data
        #X_trian shape (batch_size,channels,window_length)
        X_train,Y_train = scale_data(X_train,Y_train,mode)
        
        #normx= np.linalg.norm(X_train,ord=2,axis=2)
        #normy = np.linalg.norm(Y_train,ord=2,axis=1)
        
        #print('norm x: ', X_train[0][0])
        #print('norm y: ', Y_train[0])
        
        
    
        #Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        #Metrics
        correct =0
        within_total = 0
        within_correct = 0
        outside_total = 0
        outside_correct = 0

        total_count = 0
        
        mse_list = []
        
       

            
        #leave two out cross validation
        lpo = LeavePOut(p)

        run = 0
        
        

        for train_index, test_index in lpo.split(X_train):   
            
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
            
                net = initialize_wise_model(model,device, input_size=46, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                               nonlinearity = nonlinearity,output_size=word_vec_dim)
                
                save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_hid1_{hidden_size1}_hid2_{hidden_size2}_lr_{learning_rate}_wd_{weight_decay}_ws__{wind_length}_d_{delay}_sjt_{subject}_nrm_{norm}_loss_{loss}.pt"
                
            elif architecture == 'basic':
                
                net = initialize_model(model,device, input_size=46, hidden_size=hidden_size, num_layers=num_layers, 
                                       nonlinearity = nonlinearity,output_size=word_vec_dim,dropout=dropout)
                
                save_model = "models/model" + f"_{model}_nonl_{nonlinearity}_Lyr_{num_layers}_hs_{hidden_size}_lr_{learning_rate}_d_{dropout}_wd_{weight_decay}_wl_{wind_length}_d_{delay}_sjt_{subject}_nrm_{norm}_loss_{loss}.pt"
                
            
            net = net.to(device)
        
            
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay= weight_decay)
            criterion = torch.nn.MSELoss().to(device)
            
            
            train_loader,val_loader,test_loader  = full_dataloaders(x_train,y_train,x_test,y_test,x_val,y_val)
            
            
            
                
            print("___________________________________________________________________________________")
            train_val(train_loader,val_loader,device, optimizer, criterion, epochs,net, loss = loss,file=save_model,norm= norm)
            print("___________________________________________________________________________________")
            
            state_dict = torch.load(save_model)

            # Load the trained  state dict  into the model
            net.load_state_dict(state_dict)
            
            #prediction 
            
            predictions = prediction(net,test_loader,device)
            
            predictions = predictions.cpu().detach().numpy()
            
            
            #print('predicitons mean', np.mean(predictions[0]))
            #print('True mean', np.mean(y_test[0]))
            
            
            #random_int = random.randint(0, len(predictions))
            
            #plt.figure()
            
            #plt.plot(predictions[0], label='Predicted')
            #plt.plot(y_test[0], label='True')
            #plt.legend()
            #plt.savefig("my_plot.png")
                
        
            for k in range(len(x_test)):
                
                predict_list.append(predictions[k].reshape(1, -1))
                label_list.append(y_test[k].reshape(1, -1))
                
                mse = mean_squared_error(predict_list[k], label_list[k])
                
                mse_list.append(mse)
                
                
            correct+= leave_two_out(predict_list,label_list)
            
            total_count+=1 
           
            #Evaluate perfomance of the model within  and outside condition

            if not set(y_int_test).issubset(categories['ca1']) and not set(y_int_test).issubset(categories['ca2']):
                
              
                
                outside_total+=1
                outside_correct += leave_two_out(predict_list, label_list)         
            
            else:
                
             
                    
                within_total+=1
                within_correct+=leave_two_out(predict_list,label_list)
                        
                    
                    
                   
                        
            os.remove(save_model)
                        
                        
        accuracy = round(1.0 * correct / total_count, 2 )
        total_mse = sum(mse_list)/len(mse_list)
        within_accuracy = round(1.0 * within_correct/within_total, 2)
        outside_accuracy = round(1.0 * outside_correct/outside_total, 2)
        
        
        
        avg_accuracy.append(accuracy)
        avg_mse.append(total_mse)
        avg_within_accuracy.append(within_accuracy)
        avg_outside_accuracy.append(outside_accuracy)
        
        print('outside correct',outside_correct)
        print('outside total', outside_total)
        
        print('within correct',within_correct)
        print('within total', within_total)
                
    
    final_avg_acc = round(np.array(avg_accuracy).mean(),2)
    final_std_acc = round(np.array(avg_accuracy).std(),2)
    final_avg_mse = round(sum(avg_mse)/len(avg_mse),3)
    final_avg_within_acc = sum(avg_within_accuracy)/len(avg_within_accuracy)
    final_avg_out_acc = sum(avg_outside_accuracy)/len(avg_outside_accuracy)
    
    print(avg_accuracy)
    print(avg_mse)
    
    
    #save_file = "/home/sposso22/paper_fNIRS/avg_results_std.csv"
    
    #acc_row = [learning_rate,num_layers,hidden_size,dropout,final_avg_mse,final_avg_acc,final_std_acc]
    
    #with open(save_file, 'a') as f:
        #csvwriter = csv.writer(f, delimiter=',')
        #csvwriter.writerow(acc_row)
        
        
   
    
    wdb_logger.log({"val/matching_score":final_avg_acc,
                    "val/mse":final_avg_mse,
                    "val/std":final_std_acc,
                    "val/within_acc": final_avg_within_acc,
                    "val/out_acc":final_avg_out_acc
                   })
    
    
    wdb_logger.finish()
    
    
        
 
if __name__=='__main__':
    parser = parse_args()
    main(parser)