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

from sklearn.metrics.pairwise import cosine_similarity

from utils import (experiment,regression_experiment_data,
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
    
    stimuli = np.array(['lettuce','T-shirt', 'cat', 'cow', 'tower', 
             'eye', 'sofa', 'chair', 'table', 'pyramid', 
             'tomato', 'ant', 'pool', 'chopsticks', 
             'hammer', 'corn', 'carrot', 'housefly', 'saw', 
             'butterfly', 'pan', 'spoon', 'glass', 'jeans', 'dog', 
             'stadium', 'bicycle', 'bed', 'bookcase', 'horse',
             'arm', 'foot', 'palm', 'scissors', 'leg',
             'celery', 'dragonfly','bee', 'pliers', 'knife',
             'car', 'train', 'aircraft', 'truck','sweater',
             'skirt', 'screwdriver', 'dress','panda','tiananmen'])
    
    ###Wandb####



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



    #Metrics
    within_total = 0
    within_cosine_sum = 0
    outside_total = 0
    outside_cosine_sum = 0




        
    #leave two out cross validation
    lpo = LeavePOut(p)

    run = 0
    

    for _, test_index in lpo.split(X_train):   
        
        set_seed()
        
        run+=1
        
        y_test = np.array(Y_train[test_index])
        y_int_test = np.array(Y_int_labels[test_index])
        
        #####Train LSTM model #######
        
        #model 
        
        glove_list = []
        
        for k in range(len(y_test)):
            
            glove_list.append(y_test[k].reshape(1, -1))
            
            #mse = mean_squared_error(predict_list[k],label_list[k])
            
            #mse_list.append(mse)
            
            #print('mse: ',mse)

        
        #Evaluate perfomance of the model within  and outside condition
        
        if all (not set(y_int_test).issubset(category_list) for category_list in categories_.values()):
            
            outside_total+=1
            
            
            outside_cosine = cosine_similarity(glove_list[0], glove_list[1]).reshape(-1).item()
            
            #if outside_cosine > 0.6:
                
                #print('outside pair:',stimuli[y_int_test])
                
                #print('outside cosine:', outside_cosine)
            
            outside_cosine_sum+=outside_cosine
            
            
           
                
        for category_list in categories_.values(): 
                    
            if set(y_int_test).issubset(set(category_list)):
                
                
                #print('within pair:',y_int_test)
                
        
                within_total+=1
                within_cosine=cosine_similarity(glove_list[0], glove_list[1]).reshape(-1).item()
                
                if within_cosine > 0.6:
                    
                    print('within pair:',stimuli[y_int_test])
                    print('within cosine:', within_cosine)
                
                
                
                within_cosine_sum+=within_cosine
    
            
                 
    print("-----------------------------------------")            
    print('within cosine sum:', within_cosine_sum)
    print('outside cosine sum', outside_cosine_sum)
    print('within total:', within_total)
    print('outside total:', outside_total)
    within_cosine_mean = round(1.0 * within_cosine_sum/within_total, 6)
    outside_cosine_mean = round(1.0 * outside_cosine_sum/outside_total,6)
    
    
    print('within cosine mean:', within_cosine_mean)
    print('outside cosine mean:', outside_cosine_mean)
    
    #print('run:', run)

    #print('Total count:', total_count )
    
    

        
 
if __name__=='__main__':
    parser = parse_args()
    main(parser)