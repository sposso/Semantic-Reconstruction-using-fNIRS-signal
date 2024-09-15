from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def leave_two_out(predict_list, label_list):
    
    correct_cosine_score = cosine_similarity(predict_list[0], label_list[0]) + cosine_similarity(predict_list[1], label_list[1])
    predict_cosine_score = cosine_similarity(predict_list[0], label_list[1]) + cosine_similarity(predict_list[1], label_list[0])                  
                
    round_c = correct_cosine_score[0][0]
    round_i = predict_cosine_score[0][0]
    
   
            
    if round_c > round_i:
        return 1
    else:
        return 0


def mse_evaluation(predict, true_label):
    return mean_squared_error(predict, true_label)
 