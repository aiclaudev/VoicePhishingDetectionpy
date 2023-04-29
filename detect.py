import pickle
import numpy as np
from gensim.models import FastText
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
    return dot(A, B) / (norm(A)*norm(B))

def convrs_to_timeseries(target, keywords_vect, model) :
    target_sim = []
    target_vect = []

    for i in range(len(target)) :
        target_vect.append(model.wv[target[i]])

    for i in range(len(target_vect)) :
        tmp_sum = 0
        for j in range(len(keywords_vect)) :
            tmp_sum += cos_sim(target_vect[i], keywords_vect[j])
        if np.isnan(tmp_sum) :
            continue
        target_sim.append(tmp_sum / len(keywords_vect))

    return target_sim

def EWMA_bias_correction(lst : list, alpha : float) :
    result1 = [0]
    result2 = [0]
    for i in range(len(lst)) :
        ewma = (1-alpha) * lst[i] + alpha * result1[i]
        result1.append(ewma)
        bias = ewma / (1-alpha**(i+1))
        result2.append(bias)
    return result1, result2

def evaluate(voice_phishing, normal, threshold, count) :
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ARL1 = 0
    ARL0 = 0
    ARL1_lst = []
    ARL0_lst = []

    for i in range(len(voice_phishing)) :
        cnt = 0
        for j in range(len(voice_phishing[i])) :
            if voice_phishing[i][j] >= threshold :
                cnt += 1
            if cnt >= count :
                ARL1_lst.append(j)
                break
        if cnt >= count :
            TP += 1
        else :
            FN += 1

    for i in range(len(normal)) :
        cnt = 0
        for j in range(len(normal[i])) :
            if normal[i][j] >= threshold :
                cnt += 1
            if cnt >= count :
                ARL0_lst.append(j)
                break
        if cnt >= count :
            FP += 1
        else :
            TN += 1

    # print(len(ARL1_lst))
    # print(len(ARL0_lst))
    # print(TP)
    # print(FP)
    # print(ARL1_lst)
    # print(ARL0_lst)

    if len(ARL1_lst) == 0 :
        ARL1 = 'ARL1_lst==0'
    else :
        ARL1 = sum(ARL1_lst)/len(ARL1_lst)
    if len(ARL0_lst) == 0 :
        ARL0 = 'ARL0_lst==0'
    else :
        ARL0 = sum(ARL0_lst)/len(ARL0_lst)

    return {'Acc' : (TP + TN) / (TP + TN + FP + FN), 'Recall' : TP / (TP + FN), 'Precision' : TP / (TP + FP), 'ARL1' : ARL1, 'ARL0' : ARL0}

def main():
    # Load data
    with open('./data/augmented_data/normal_test.pickle','rb') as f:
        normal_test_token = pickle.load(f)
    with open('./data/augmented_data/normal_train.pickle','rb') as f:
        normal_train_token = pickle.load(f)
    with open('./data/augmented_data/voice_train.pickle','rb') as f:
        voice_phishing_train_token = pickle.load(f)
    with open('./data/augmented_data/voice_test.pickle','rb') as f:
        voice_phishing_test_token = pickle.load(f)


    # Load pretrained(+continue trained) FastText model
    model= FastText.load('./fasttext/continue_trained/fasttext')

    # Keywords about voice phishing
    keywords = ['대포통장', '명의도용', '개인정보유출', '금융범죄', '수사관', '동결', '서울중앙지검', '신용카드', '피해자', '녹취']
    keywords_vect = []
    for i in range(len(keywords)) :
        keywords_vect.append(model.wv[keywords[i]])

    # Turn conversations into time series data with cosine similarity between embedded values
    voice_phishing_train_time_series = []
    normal_train_time_series = []
    voice_phishing_test_time_series = []
    normal_test_time_series = []

    for i in range(len(normal_train_token)) :
        normal_train_time_series.append(convrs_to_timeseries(normal_train_token[i], keywords_vect, model))
    for i in range(len(voice_phishing_train_token)) :
        voice_phishing_train_time_series.append(convrs_to_timeseries(voice_phishing_train_token[i], keywords_vect, model))
    for i in range(len(normal_test_token)) :
        normal_test_time_series.append(convrs_to_timeseries(normal_test_token[i], keywords_vect, model))
    for i in range(len(voice_phishing_test_token)) :
        voice_phishing_test_time_series.append(convrs_to_timeseries(voice_phishing_test_token[i], keywords_vect, model))

    # Apply bias-correction EWMA to conversation time series data
    voice_phishing_train_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in voice_phishing_train_time_series]
    normal_train_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in normal_train_time_series]
    voice_phishing_test_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in voice_phishing_test_time_series]
    normal_test_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in normal_test_time_series]

    print('Train set', evaluate(voice_phishing_train_EWMA, normal_train_EWMA, 0.44, 10))
    print('Test set', evaluate(voice_phishing_test_EWMA, normal_test_EWMA, 0.44, 10))

if __name__ == '__main__':
    main()