import pickle
import math

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from detect import cos_sim, convrs_to_timeseries, EWMA_bias_correction, evaluate
from gensim.models import FastText

def evaluate_keyword_matching(voice_phishing, normal, keyword) :
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ARL1 = 0
    ARL0 = 0
    ARL1_lst = []
    ARL0_lst = []

    for i in range(len(voice_phishing)) :
        all_sent = ''.join(voice_phishing[i])
        min = 1e9
        for j in keyword :
            v = all_sent.find(j)
            if v != -1 and v < min:
                min = v
        if min == 1e9 :
            FN += 1
        else :
            TP += 1
            ARL1_lst.append(min)

    for i in range(len(normal)) :
        all_sent = ''.join(normal[i])
        min = 1e9
        for j in keyword :
            v = all_sent.find(j)
            if v != -1 and v < min:
                min = v
        if min == 1e9 :
            TN += 1
        else :
            FP += 1
            ARL0_lst.append(min)

    if len(ARL1_lst) == 0 :
        ARL1 = 'ARL1_lst==0'
    else :
        ARL1 = sum(ARL1_lst)/len(ARL1_lst)

    if len(ARL0_lst) == 0 :
        ARL0 = 'ARL0_lst==0'
    else :
        ARL0 = sum(ARL0_lst)/len(ARL0_lst)

    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    # Discard the third decimal place
    acc = math.floor(acc*100)/100
    recall = math.floor(recall*100)/100
    precision = math.floor(precision*100)/100
    ARL1 = math.floor(ARL1*100)/100
    ARL0 = math.floor(ARL0*100)/100

    return f'Accuracy : {acc}, Recall : {recall}, Precision : {precision}, ARL1 : {ARL1}, ARL0 : {ARL0}', acc, recall, precision, ARL1, ARL0

def main():
    # Load data
    with open('./data/augmented_data/normal_test.pickle','rb') as f:
        normal_test_token = pickle.load(f)
    with open('./data/augmented_data/voice_test.pickle','rb') as f:
        voice_phishing_test_token = pickle.load(f)

    # Define voice phishing keywords
    keywords = ['대포통장', '명의도용', '개인정보유출', '금융범죄', '수사관', '동결', '서울중앙지검', '신용카드', '피해자', '녹취']

    # Load pretrained(+continue trained) FastText model
    model= FastText.load('./fasttext/continue_trained/fasttext')
    keywords_vect = []
    for i in range(len(keywords)) :
        keywords_vect.append(model.wv[keywords[i]])

    # Turn conversations into time series data with cosine similarity between embedded values
    voice_phishing_test_time_series = []
    normal_test_time_series = []

    for i in range(len(normal_test_token)) :
        normal_test_time_series.append(convrs_to_timeseries(normal_test_token[i], keywords_vect, model))
    for i in range(len(voice_phishing_test_token)) :
        voice_phishing_test_time_series.append(convrs_to_timeseries(voice_phishing_test_token[i], keywords_vect, model))

    # Apply bias-correction EWMA to conversation time series data
    voice_phishing_test_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in voice_phishing_test_time_series]
    normal_test_EWMA = [EWMA_bias_correction(i, 0.9)[1] for i in normal_test_time_series]

    print('Our method : ', evaluate(voice_phishing_test_EWMA, normal_test_EWMA, 0.425, 9)[0])
    print('Keyword matching method : ', evaluate_keyword_matching(voice_phishing_test_token, normal_test_token, keywords)[0])

if __name__ == '__main__':
    main()