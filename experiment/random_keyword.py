import pickle
import random
import math
from tqdm.auto import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from keyword_matching import evaluate_keyword_matching
from detect import convrs_to_timeseries, EWMA_bias_correction, evaluate
from gensim.models import FastText

def random_keyword_evaluate(keywords, model, voice_phishing_test_token, normal_test_token, num) :
    random.seed(777)

    our_method = {'Accuracy' : [], 'Recall' : [], 'Precision' : [], 'ARL1' : [], 'ARL0' : []}
    search_based = {'Accuracy' : [], 'Recall' : [], 'Precision' : [], 'ARL1' : [], 'ARL0' : []}

    for _ in tqdm(range(num)) :
        keyword = random.sample(keywords, 7)
        print(sorted(keyword))
        keyword_vect = []

        for i in range(len(keyword)) :
            keyword_vect.append(model.wv[keyword[i]])
        voice_phishing_test_vanila_ = []
        normal_test_vanila_ = []

        for i in range(len(normal_test_token)) :
            normal_test_vanila_.append(convrs_to_timeseries(normal_test_token[i], keyword_vect, model))

        for i in range(len(voice_phishing_test_token)) :
            voice_phishing_test_vanila_.append(convrs_to_timeseries(voice_phishing_test_token[i], keyword_vect, model))
    
        voice_phishing_test_EWMA2 = [EWMA_bias_correction(i, 0.9)[1] for i in voice_phishing_test_vanila_]
        normal_test_EWMA2 = [EWMA_bias_correction(i, 0.9)[1] for i in normal_test_vanila_]
        
        our_method_result = evaluate(voice_phishing_test_EWMA2, normal_test_EWMA2, 0.44, 10)
        search_based_result = evaluate_keyword_matching(voice_phishing_test_token, normal_test_token, keyword)

        for n, key in enumerate(our_method.keys()):
            our_method[key].append(our_method_result[n+1])
            search_based[key].append(search_based_result[n+1])

    return our_method, search_based

def dictionary_to_result(dic, num):
    acc_lst = dic['Accuracy']
    recall_lst = dic['Recall']
    precision_lst = dic['Precision']
    ARL1_lst = dic['ARL1']
    ARL0_lst = dic['ARL0']
    ARL0_lst = [arl0 for arl0 in ARL0_lst if type(arl0) != str]

    acc = sum(acc_lst) / len(acc_lst)
    recall = sum(recall_lst) / len(recall_lst)
    precision = sum(precision_lst) / len(precision_lst)
    ARL1 = sum(ARL1_lst) / len(ARL1_lst)
    ARL0 = sum(ARL0_lst) / len(ARL0_lst)

    # Discard the third decimal place
    acc = math.floor(acc*100)/100
    recall = math.floor(recall*100)/100
    precision = math.floor(precision*100)/100
    ARL1 = math.floor(ARL1*100)/100
    ARL0 = math.floor(ARL0*100)/100

    return f'Accuracy : {acc}, Recall : {recall}, Precision : {precision}, ARL1 : {ARL1}, ARL0 : {ARL0}'

def main():
    # Load data
    with open('./data/augmented_data/normal_test.pickle','rb') as f:
        normal_test_token = pickle.load(f)
    with open('./data/augmented_data/voice_test.pickle','rb') as f:
        voice_phishing_test_token = pickle.load(f)

    # Define voice phishing keywords
    keywords = ['대포통장', '명의도용', '개인정보유출', '금융범죄', '수사관', '동결', '서울중앙지검', '신용카드', '피해자', '녹취']

    # Load pretrained(+continue trained) FastText model
    model = FastText.load('./fasttext/continue_trained/fasttext')
    
    num = 3
    our_method, search_based = random_keyword_evaluate(keywords, model, voice_phishing_test_token, normal_test_token, num)
    our_method_result = dictionary_to_result(our_method, num)
    search_based_result = dictionary_to_result(search_based, num)
    
    print('Our method : ', our_method_result)
    print('Keyword matching method : ', search_based_result)

if __name__ == '__main__':
    main()