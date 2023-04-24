import random
import pandas as pd
import pickle

from eda import EDA
from eunjeon import Mecab

if __name__ == '__main__' :

    with open('data/source_data/bank_fraud.txt', 'r', encoding='UTF8') as f :
        bank_fraud = f.readlines()

    with open('data/source_data/his_voice.txt', 'r', encoding='UTF8') as f :
        his_voice = f.readlines()

    with open('data/source_data/impersonation.txt', 'r', encoding='UTF8') as f :
        impersonation = f.readlines()

    bank_fraud = list(set([v[:-1] for v in bank_fraud]))
    his_voice = list(set([v[:-1] for v in his_voice]))
    impersonation = list(set([v[:-1] for v in impersonation]))
    total = bank_fraud + his_voice + impersonation

    print("number of total data : ", len(total))
    
    voice_phishing_train = random.sample(total, 220)
    voice_phishing_test = [v for v in total if v not in voice_phishing_train]
    print("number of voice phishing train data : ", len(voice_phishing_train))
    print("number of voice phishing test data : ", len(voice_phishing_test))

    # save txt to test deplication
    mecab = Mecab()
    voice_phishing_train_token_tmp = [mecab.nouns(w) for w in voice_phishing_train]
    voice_phishing_test_token_tmp = [mecab.nouns(w) for w in voice_phishing_test]

    voice_phishing_train_token_tmp = [' '.join(w)+'\n' for w in voice_phishing_train_token_tmp]
    voice_phishing_test_token_tmp = [' '.join(w)+'\n' for w in voice_phishing_test_token_tmp]

    print("number of train data for duplication test : ", len(voice_phishing_train_token_tmp))
    print("number of test data for duplication test : ", len(voice_phishing_test_token_tmp))

    with open('data/duplication_test_data/voice_train.txt', 'w', encoding='UTF8') as f:
        f.writelines(voice_phishing_train_token_tmp)
    with open('data/duplication_test_data/voice_test.txt', 'w', encoding='UTF8') as f:
        f.writelines(voice_phishing_test_token_tmp)
    
    del voice_phishing_train_token_tmp
    del voice_phishing_test_token_tmp


    voice_phishing_train_augmented = []
    voice_phishing_test_augmented = []

    for v in voice_phishing_train:
        augs = EDA(v)
        for aug in augs:
            voice_phishing_train_augmented.append(aug)

    for v in voice_phishing_test:
        augs = EDA(v)
        for aug in augs:
            voice_phishing_test_augmented.append(aug)

    print("number of voice phishing train data after EDA : ", len(voice_phishing_train_augmented))
    print("number of voice phishing test data after EDA : ", len(voice_phishing_test_augmented))

    voice_phishing_train_token = [mecab.nouns(w) for w in voice_phishing_train_augmented]
    voice_phishing_test_token = [mecab.nouns(w) for w in voice_phishing_test_augmented]    

    tmp1 = []
    for i in range(len(voice_phishing_train_token)) :
        tmp2 = []
        for j in range(len(voice_phishing_train_token[i])) :
            if len(voice_phishing_train_token[i][j]) != 1 :
                tmp2.append(voice_phishing_train_token[i][j])
        if len(tmp2) >= 30 :
            tmp1.append(tmp2)
    voice_phishing_train_token = tmp1[:]

    tmp1 = []
    for i in range(len(voice_phishing_test_token)) :
        tmp2 = []
        for j in range(len(voice_phishing_test_token[i])) :
            if len(voice_phishing_test_token[i][j]) != 1 :
                tmp2.append(voice_phishing_test_token[i][j])
        if len(tmp2) >= 30 :
            tmp1.append(tmp2)
    voice_phishing_test_token = tmp1[:]

    print("number of voice phishing train data after EDA and preprocessing : ", len(voice_phishing_train_token))
    print("number of voice phishing test data after EDA and preprocessing : ", len(voice_phishing_test_token))
    
# ----------------------------------------------------------------------------------------------------------------
    
    normal_df = pd.read_csv('data/source_data/normal.csv')
    normal_all = random.sample(list(normal_df['sentence']), 5000)
    
    normal_all_token = [mecab.nouns(w) for w in normal_all]

    tmp1 = []
    for i in range(len(normal_all_token)) :
        tmp2 = []
        for j in range(len(normal_all_token[i])) :
            if len(normal_all_token[i][j]) != 1 :
                tmp2.append(normal_all_token[i][j])
        if len(tmp2) >= 30 :
            tmp1.append(tmp2)
    normal_all_token = tmp1[:]

    normal_train_token = random.sample(normal_all_token, len(voice_phishing_train_token))
    normal_test_token = [v for v in normal_all_token if v not in normal_train_token]

    normal_test_token = random.sample(normal_test_token, len(voice_phishing_test_token))

    print("number of normal train data after preprocessing : ",len(voice_phishing_train_token))
    print("number of normal test data after preprocessing : ",len(voice_phishing_test_token))
    
    random.shuffle(voice_phishing_train_token)
    random.shuffle(voice_phishing_test_token)
    random.shuffle(normal_train_token)
    random.shuffle(normal_test_token)
    
    # save pickle
    with open('data/augmented_data/voice_train.pickle', 'wb') as f:
        pickle.dump(voice_phishing_train_token, f)
    with open('data/augmented_data/voice_test.pickle', 'wb') as f:
        pickle.dump(voice_phishing_test_token, f)
    with open('data/augmented_data/normal_train.pickle', 'wb') as f:
        pickle.dump(normal_train_token, f)
    with open('data/augmented_data/normal_test.pickle', 'wb') as f:
        pickle.dump(normal_test_token, f)