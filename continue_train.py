import pickle
from gensim import models

# Load data
with open('./data/augmented_data/normal_test.pickle','rb') as f:
  normal_test_token = pickle.load(f)

with open('./data/augmented_data/normal_train.pickle','rb') as f:
  normal_train_token = pickle.load(f)

with open('./data/augmented_data/voice_train.pickle','rb') as f:
  voice_phishing_train_token = pickle.load(f)

with open('./data/augmented_data/voice_test.pickle','rb') as f:
  voice_phishing_test_token = pickle.load(f)


# Pretrained Fasttext model
model = models.fasttext.load_facebook_model('./fasttext/pretrained/cc.ko.300.bin')

# Continue training with voicephishing trainset
model.build_vocab(voice_phishing_train_token, update=True)
model.train(voice_phishing_train_token, total_examples=len(voice_phishing_train_token), epochs=10)
model.save('./fasttext/continue_trained/fasttext')