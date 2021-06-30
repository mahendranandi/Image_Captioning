'''
this is another most inportant part of the whole project. it i svery much necessary to practice to right your own dataloader function , because in each and every case the function is slightly going to be changed according the the raw data and the input taking be the model

'''
from tqdm import tqdm
#import tqdm 
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
# follow that this file is an independent file, not depending on any other files

# below codes are pretty much simple , if you feel  probelm to understand those make a print every 
# time and check the output, you have ot learn this so i am not commenting it,be master of it with your own.
def data_generator(train_encoded_captions,train_features,batch_size,vocab_size):
  X1, X2, Y = list(), list(), list()
  max_length=40
  n=0
  for img_id in tqdm(train_encoded_captions):
    n+=1
    for i in range(5):
      for j in range(1,40):
        curr_sequence=np.copy(train_encoded_captions[img_id][i][0:j].tolist())
        next_word=np.copy(train_encoded_captions[img_id][i][j])
        curr_sequence=pad_sequences([curr_sequence], maxlen=max_length, padding='post')[0]
        one_hot_next_word=np_utils.to_categorical([next_word],vocab_size)[0]
        X1.append(train_features[img_id])
        X2.append(curr_sequence)
        Y.append(one_hot_next_word)
    if(n==batch_size):
      yield ((np.asarray(X1), np.asarray(X2)), np.asarray(Y))
      X1, X2, Y = list(), list(), list()
      n=0