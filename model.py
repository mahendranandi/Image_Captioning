'''
 # building the LSTM model here using the keras lybrary
 YOUR WORK: you should try with an addition layer after you are done with this simple one. 

'''

#importing dependencies
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import add
from keras.layers.embeddings import Embedding
from keras.utils import np_utils   # be aware here , i was facing some issues here regarding version. I will keep a cell in any of the notebooks printing the versions of the lybraries i am using.

# follow that this file is an independent file, not depending on any other files


class DefineModel():
  def __init__(self,maximum_length,vocab_size):
    self.vocab_size = vocab_size
    self.max_length = maximum_length

  def make_model(self):
    input_1=Input(shape=(2048,))
    dropout_1=Dropout(0.2)(input_1)
    dense_1=Dense(256,activation='relu')(dropout_1)

    input_2=Input(shape=(self.max_length,))
    embedding_1=Embedding(self.vocab_size,256)(input_2)
    dropout_2=Dropout(0.2)(embedding_1)
    lstm_1=LSTM(256)(dropout_2)

    add_1=add([dense_1,lstm_1])
    dense_2=Dense(256,activation='relu')(add_1)
    dense_3=Dense(self.vocab_size,activation='softmax')(dense_2)

    model=Model(inputs=[input_1,input_2],outputs=dense_3)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    self.model = model

    return self
# be aware of the fact that by calling only the make_model function you are not spposed to get the model,
  #  infact here you have to call it successively because make_model returns 'self'.
  def get_model(self):
    return self.model
  
  def get_model_summary(self):
    return self.model.summary()

















