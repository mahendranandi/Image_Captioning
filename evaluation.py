'''
evaluation is done using BLEU score. Now to get generated captions we can use two different methods..
 one of them is Greedy search.. and another one is beam search . So started with defining them.. and 
 latter called to calculate BLEU score for bith cases. Greedy search is simple one but If you are struggling
  to understand the Beam search please follow the references supplied in the Readme.me file. and ,yes, another
   thing, to calculate BLEU score you will not need to take headache , use the inbuild lybrary, but, but it will
    be good if you please visite the reference ( agai supplied in the attached readmeme file) for learning purpose. 

'''
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from keras.preprocessing.sequence import pad_sequences

# importing other required py files on which this file is depends on
from image_processing import ImageProcess
from text_processing import TextProcess
from model import DefineModel


class Evaluation():
  def __init__(self,words_to_indices,indices_to_words,model,maximum_length,k):
    self.max_length=maximum_length
    self.words_to_indices=words_to_indices
    self.indices_to_words=indices_to_words
    self.k=k
    self.model=model

  def greedy_search(self,single_feature):
    self.single_feature=single_feature
    photo=self.single_feature.reshape(1,2048)
    in_text='<start>'
    for i in range(self.max_length):
      sequence = [self.words_to_indices[s] for s in in_text.split(" ") if s in self.words_to_indices]
      sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
      y_pred = self.model.predict([photo,sequence],verbose=0)
      y_pred = np.argmax(y_pred[0])
      word = self.indices_to_words[y_pred]
      in_text += ' ' + word
      if word == '<end>':
        break
    final = in_text.split()
    final = final[1:-1]
    #final = " ".join(final)
    return final
  

  def beam_search(self,single_feature):
    self.single_feature=single_feature
    photo=self.single_feature.reshape(1,2048)
    in_text='<start>'
    sequence = [self.words_to_indices[s] for s in in_text.split(" ") if s in self.words_to_indices]
    sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
    y_pred = self.model.predict([photo,sequence],verbose=0)
    predicted=[]
    y_pred=y_pred.reshape(-1)
    for i in range(y_pred.shape[0]):
      predicted.append((i,y_pred[i]))
    predicted=sorted(predicted,key=lambda x:x[1])[::-1]
    b_search=[]
    for i in range(self.k):
      word = self.indices_to_words[predicted[i][0]]
      b_search.append((in_text +' ' + word,predicted[i][1]))
      
    for idx in range(self.max_length):
      b_search_square=[]
      for text in b_search:
        if text[0].split(" ")[-1]=="<end>":
          break
        sequence = [self.words_to_indices[s] for s in text[0].split(" ") if s in self.words_to_indices]
        sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        y_pred = self.model.predict([photo,sequence],verbose=0)
        predicted=[]
        y_pred=y_pred.reshape(-1)
        for i in range(y_pred.shape[0]):
          predicted.append((i,y_pred[i]))
        predicted=sorted(predicted,key=lambda x:x[1])[::-1]
        for i in range(self.k):
          word = self.indices_to_words[predicted[i][0]]
          b_search_square.append((text[0] +' ' + word,predicted[i][1]*text[1]))
      if(len(b_search_square)>0):
        b_search=(sorted(b_search_square,key=lambda x:x[1])[::-1])[:5]
    final=b_search[0][0].split()
    final = final[1:-1]
    #final=" ".join(final)
    return final


##
def calculate_average_BLEU_score(test_features,test_captions,words_to_indices,indices_to_words,model,maximum_length,k,method):
  i=0
  tot_score=0
  for img_id in tqdm(test_features):
    i+=1
    photo=test_features[img_id]
    reference=[]
    for caps in test_captions[img_id]:
      list_caps=caps.split(" ")
      list_caps=list_caps[1:-1]
      reference.append(list_caps)
    if method=="greedy":
      candidate=Evaluation(words_to_indices,indices_to_words,model,maximum_length,k).greedy_search(photo)
    else:
      candidate=Evaluation(words_to_indices,indices_to_words,model,maximum_length,k).beam_search(photo)
    score = sentence_bleu(reference, candidate)
    tot_score+=score
  avg_score=np.round(tot_score/i , 3)
  print("total image = ",i)
  if method=="greedy":
    print("\n * Bleu score calculated on Greedy search for {} images".format(i))
  else:
    print("\n Bleu score calculated on Beam search for K={} for {} images".format(k,i))
  print("Score: ",avg_score)
  return print("\n***\n***\n***\nAre You happy with your BLUE score ?\n If not try again with another set of parameters. \nThank YOU ,call my functions me again \n***\n***\n***\n")




if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-data_dir', type = str, default = 'data/')
  parser.add_argument('-image_folder_path', type = str, default = 'Flickr8k_Dataset/Flicker8k_Dataset/')
  parser.add_argument('-lemma_token_txt', type = str, default = 'Flickr8k_text/Flickr8k.lemma.token.txt')
  parser.add_argument('-train_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.trainImages.txt')
  parser.add_argument('-test_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.testImages.txt')
  parser.add_argument('-dev_images_txt', type = str, default ='Flickr8k_text/Flickr_8k.devImages.txt')

  parser.add_argument('-output_path', type = str, default = "./output/")
  parser.add_argument('-weight_loading_path', type = str, default = "./output15/LSTM_Model_Weights60/")
  parser.add_argument('-maximum_length', type=int,default=40)
  parser.add_argument('-architecture', type = str, default = 'resnet50')
# only for testing/evaluating
  parser.add_argument('-evaluation_method', type = str, default = 'greedy')
  parser.add_argument('-k_for_beam_search', type = int, default = 3)

  options = parser.parse_args()  
  options.image_folder_path=options.data_dir+options.image_folder_path
  options.lemma_token_txt=options.data_dir+options.lemma_token_txt
  options.train_images_txt=options.data_dir+options.train_images_txt
  options.test_images_txt=options.data_dir+options.test_images_txt
  options.dev_images_txt=options.data_dir+options.dev_images_txt

  
  image_preprocess = ImageProcess().initialize_default(options)
  text_preprocess = TextProcess().initialize_default(options).process()
  model = DefineModel(options.maximum_length, text_preprocess.get_vocab_size()).make_model().get_model()
  
  weight_loading_path=options.weight_loading_path
  model.load_weights(weight_loading_path+'my_weights')

  test_captions=text_preprocess.get_test_captions()
  test_features=image_preprocess.get_image_features(test_captions)
  words_to_indices=text_preprocess.get_w2i()
  indices_to_words=text_preprocess.get_i2w()
  vocab_size=text_preprocess.get_vocab_size()
  maximum_length=options.maximum_length
  k=options.k_for_beam_search
  method=options.evaluation_method

  # calculate_average_BLEU_score(test_features,test_captions,words_to_indices,indices_to_words,model,maximum_length,k,method="greedy")
  calculate_average_BLEU_score(test_features,test_captions,words_to_indices,indices_to_words,model,maximum_length,k,method)








