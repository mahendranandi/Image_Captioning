'''
this py file is totally created for easy use of the test.ipyn file for visualization of testing data.
 to make that file cleane I just moved the code from there to here. HERE 2 classes are here and each of 
 them containing 2 definitions. Among them 2nd onesare for plotting the images and captuons gathered in 1st ones in each class.

'''

from text_processing import TextProcess
from image_processing import ImageProcess
import cv2
import matplotlib.pyplot as plt
import argparse
import pickle
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

from model import DefineModel
from evaluation import Evaluation

data_dir = 'data/'  # you please make your present directory accordingly
lemma_token_txt_path=data_dir+'Flickr8k_text/Flickr8k.lemma.token.txt'
train_images_txt_path=data_dir+'Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_txt_path=data_dir+'Flickr8k_text/Flickr_8k.testImages.txt'
dev_images_txt_path=data_dir+'Flickr8k_text/Flickr_8k.devImages.txt'
maximum_length=40
text_preprocess = TextProcess().initialize_custom(lemma_token_txt_path, train_images_txt_path, test_images_txt_path, dev_images_txt_path,maximum_length).process()
k=3
vocab_size=text_preprocess.get_vocab_size()
test_captions=text_preprocess.get_test_captions()
words_to_indices=text_preprocess.get_w2i()
indices_to_words=text_preprocess.get_i2w()

model = DefineModel(maximum_length,vocab_size).make_model().get_model()
model.load_weights("/output15/LSTM_Model_Weights60/my_weights")  # you can change the path here .or can use the path inside the class(below) as variable sot that you can easily use it
evaluation=Evaluation(words_to_indices,indices_to_words,model,maximum_length,k)

image_folder_path ='data/Flickr8k_Dataset/Flicker8k_Dataset/'
architechture="resnet50"
image_preprocess=ImageProcess().initialize_custom(architechture,image_folder_path)
test_features=image_preprocess.get_image_features(test_captions)

import pickle
# pickle_in=open("/content/drive/MyDrive/BDA2020_MN/Projects/New_project/pickle_files/test_captions.pickle","rb")
# test_captions=pickle.load(pickle_in)

# pickle_in=open("/content/drive/MyDrive/BDA2020_MN/Projects/New_project/pickle_files/test_features.pickle","rb")
# test_features=pickle.load(pickle_in)

# pickle_in=open("/content/drive/MyDrive/BDA2020_MN/Projects/New_project/pickle_files/words_to_indices.pickle","rb")
# words_to_indices=pickle.load(pickle_in)

# pickle_in=open("/content/drive/MyDrive/BDA2020_MN/Projects/New_project/pickle_files/indices_to_words.pickle","rb")
# indices_to_words=pickle.load(pickle_in)




class TestVisuals():
  def initialization_default(self):
    self.test_features=test_features
    self.image_folder_path=image_folder_path
    self.test_captions=test_captions
    self.evaluation=evaluation
    return self

  def initialization_custom(self,test_features,test_captions,image_folder_path,evaluation):
    self.test_features=test_features
    self.image_folder_path=image_folder_path
    self.test_captions=test_captions
    self.evaluation=evaluation
    return self    

  def generate_caption(self,n=10,method="greedy"):
    i=0
    for img_id in self.test_features:
      i+=1
      img=cv2.imread(self.image_folder_path + "/" + img_id)
      plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
      photo=self.test_features[img_id]
      plt.show()
      reference=[]
      for caps in self.test_captions[img_id]:
        list_caps=caps.split(" ")
        list_caps=list_caps[1:-1]
        reference.append(list_caps)
      if method=="greedy":
        candidate=self.evaluation.greedy_search(photo)
      else:
        candidate=self.evaluation.beam_search(photo)
      score = sentence_bleu(reference, candidate)
      print("\n* Referance Captions:\n ")
      for cap in reference:
        print(" ".join(cap))
      print("\n* Predicted Caption : \n")
      print(" ".join(candidate))
      print("\n@ BLEU score in {} search: \n".format(method),score)
      if(i==n):
        break
  def plot_pred_caption(self,no_pic = 5,no_pixel = 224):
    test_img_ftr=[]
    for i in self.test_features:
      test_img_ftr.append((i,self.test_features[i]))
    self.test_img_ftr=test_img_ftr
    target_size = (no_pixel,no_pixel,3)

    count = 0
    count_pic=0
    fig = plt.figure(figsize=(10,20))
    # for jpgfnm, image_feature in zip(fnm_test[8:13],di_test[8:13]):
    for img_id, image_feature in self.test_img_ftr[10:16]:
      ## images 
      count += 1
      filename = self.image_folder_path + img_id
      image_load = load_img(filename, target_size=target_size)
      ax = fig.add_subplot(no_pic,2,count,xticks=[],yticks=[])
      ax.imshow(image_load)
      count += 1

      ## captions
      photo=image_feature.reshape(1,len(image_feature))
      pred_caption = self.evaluation.greedy_search(photo)
      pred_caption=" ".join(pred_caption)
      # print(pred_caption)
      captions=self.test_captions[img_id]
      ax = fig.add_subplot(no_pic,2,count)
      plt.axis('off')
      ax.plot()
      ax.set_xlim(0,1)
      ax.set_ylim(0,len(captions)+1)
      for i, caption in enumerate(captions):
        caption=" ".join(caption.split(" ")[1:-1])
        ax.text(0,i+1.5,caption,fontsize=20,style="italic")
      
      reference=[]
      for caps in self.test_captions[img_id]:
        list_caps=caps.split(" ")
        list_caps=list_caps[1:-1]
        reference.append(list_caps)
      candidate_g=self.evaluation.greedy_search(photo)
      candidate_b=self.evaluation.beam_search(photo)
      BS_Greedy = np.round(sentence_bleu(reference, candidate_g),3)
      BS_Beam3 = np.round(sentence_bleu(reference, candidate_b),3)
      ax.text(0,0.5,pred_caption+r"  [ BS_Greedy : "+str(BS_Greedy)+" , BS_Beam3 : "+str(BS_Beam3)+" ]"   ,fontsize=20, fontweight='bold', bbox=dict(facecolor='red', alpha=0.1))
      count_pic+=1
      if count_pic==no_pic:
        break

    return plt.show()




class GooBadCaptions( ):
  def __init__(self):
    self.evaluation=evaluation
    self.test_captions=test_captions
    self.image_folder_path=image_folder_path
    self.test_features=test_features


  def gd_bd_caps(self,n=1000,no_pic=5):
    test_img_ftr=[]
    for i in self.test_features:
      test_img_ftr.append((i,self.test_features[i]))
    self.test_img_ftr=test_img_ftr

    excellent_caption=[]
    good_captions=[]
    bad_captions=[]
    bleu_score_list=[]
    count_good=0
    count_bad=0
    count_medium=0
    count_exc=0

    count=0

    for img_id,img_feature in self.test_img_ftr:
      captions=self.test_captions[img_id]
      reference=[]
      for cap in captions:
        cap_list=cap.split(" ")
        cap_list=cap_list[1:-1]
        reference.append(cap_list)
      # if count==0:
      #   pred_cap=self.evaluation.greedy_search(img_feature.reshape(1,len(img_feature[0])))
      # else:
      pred_cap=self.evaluation.greedy_search(img_feature.reshape(1,len(img_feature)))
      pred_cap=" ".join(pred_cap)
      score=np.round(sentence_bleu(reference,pred_cap),3)
      bleu_score_list.append([img_id,score])
      if score >= 0.8:
        count_exc+=1
        if len(excellent_caption)<no_pic :
          excellent_caption.append([img_id,score,captions,pred_cap])
      if score >= 0.6 and score <0.8  :
        count_good+=1
        if len(good_captions)<no_pic :
          good_captions.append([img_id,score,captions,pred_cap])
      elif score<0.4:
        count_bad+=1
        if len(bad_captions) <no_pic :
          bad_captions.append([img_id,score,captions,pred_cap])
      else:
        count_medium+=1
      # print(len(good_captions),len(bad_captions))
      count+=1

    self.excellent_caption=excellent_caption
    self.good_captions=good_captions
    self.bad_captions=bad_captions
    self.bleu_score_list=bleu_score_list
    self.count_good=count_good
    self.count_bad=count_bad
    self.count_medium=count_medium
    self.count_exc=count_exc


    return self

  def plot_good_bad_caps(self,quality):
    if quality=="exc":
      info=self.excellent_caption
    if quality=="good":
      info=self.good_captions
    if quality=="bad":
      info=self.bad_captions
    count=0
    no_pic=(len(info))
    fig = plt.figure(figsize=(10,20))
    for img_info in info: 
      count+=1
      filename=self.image_folder_path + img_info[0]
      img= load_img(filename,target_size=(224,224,3))
      ax=fig.add_subplot(no_pic,2,count,xticks=[],yticks=[])
      ax.imshow(img)
      
      count+=1
      ax=fig.add_subplot(no_pic,2,count)
      plt.axis("off")
      ax.plot()
      ax.set_xlim(0,1)
      ax.set_ylim(0,8)
      for i,cap in enumerate(img_info[2]):
        cap=" ".join(cap.split(" ")[1:-1])
        ax.text(0,i+3,"true: "+cap,fontsize=25)
      ax.text(0,1.5,"pred: "+img_info[3],fontsize=20, style='italic',bbox=dict(facecolor='red', alpha=0.1))
      ax.text(0,0.5,r"BLEU score: "+str(img_info[1]),fontsize=20,bbox=dict(facecolor='yellow', alpha=0.1))
    return plt.show()

  def get_excellent_caption(self):
    return self.excellent_caption
  def get_good_captions(self):
    return self.good_captions    
  def get_bad_captions(self):
    return self.bad_captions
  def get_bleu_score_list(self):
    return self.bleu_score_list

  def get_quality_count(self):
    return self.count_exc,self.count_good ,self.count_medium,self.count_bad

