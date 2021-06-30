'''
In this file you can find a single class named TextProcess , inside which many functions are defined 
The requirements for the class to build are some inbuild lybraries and lacal paths and custom variables 
For the lybraries we can directly import them, but for the paths and the variables we take help of arg-parser[know about argparse]. 

when you will call the function the libraries will be imported automatically and the variavles will be 
assigned automatically by the argparse [ here I named it 'options'] 

Improvement Can be done by You : here i have used argparse in a bad way, so you learn about it and try to use another seperate class for that 
                                 so , wherever in any py file you need it you can call . It will make the py file clear as well as understandable ans short 


'''
#importing the libraries
import pandas as pd  #we need the dataframe 
import argparse  # to use options for variable to assign from terminal
import pickle # to dump some files we need this 
from tqdm import tqdm # this is just to show the progress bar 
from keras.preprocessing.sequence import pad_sequences # for padding the sequences to an equal length 
import numpy as np # for mathical calculations

# importing others py files 
#[ as in the chain of py files this is the first file so we don't need to import those ]







#    here I have used 2 different initialization function one is default and another is costom . 
#    the default one is for using from the terminal using argparser   but the second one is to run from the jupyter notebook

class TextProcess():
  def initialize_default (self,options):
    self.all_image_captions_path = options.lemma_token_txt
    self.train_image_id_path =options.train_images_txt
    self.test_image_id_path =options.test_images_txt
    self.valid_image_id_path =options.dev_images_txt
    self.max_length=options.maximum_length

    self.image_captions=pd.read_csv(self.all_image_captions_path,sep='\t',names=["img_id","img_caption"])
    self.train_image_names=pd.read_csv(self.train_image_id_path,names=["img_id"])
    self.test_image_names=pd.read_csv(self.test_image_id_path,names=["img_id"])
    self.val_image_names=pd.read_csv(self.valid_image_id_path,names=["img_id"])

    return self
    

  def initialize_custom (self,lemma_token_txt_path, train_images_txt_path, test_images_txt_path, dev_images_txt_path,maximum_length):
    self.all_image_captions_path = lemma_token_txt_path
    self.train_image_id_path = train_images_txt_path
    self.test_image_id_path = test_images_txt_path
    self.valid_image_id_path = dev_images_txt_path

    self.image_captions=pd.read_csv(self.all_image_captions_path,sep='\t',names=["img_id","img_caption"])
    self.train_image_names=pd.read_csv(self.train_image_id_path,names=["img_id"])
    self.test_image_names=pd.read_csv(self.test_image_id_path,names=["img_id"])
    self.val_image_names=pd.read_csv(self.valid_image_id_path,names=["img_id"])
    self.max_length=maximum_length

    return self

  def process(self):
    self.image_captions["img_id"] = self.image_captions["img_id"].map(lambda x: x[:len(x)-2] )
    self.image_captions["img_caption"]=self.image_captions["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")
    TTV_image_names=[self.train_image_names,self.test_image_names,self.val_image_names]
    train_captions={}
    test_captions={}
    validation_captions={}
    TTV_captions_dictionaries=[train_captions,test_captions,validation_captions]

    for j in range(3):
      ttv_image_names=TTV_image_names[j]
      ttv_captions_dictionaries=TTV_captions_dictionaries[j]

      for i in tqdm(range(len(ttv_image_names))):
        l=[caption for caption in(self.image_captions[self.image_captions["img_id"]==ttv_image_names["img_id"].iloc[i]].img_caption)]
        ttv_captions_dictionaries[ttv_image_names["img_id"].iloc[i]]=l
    
    self.train_captions=train_captions
    self.test_captions=test_captions
    self.validation_captions=validation_captions

    # here we are creating our required vocabulary, word to indices[ ,i.e, each unique word after geting sorted should have an unique index,
    # so we are trying to represent the word mathematically by using indices instead of the words], 
    # and indices to words dictionaries, and we also keep a singlstring containging all the words in the whole bag of captions 
    all_train_captions=[]
    for img_id in tqdm(train_captions):
      for captions in train_captions[img_id]:
        all_train_captions.append(captions)
    all_words=" ".join(all_train_captions)
    unique_words=list(sorted(set(all_words.strip().split(" "))))
    vocab_size=len(unique_words)+1

    self.unique_words=unique_words
    self.vocab_size=vocab_size
    self.all_words=all_words
    
    words_to_indices={val:index+1 for index ,val in enumerate(unique_words)}
    words_to_indices["Unk"]=0
    indices_to_words={index+1:val for index, val in enumerate(unique_words)}
    indices_to_words[0]="Unk"

    self.words_to_indices=words_to_indices
    self.indices_to_words=indices_to_words
    
    TTV=[train_captions,validation_captions,test_captions]
    train_encoded_captions={}
    test_encoded_captions={}
    validation_encoded_captions={}
    TTV_encoded_dictionaries=[train_encoded_captions,validation_encoded_captions,test_encoded_captions]
    
    for j in range(2):        # no need to get test encoded captions as they are not being gone to feed in the network
      ttv=TTV[j]
      ttv_encded_dict=TTV_encoded_dictionaries[j]
      
      for img_id in tqdm(ttv):
        ttv_encded_dict[img_id]=[]
        for i in range(5):
          ttv_encded_dict[img_id].append([words_to_indices[s] for s in ttv[img_id][i].split(" ") if s in words_to_indices ])
      
      for img_id in tqdm(ttv_encded_dict):
        ttv_encded_dict[img_id]=pad_sequences(ttv_encded_dict[img_id], maxlen=self.max_length, padding='post')
    
    self.train_encoded_captions = train_encoded_captions
    self.test_encoded_captions = test_encoded_captions
    self.validation_encoded_captions = validation_encoded_captions

    return self

  def get_train_encoded_captions(self): # this definitions or functions are easy ways to get those saved variables.
    return self.train_encoded_captions

  def get_test_encoded_captions(self):
    return self.test_encoded_captions

  def get_validation_encoded_captions(self):
    return self.validation_encoded_captions

  def get_w2i(self):
    return self.words_to_indices

  def get_i2w(self):
    return self.indices_to_words

  def get_vocab_size(self):
    return self.vocab_size

  def get_train_captions(self):
    return self.train_captions

  def get_test_captions(self):
    return self.test_captions

  def get_validation_captions(self):
    return self.validation_captions
    
  def get_all_words(self):
    return self.all_words    
    

## you have to run this file if you want to dump(save locally) this files for future easy use , ANd clearly if you want
  # to run it seperately in comand line { !python text_processing.py } then the below function has to be present
  # like this--(you will see this function repeatedly , so dont be confused it is just ot run the single file ,
  #  if it is depending on another files tooo, then all the files will be run in a chain and required manner .
if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-data_dir', type = str, default = 'data/')
  parser.add_argument('-image_folder_path', type = str, default = 'Flickr8k_Dataset/Flicker8k_Dataset/')
  parser.add_argument('-lemma_token_txt', type = str, default = 'Flickr8k_text/Flickr8k.lemma.token.txt')
  parser.add_argument('-train_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.trainImages.txt')
  parser.add_argument('-test_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.testImages.txt')
  parser.add_argument('-dev_images_txt', type = str, default ='Flickr8k_text/Flickr_8k.devImages.txt')

  parser.add_argument('-architecture', type = str, default = 'resnet50') 
  parser.add_argument('-pickle_path', type = str, default = "./pickle_files/")
  parser.add_argument('-output_path', type = str, default = "./output/")
  parser.add_argument('-maximum_length', type=int,default=40)
  
# required parser while training only
  parser.add_argument('-save_iter', type = int, default = 5)
  parser.add_argument('-epochs', type=int,default=50)
  parser.add_argument('-train_batch', type=int,default=60)
  parser.add_argument('-valid_batch', type=int,default=50)

  options = parser.parse_args()  
  options.image_folder_path=options.data_dir+options.image_folder_path
  options.lemma_token_txt=options.data_dir+options.lemma_token_txt
  options.train_images_txt=options.data_dir+options.train_images_txt
  options.test_images_txt=options.data_dir+options.test_images_txt
  options.dev_images_txt=options.data_dir+options.dev_images_txt



  text_preprocess = TextProcess().initialize_default(options).process()

  train_captions=text_preprocess.get_train_captions()
  test_captions=text_preprocess.get_test_captions()
  validation_captions=text_preprocess.get_validation_captions()
  pickle_out= open(options.pickle_path+"train_captions.pickle","wb")                                 
  pickle.dump(train_captions,pickle_out)
  pickle_out= open(options.pickle_path+"test_captions.pickle","wb")                                 
  pickle.dump(test_captions,pickle_out)
  pickle_out= open(options.pickle_path+"validation_captions.pickle","wb")                                 
  pickle.dump(validation_captions,pickle_out)
  
  all_words=text_preprocess.get_all_words()
  pickle_out= open(options.pickle_path+"all_words.pickle","wb")                                 
  pickle.dump(all_words,pickle_out)

  words_to_indices=text_preprocess.get_w2i()
  indices_to_words=text_preprocess.get_i2w()
  pickle_out= open(options.pickle_path+"words_to_indices.pickle","wb")                                 
  pickle.dump(words_to_indices,pickle_out)
  pickle_out= open(options.pickle_path+"indices_to_words.pickle","wb")                                 
  pickle.dump(indices_to_words,pickle_out)


  train_encoded_captions =text_preprocess.get_train_encoded_captions()
  validation_encoded_captions =text_preprocess.get_validation_encoded_captions()
  pickle_out= open(options.pickle_path+"train_encoded_captions.pickle","wb")                                 
  pickle.dump(train_encoded_captions,pickle_out)
  pickle_out= open(options.pickle_path+"validation_encoded_captions.pickle","wb")                                 
  pickle.dump(validation_encoded_captions,pickle_out)








