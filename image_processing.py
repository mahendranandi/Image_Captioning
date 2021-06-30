'''
here we can do the prerprocessing task of an image to feed it to an ResNet50 model and also we can extract the image features 
of each mage by using the  pretrained CNN model. We use here transfer learning ,so need not train the model. 
'''


from tqdm import tqdm
import argparse
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input        
import numpy as np

# from encoder_model import get_cnn
from keras.applications.resnet50 import ResNet50

# importing other required py files on which this file is depends on
from text_processing import TextProcess



 # this function was for choosing the CNN model we are going to use for feature extraction[ actually a different py file 
  # is made for this but as i yet to add other cnn model i put the function here itslef]
def get_cnn(architecture = 'resnet50'):
  cnn = ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))
  # if architecture == 'resnet18':
  # 	cnn = Resnet(embedding_dim = embedding_dim)
  # elif architecture == 'resnet152':
  # you can add more here 
  return cnn


class ImageProcess():
  def initialize_default (self,options):
    self.image_dir=options.image_folder_path
    self.model=get_cnn(options.architecture)
    return self

  def initialize_custom (self, architecture, image_folder_path):  ## here you may get erro rbecause i changed the no of argument
    self.image_dir=image_folder_path
    self.model=get_cnn()                                           ## dont forget to mention parameter "architecture" if you have used more than one
    return self
    
  def get_image_features(self, dictionary_for_img_name):                    
    out_features={}
    for image_name in tqdm(dictionary_for_img_name):
      img_path=self.image_dir + image_name
      img=image.load_img(img_path,target_size=(224,224))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      features = self.model.predict(x)
      out_features[image_name]=features.squeeze()
    return out_features



  


## you have to run this file if you want to dump(save locally) this files for future easy use

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



  image_process=ImageProcess().initialize_default(options)
  text_preprocess = TextProcess().initialize_default(options).process()

  train_captions=text_preprocess.get_train_captions()
  train_features=image_process.get_image_features(train_captions)
  pickle_out= open(options.pickle_path+"train_features.pickle","wb")                                 
  pickle.dump(train_features,pickle_out)

  validation_captions=text_preprocess.get_validation_captions()
  validation_features=image_process.get_image_features(validation_captions)
  pickle_out= open(options.pickle_path+"validation_features.pickle","wb")                                 
  pickle.dump(validation_features,pickle_out)

  test_captions=text_preprocess.get_test_captions()
  test_features=image_process.get_image_features(test_captions)
  pickle_out= open(options.pickle_path+"test_features.pickle","wb")                                 
  pickle.dump(test_features,pickle_out)








