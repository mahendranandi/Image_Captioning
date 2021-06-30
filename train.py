'''
this is the most important file as you kmow , after preprocessing has been done we train the model
 (defining a model structure) and (using dataloader to load the data in parts so that the RAM should
  not get crashed).

'''

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import pickle
import argparse

# importing other required py files on which this file is depends on
from image_processing import ImageProcess
from text_processing import TextProcess
from dataloader import data_generator
from model import DefineModel
import matplotlib.pyplot as plt  #if you are calling the py file by terminal command the plots here will not be visible , so use notebook , though the codes for notebook is not given here, you have to build it in your own as You have seen it in previous py files where i have put different initialization funtion for a classs to use costomly.




def main(options):
#   if you want to save some time you may use the picjle files directly
#   pickle_in= open(options.pickle_path+"train_encoded_captions.pickle","rb")
#   train_encoded_captions=pickle.load(pickle_in)
#   pickle_in= open(options.pickle_path+"validation_encoded_captions.pickle","rb")
#   validation_encoded_captions=pickle.load(pickle_in)    
  pickle_in= open(options.pickle_path+"train_features.pickle","rb")
  train_features=pickle.load(pickle_in)
  pickle_in= open(options.pickle_path+"validation_features.pickle","rb")
  validation_features=pickle.load(pickle_in)

  image_preprocess = ImageProcess().initialize_default(options)
  text_preprocess = TextProcess().initialize_default(options).process()
  model = DefineModel(options.maximum_length, text_preprocess.get_vocab_size()).make_model().get_model()
  if options.continue_training==True:
    model.load_weights(options.weight_loading_path + 'my_weights')
  train_encoded_captions = text_preprocess.get_train_encoded_captions()
  validation_encoded_captions = text_preprocess.get_validation_encoded_captions()

  # train_features = image_preprocess.get_image_features(text_preprocess.get_train_captions())
  # validation_features = image_preprocess.get_image_features(text_preprocess.get_validation_captions())

  steps=len(train_encoded_captions)//options.train_batch
  vocab_size=text_preprocess.get_vocab_size()
  train_loss_list=[]
  validation_loss_list=[]
  train_accuracy_list=[]
  validation_accuracy_list=[]

  epoch_completed=0
  for i in range(options.epochs):
    train_generator=data_generator(train_encoded_captions,train_features,options.train_batch,vocab_size)
    validation_generator=data_generator(validation_encoded_captions,validation_features,options.valid_batch,vocab_size)
    train_hist=model.fit(train_generator,epochs=1, steps_per_epoch=steps,verbose=1,validation_data=validation_generator)

    train_loss_list.append(train_hist.history['loss'])
    validation_loss_list.append(train_hist.history['val_loss'])
    train_accuracy_list.append(train_hist.history['accuracy'])
    validation_accuracy_list.append(train_hist.history['val_accuracy'])

    epoch_completed+=1
    print("\n****\n****\n****\n      epoch completed = {}   \n****\n****\n****".format( epoch_completed ))
    output_path= options.output_path
    weight_saving_path=options.weight_saving_path
    if (i+1)%options.save_iter==0:
      model.save_weights(weight_saving_path +'my_weights')
      print("\n****\n****\n****\n    weight saved for total {} epohcs  \n****\n****\n****".format(epoch_completed))

      pickle_out= open(output_path+"train_loss_list.pickle","wb")                                 
      pickle.dump(train_loss_list,pickle_out)
      pickle_out.close()
      pickle_out= open(output_path+"validation_loss_list.pickle","wb")                                  
      pickle.dump(validation_loss_list,pickle_out)
      pickle_out.close()
      pickle_out= open(output_path+"train_accuracy_list.pickle","wb")                                
      pickle.dump(train_accuracy_list,pickle_out)
      pickle_out.close()
      pickle_out= open(output_path+"validation_accuracy_list.pickle","wb")                                 
      pickle.dump(validation_accuracy_list,pickle_out)
      pickle_out.close()

    if i>1:   
      if train_hist.history["val_loss"] < min(validation_loss_list[:-1]):
        model.save_weights(options.optimal_weight_saving_path +'my_weights')
        print("optimal weights saved in {}th iteration".format(i+1))

  # Save the weights
  model.save_weights(weight_saving_path +'my_weights')
  print("\n****\n****\n****\n    weight saved for total {} epohcs  \n****\n****\n****".format(epoch_completed))

  pickle_out= open(output_path+"train_loss_list.pickle","wb")                                 
  pickle.dump(train_loss_list,pickle_out)
  pickle_out.close()
  pickle_out= open(output_path+"validation_loss_list.pickle","wb")                                  
  pickle.dump(validation_loss_list,pickle_out)
  pickle_out.close()
  pickle_out= open(output_path+"train_accuracy_list.pickle","wb")                                
  pickle.dump(train_accuracy_list,pickle_out)
  pickle_out.close()
  pickle_out= open(output_path+"validation_accuracy_list.pickle","wb")                                 
  pickle.dump(validation_accuracy_list,pickle_out)
  pickle_out.close()




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-architecture', type = str, default = 'resnet50',help="architecture you want to use as your ENcoder CNN model")
  parser.add_argument('-data_dir', type = str, default = 'data/')  # if you are working with Flickr8k data , you need not give the below path variables.
  parser.add_argument('-image_folder_path', type = str, default = 'Flickr8k_Dataset/Flicker8k_Dataset/',help="the path to the folder where all imges can be found")
  parser.add_argument('-lemma_token_txt', type = str, default = 'Flickr8k_text/Flickr8k.lemma.token.txt',help="path to  the text file where all the captions corresponds to the images are found ")
  parser.add_argument('-train_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.trainImages.txt',help="path to the text file in which all the image id of the train images are given")
  parser.add_argument('-test_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.testImages.txt',help="path to the text file in which all the image id of the test images are given")
  parser.add_argument('-dev_images_txt', type = str, default = 'Flickr8k_text/Flickr_8k.devImages.txt',help="path to the text file in which all the image id of the validation images are given")

  parser.add_argument('-pickle_path', type = str, default = "./pickle_files/",help="this is your choice, where you want to save some files dumped usig pickle")
  parser.add_argument('-output_path', type = str, default = "./output/",help="its your choice where you want to save losse and accuraces of traing")
  parser.add_argument('-weight_saving_path', type = str, default = "./output/LSTM_Model_Weights/",help="your choice, where you wnt to save the model weights")
  parser.add_argument('-optimal_weight_saving_path', type = str, default = "./output15/optimal_weights60/",help="your choice, where you want to save the model at a instance when the  loss is minimum in validation")
  parser.add_argument('-maximum_length', type=int,default=40,help="maximum length of the predected caption")
  parser.add_argument('-weight_loading_path', type = str, default = "./output15/LSTM_Model_Weights60/") # if you had trained the model multiple times with different hyper-parameters
                                                                                                        # then this is the path of the mdel weights you want to use for now

# required parser while training only
  parser.add_argument('-save_iter', type = int, default = 5)
  parser.add_argument('-epochs', type=int,default=50)
  parser.add_argument('-train_batch', type=int,default=60)
  parser.add_argument('-valid_batch', type=int,default=50)
  parser.add_argument('-continue_training',type=bool,default=False)

  options = parser.parse_args()  
  options.image_folder_path=options.data_dir+options.image_folder_path
  options.lemma_token_txt=options.data_dir+options.lemma_token_txt
  options.train_images_txt=options.data_dir+options.train_images_txt
  options.test_images_txt=options.data_dir+options.test_images_txt
  options.dev_images_txt=options.data_dir+options.dev_images_txt

  main(options)