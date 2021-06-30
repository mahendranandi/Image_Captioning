'''
this is totally for the visualization.ipynb file..where to make that file clean and understandabe i moved the functions here , 
This file can be run after the text_processing and image_rocessing is done. actually the goal is to do an EDA analysis before proceeding futher heavy duty stuff. It is good practice to fdo EDA before any analysis, it will help ou infering many thing as well as keep you in a right thinking and way.
If I forget to add the title for the plots , please make that complete, without the title other can't understand what you tried to visualiza. 

'''

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import OrderedDict
from keras.preprocessing.image import load_img, img_to_array
from IPython.display import display
import cv2
import pandas as pd
import pickle
import numpy as np
from collections import Counter

from text_processing import TextProcess
from image_processing import ImageProcess


class Visualization():
    def __init__(self,all_words,img_folder_path):
        self.all_words=all_words
        self.img_folder_path=img_folder_path
        
        clustered_pic = OrderedDict()
        clustered_pic["purple"]  = [533,935,907,30,207]
        clustered_pic["red"]     = [841,22,759,53,810]
        clustered_pic["black"]     = [446,353,980,885,42]
        clustered_pic["blue"]    = [694,761,192,7,311]
        clustered_pic["green"]   = [278,659,91,560,173]
        clustered_pic["magenta"] = [494,448,400,104,843]
        clustered_pic["yellow"]  = [930,258,432,834,488]
        self.clustered_pic=clustered_pic
        
        list_all_wrd=all_words.split(" ")
        lst_of_counts=Counter(Counter(list_all_wrd)).most_common()
        words=[]
        count=[]
        for (x,y) in lst_of_counts:
            words.append(x)
            count.append(y)
        self.words=words
        self.count=count
        

    def plot_word_hist(self,top=True,top_n=50):
        count_df=pd.DataFrame({"word":self.words,"count":self.count})
        if top==True:
            df=count_df.iloc[:top_n,:]
            title="The top 50 most frequently appearing words"
        else:
            df=count_df.iloc[-top_n:,:]
            title="The top 50 least frequently appearing words"
        plt.figure(figsize=(20,3))
        plt.bar(df.index,df["count"])
        plt.yticks(fontsize=20)
        plt.xticks(df.index,df["word"],rotation=90,fontsize=20)
        plt.title(title,fontsize=20)
        plt.show()

    def print_counts(self):
        for i in range(1,6,1):
            print("\n\nThere are {} words which are present {} time(s) in the whole captions\n\n".format(self.count.count(i),i))
        print("And there are {} words which are present more than 5 but less than 100 times in the whole captions\n".format(sum([self.count.count(i) for i in range(6,100,1)])))
        print("And there are {} words which are present more or equal 100 times in the whole captions\n".format(sum([self.count.count(100+i) for i in range(32970)])))

    def disply_img_and_5captions(self,img_captions,no_pic=4,no_pixel=224):
        target_size = (no_pixel,no_pixel,3)
        count=0
        count_pic=0
        fig=plt.figure(figsize=(10,20),edgecolor = 'red',frameon=False,linewidth=12)
        for img_id in img_captions:
            count+=1
            filename=self.img_folder_path + img_id
            captions= img_captions[img_id]
            img=load_img(filename , target_size=target_size)

            ax=fig.add_subplot(no_pic,2,count,xticks=[],yticks=[])
            ax.imshow(img)
            count+=1
            ax = fig.add_subplot(no_pic,2,count)
            plt.axis("off")
            ax.plot()
            ax.set_xlim(0,1)
            ax.set_ylim(0,len(captions))
            for i, caption in enumerate(captions):
                caption=" ".join(caption.split(" ")[1:-1])
                ax.text(0,i+0.5,r""+caption ,fontsize = 20, style='italic', bbox=dict(facecolor='yellow', alpha=0.3))
            count_pic+=1
            if count_pic==no_pic:
                break
        return plt.show()


    def plot_PCA(self,test_features):
        values = np.array(list(test_features.values()))
        pca = PCA(n_components=2)
        PCA_points = pca.fit_transform(values)

        fig, ax = plt.subplots(figsize=(15,15))
        ax.scatter(PCA_points[:,0],PCA_points[:,1],c="white")

        for point in range(PCA_points.shape[0]):
            ax.annotate(point,PCA_points[point,:],color="black",alpha=0.5)
        for color, points in self.clustered_pic.items():
            for point in points:
                ax.annotate(point,PCA_points[point,:],color=color)
        ax.set_xlabel("PCA_1",fontsize=30)
        ax.set_ylabel("PCA_2",fontsize=30)
        return plt.show()


    def show_clustered_pic_example(self,test_features):
        fig = plt.figure(figsize=(30,30))
        count = 1
        target_size=(224,224,3)
        for color, irows in self.clustered_pic.items():
            for ivec in irows:
                name = list(test_features.keys())[ivec]
                filename = self.img_folder_path + name
                image = load_img(filename, target_size=target_size)

                ax = fig.add_subplot(len(self.clustered_pic),5,count,
                                xticks=[],yticks=[])
                count += 1
                plt.imshow(image)
                plt.title("{} ({})".format(ivec,color),fontsize=25)
        return plt.show()

