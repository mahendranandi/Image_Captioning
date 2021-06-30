'''


'''

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet152 import Resnet152
from keras.applications.resnet18 import resnet18
from keras.applications.alexnet import alexnet
from keras.applications.dense import DenseNet
from keras.applications.inception import inception

def get_cnn(architecture = 'resnet50'):
  cnn = ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))

	if architecture == 'resnet50':
		cnn = ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))
	# if architecture == 'resnet18':
	# 	cnn = resnet18(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3),embedding_dim = embedding_dim)
	# elif architecture == 'resnet152':
	# 	cnn = Resnet152(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3),embedding_dim = embedding_dim)
	# elif architecture == 'alexnet':
	# 	cnn = alexnet(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3),embedding_dim = embedding_dim) 
	# elif architecture == 'inception':
	# 	cnn = inception(embedding_dim = embedding_dim) 
	# elif architecture == 'dense':
	# 	cnn = DenseNet(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3),embedding_dim = embedding_dim) 
	return cnn