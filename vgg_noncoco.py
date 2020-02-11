# ~~ vgg_noncoco.py ~~ 
# Author: Nanditha Dissanayake
# Contact: nandithad@analyticalsolutions.net
# 
# A simple transfer learning based neural network to identify armed individuals in urbun environments 
# using input from RGB camera data. A feature vector is created for the input RGB image by inferencing
# using VGG16 model pretrained on ImageNet. The feature vector is used as input to a single hidden layer
# neural network and trained for binary classification. If the input image contains an invidual armed with 
# a weapon the image is classified as a Treat. If not, the image is classified as a Background. 
#
# Usage:
# 
# To preprocess input data and train the model: 
#   1. Create a directory called: 'train_noncoco'
#   2. Download training images in the above directory
#   3. Create annotation file using LabelImg (https://github.com/tzutalin/labelIm://github.com/tzutalin/labelImg) to annotate the images
#   4. Use the xml_to_csv.py to convert the xml annotation file to CSV.
#   5. Use the set_data() function in this script compile the training images and labels in to a binary data file.
#   6. Use the train_model() to train a neural network using the above binary data.
# Inference:
#   1. One trained the model and the weights can be loded using vgg_gun_model_noncoco.h5 data file
#   2. Use the single_pic_inferencing() function to classify an individual image.
###########################################################################################################################################
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import os
import pickle

def set_data():
    
        #set the training and test data set using the images folder data
        img_path = 'train_noncoco'
        files_path = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
	model = VGG16(weights='imagenet', include_top=False) #this model captures the featureas
	import pandas as pd
	l = pd.read_csv('labels_noncoco.csv')


	def label(x):
		if x == 'gun':
			return 1;
		else:
			return 0;

	feature_list = []
	label_list = [ label(x) for x in l['class']]
	files_path = [ f for f in l['filename']]

	files_path = files_path
	label_list = label_list

	for f in files_path:
		img = image.load_img(img_path + "/" + f, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		features = model.predict(x)
		feature_list.append(features)


	#reshape to get all input features per image in one vector
	(a,b,c,d) = feature_list[0].shape
	feature_size = a*b*c*d

	np_features=np.zeros((len(feature_list),feature_size))
	for i in range(len(feature_list)):
		feat=feature_list[i]
		reshaped_feat=feat.reshape(1,-1)
		np_features[i]=reshaped_feat

	X = np_features
	print(X.shape)

	from sklearn.preprocessing import LabelBinarizer
	mlb=LabelBinarizer()
	Y=mlb.fit_transform(label_list)

	print(Y.shape)

	#save the data
	import pickle
	gun_data = (X,Y)
	f8 = open('gun_detection_data_noncoco.pckl','wb')
	pickle.dump(gun_data,f8)
	f8.close()

	return



def train_model():
	#read data and train
	f8=open('gun_detection_data_noncoco.pckl','rb')
	gun_data=pickle.load(f8)
	f8.close()

	(X,Y) = gun_data

	#randomize the train and test data set
	mask = np.random.rand(len(X)) < 0.8
	X_train = X[mask]
	X_test = X[~mask]
	Y_train = Y[mask]
	Y_test = Y[~mask]

	print(Y_train)

	#get a simple network using keras
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Activation
	from tensorflow.keras import optimizers

	model_visual = Sequential([
		Dense(1024,input_shape=(25088,)),
		Activation('relu'),
		Dense(256),
		Activation('relu'),
		Dense(1),
		Activation('sigmoid'),])

	opt =  optimizers.RMSprop(lr=0.001, decay = 1e-6)

	model_visual.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

	model_visual.fit(X_train,Y_train,epochs=10, batch_size=1, verbose=1)

	model_visual.save('vgg_gun_model_noncoco.h5')

	Y_preds = model_visual.predict(X_test)

	print(Y_preds)

	return

#set_data()
#train_model()

def inference():
	model = tf.keras.models.load_model('vgg_gun_model_noncoco.h5')
	#model = tf.keras.models.load_model('vgg_gun_model.h5')
	(X,Y) = pickle.load(open('gun_detection_data_noncoco.pckl','rb'))
	#(X,Y) = pickle.load(open('gun_detection_data.pckl','rb'))
	Y_preds = model.predict(X)
	from sklearn.metrics import accuracy_score
	print('accuracy score:%f'%accuracy_score(Y,Y_preds))
	return

#inference()


def single_pic_inference(img):
	#vgg features for the image
	vggmodel = VGG16(weights='imagenet', include_top=False) #this model captures the featureas
	img = image.load_img(img,target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = vggmodel.predict(x)
	X = features.reshape(1,-1)
	model = tf.keras.models.load_model('vgg_gun_model_noncoco.h5')
	Y_preds = model.predict(X)
	
	if Y_preds == 1.0:
		print("this is a threat")
	else:
		print("background")

	return


import sys

if len(sys.argv) == 1:
	print("enter a file name:")
	quit()
else:
	img = str(sys.argv[1])
	single_pic_inference(img)
	
