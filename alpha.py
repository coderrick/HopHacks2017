#Start recording
#Save recording
#Convert recording to .wav
#Analyze .wav file
#Create numpy of labels
#Create a huge numpy of all the outputs of the analyzed .wav files
#Feed it into Tensorflow
#Save model and run on trained model


########################################################PREPARE THE DATA FOR MODEL##############################################################
import test
import recorder
import eyetracker as eye
import pickle
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import random
import tensorflow as tf 
import numpy
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import math
import nltk
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#IMPORTANT: replace all zeroes with a count variable

"""
This is the code to get the tone vector
Start recording a .wav file that you save in the current directory and name filename that
"""
def fuckin_hype_train():
	try:
		tone_vector=pickle.load( open( "tone_vector.p", "rb" ) )
		eye_vector=pickle.load( open( "eye_vector.p", "rb" ) )
		#muse_vector=pickle.load( open( "muse_vector.p", "rb" ) )
	except:
		print("Nothing saved yet")



	try:
		eye_vector.append(eye.start(6, "input_sound.wav"))

		#recorder.record_on_button_press("input_sound.wav")
		#Split up response and save the answers

		test.get_response_segments("input_sound.wav")
		#Analyze the input files in the working directory
		#Returns 34 arrays in which we want 8:18, think of a way to make it shorter
		F = test.analyze_audio_response("just_response.wav")
		tone_vector.append(test.get_final_vector_for_one_row(F))

		#if the value is lass than 100, throw exception, if its between 200 and 300, trim down to 200, if over, then throw exception
		#if the value is more than the smallest size, delete random zeroes, if their values are less then add zeroes to the beginning
		#When they're inputting, if its too low, don't use their answer

		print(len(tone_vector))
		print(len(eye_vector))

		pickle.dump( tone_vector, open( "tone_vector.p", "wb" ) )
		pickle.dump( eye_vector, open( "eye_vector.p", "wb" ) )
		#pickle.dump( muse_vector, open( "muse_vector.p", "wb" ) )
	except Exception as e:
		print(len(tone_vector))
		print(len(eye_vector))
		print(e)


#method that does this shit to the data to clean it
#if the length is lass than 100, throw exception, if its between 200 and 300, trim down to 100, if over, then throw exception
#if the length is more than the smallest size, delete random zeroes, if their values are less then add zeroes to the beginning
#When they're inputting, if its too low, don't use their answer

def clean_tone_data(tone):
	temp = tone
	if len(tone)>100:
		temp = tone[:100]
		return temp
	elif len(tone)<100:
		raise Exception("Fucked tone vector")
	else:
		return temp

def clean_eye_data(eye):
	temp = eye
	if len(eye)<50:
		temp=[0]*(50-len(eye))
		temp.extend(eye)
		return temp
	elif len(eye)>50:
		while len(eye)>50:
			random_index=random.randint(0,len(eye)-1)
			if temp[random_index]==0:
				temp.pop(random_index)
		return temp
	else:
		return temp


def analyze_user_answer(filename):
	test.clean_response_from_user(filename)
	F = test.analyze_audio_response("user_response.wav")
	return test.get_final_vector_for_one_row(F)
	#distill and clean the answer .wav file
	#run analysis and return the array

def empty_pickles():
	tone_vector=[]
	eye_vector=[]
	pickle.dump( tone_vector, open( "tone_vector.p", "wb" ) )
	pickle.dump( eye_vector, open( "eye_vector.p", "wb" ) )

def combine_vectors_for_final(tone,eye):
	temp=numpy.concatenate([tone , eye])
	return temp



#cleans data
# for i in range(0,10):
# 	tone_vector[i]=clean_tone_data(tone_vector[i])
# 	eye_vector[i]=clean_eye_data(eye_vector[i])

# pickle.dump( tone_vector, open( "tone_vector.p", "wb" ) )
# pickle.dump( eye_vector, open( "eye_vector.p", "wb" ) )



########################################################SETTING UP/TRAINING/SAVING THE MODEL##############################################################
def preproc(unclean_batch_x):
    #Convert values to range 0-1
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(rng,batch_size, dataset_length, dataset_name):
    #Create batch with random samples and return appropriate format
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('trainingDataNumpy')[[batch_mask]].reshape(-1, input_num_units)
    
    batch_y = eval('trainingLabelsNumpy')[[batch_mask]].reshape(-1,output_num_units)
        
    return batch_x, batch_y

#This code is for loading and using the model on never before seen data

def predict_new_data(filename):
	#Take in tone and eye data
	#create a vector after cleaning
	#fix with numpy and reshape and call it inputvector
	
	eye_vector = clean_eye_data(eye.start_alone(6))
	test.clean_response_from_user(filename)
	F = test.analyze_audio_response("user_response.wav")
	tone_vector = clean_tone_data(test.get_final_vector_for_one_row(F))
	

	tempVector=combine_vectors_for_final(tone_vector,eye_vector)

	inputVector = numpy.float32(tempVector).reshape(1,150)
	with tf.Session() as session:
		saver.restore(session, "model")
		#feed vector into model
		feed_dict = {x: inputVector}
		classification = session.run(activation_OP, feed_dict)
		#Output predicted value
		if int(classification[0][0]) == 1:
			return True
		else:
			return False


#predict_new_data("user_final_input.wav")

def get_additional_input_from_user(filename):
	try:
		tone_vector=pickle.load( open( "tone_vector_from_user.p", "rb" ) )
		eye_vector=pickle.load( open( "eye_vector_from_user.p", "rb" ) )
	except:
		tone_vector=[]
		eye_vector=[]
		print("havent created it yet")

	eye_vector.append(clean_eye_data(eye.start_alone(6)))
	print("1")
	#test.clean_response_from_user(filename)
	F = test.analyze_audio_response(filename)
	print("2")

	tone_vector.append(clean_tone_data(test.get_final_vector_for_one_row(F)))
	print("3")


	pickle.dump( tone_vector, open( "tone_vector_from_user.p", "wb" ) )
	pickle.dump( eye_vector, open( "eye_vector_from_user.p", "wb" ) )


def train_model_with_new_input():
	label_vector=[1,0,0,1,1,0,1,0,1,0]
	tone_vector=pickle.load( open( "tone_vector_from_user.p", "rb" ) )
	eye_vector=pickle.load( open( "eye_vector_from_user.p", "rb" ) )

	inputVector=[]
	for i in range(0,len(tone_vector)):
		inputVector.append(numpy.concatenate([tone_vector[i],eye_vector[i]]))



	trainingDataNumpy = numpy.float32(inputVector)
	trainingLabelsNumpy = numpy.float32(label_vector)
	#Setting up
	seed = 128
	rng = numpy.random.RandomState(seed)

	# number of neurons in each layer
	input_num_units = NUM_COLS
	hidden_num_units = 100
	output_num_units = 1


	x = tf.placeholder(tf.float32, [None, input_num_units])
	y = tf.placeholder(tf.float32, [None, output_num_units])

	# set remaining variables
	epochs = 200
	batch_size = 5
	learning_rate = 0.01

	weights = {
	    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
	}

	biases = {
	    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
	}
	#set up layers
	hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
	hidden_layer = tf.nn.relu(hidden_layer)
	output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
	activation_OP = tf.nn.sigmoid(output_layer, name="activation")

	#define cost
	cost = tf.nn.l2_loss(activation_OP-y, name="squared_error_cost")

	#adam gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


	#initialize variables
	init=tf.global_variables_initializer()

	with tf.Session() as session:
		saver.restore(session, "model")
		session.run(init)
		for epoch in range(epochs):
			avg_cost = 0
			total_batch = int(trainingDataNumpy.shape[0]/batch_size)
			for i in range(total_batch):
				batch_x, batch_y = batch_creator(rng, batch_size, trainingDataNumpy.shape[0],'train')
				_, c, output= session.run([optimizer, cost,activation_OP], feed_dict = {x: batch_x, y: batch_y})
				avg_cost += c / total_batch
				print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

		
		print("\nTraining complete!")
		saver.save(session, "model")

if __name__ == '__main__':

	NUM_COLS=150
	seed = 128
	rng = numpy.random.RandomState(seed)

	# number of neurons in each layer
	input_num_units = NUM_COLS
	hidden_num_units = 100
	output_num_units = 1


	x = tf.placeholder(tf.float32, [None, input_num_units])
	y = tf.placeholder(tf.float32, [None, output_num_units])

	# set remaining variables
	epochs = 200
	batch_size = 5
	learning_rate = 0.01

	weights = {
	    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
	}

	biases = {
	    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
	}
	#set up layers
	hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
	hidden_layer = tf.nn.relu(hidden_layer)
	output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
	activation_OP = tf.nn.sigmoid(output_layer, name="activation")

	#define cost
	cost = tf.nn.l2_loss(activation_OP-y, name="squared_error_cost")

	#adam gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


	#initialize variables
	init=tf.global_variables_initializer()
	saver = tf.train.Saver()

	#get_additional_input_from_user("first-test.wav")
	train_model_with_new_input()