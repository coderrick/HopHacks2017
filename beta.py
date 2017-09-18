#This code is for loading and using the model on never before seen data
import alpha


def preproc(unclean_batch_x):
    #Convert values to range 0-1
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    #Create batch with random samples and return appropriate format
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('trainingDataNumpy')[[batch_mask]].reshape(-1, input_num_units)
    
    batch_y = eval('trainingLabelsNumpy')[[batch_mask]].reshape(-1,output_num_units)
        
    return batch_x, batch_y


def predict_new_data(filename):
	#Take in tone and eye data
	#create a vector after cleaning
	#fix with numpy and reshape and call it inputvector
	eye_vector = clean_voice_data(eye.start_alone(6))
	tone_vector = clean_tone_data(analyze_user_answer(filename))
	

	inputVector=combine_vectors_for_final(tone_vector,eye_vector)
	inputVector = numpy.float32(inputVector).reshape(1,150)
	print(inputVector)
	with tf.Session() as session:
		saver.restore(session, "model")
		#feed vector into model
		feed_dict = {x: inputVector}
		classification = session.run(activation_OP, feed_dict)
		#Output predicted value
		print(classification)


#predict_new_data("user_final_input.wav")

def get_additional_input_from_user(filename):
	try:
		tone_vector=pickle.load( open( "tone_vector_from_user.p", "rb" ) )
		eye_vector=pickle.load( open( "eye_vector_from_user.p", "rb" ) )
	except:
		tone_vector=[]
		eye_vector=[]
		print("havent created it yet")

	eye_vector.append(clean_voice_data(eye.start_alone(6)))
	tone_vector.append(clean_tone_data(analyze_user_answer(filename)))

	pickle.dump( tone_vector, open( "tone_vector_from_user.p", "wb" ) )
	pickle.dump( eye_vector, open( "eye_vector_from_user.p", "wb" ) )


def train_model_with_new_input():
	label_vector=[1,0,0,1,1,0,1,0,1,0]
	tone_vector=pickle.load( open( "tone_vector_from_user.p", "rb" ) )
	eye_vector=pickle.load( open( "eye_vector_from_user.p", "rb" ) )
	inputVector=combine_vectors_for_final(tone_vector,eye_vector)


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
		sess.run(init)
    
	    for epoch in range(epochs):
	        avg_cost = 0
	        total_batch = int(trainingDataNumpy.shape[0]/batch_size)
	        for i in range(total_batch):
	            batch_x, batch_y = batch_creator(batch_size, trainingDataNumpy.shape[0],'train')
	            _, c, output= sess.run([optimizer, cost,activation_OP], feed_dict = {x: batch_x, y: batch_y})
	            avg_cost += c / total_batch
	        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
	    
	    print("\nTraining complete!")
	    saver.save(sess, "model")







