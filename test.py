def analyze_audio_response(filename):
	#This function analyzes the yes/no response of a .wav file and returns a 34x (number of time frames depending on how long the response is) matrix
	from pyAudioAnalysis import audioBasicIO
	from pyAudioAnalysis import audioFeatureExtraction
	import matplotlib.pyplot as plt
	[Fs, x] = audioBasicIO.readAudioFile(filename);
	F = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, Fs*0.05, Fs*0.025);
	return F

def get_response_segments(filename):
	#This method takes in the original question and answer recording and saves just the response
	from pyAudioAnalysis import audioBasicIO as aIO
	from pyAudioAnalysis import audioSegmentation as aS
	import scipy.io.wavfile as wavfile
	
	#filename="data/practice_run1.wav"
	[Fs, x] = aIO.readAudioFile(filename)
	segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow = 0.5, Weight = 0.5, plot = False)
	segments = segments[1]
	strOut = "just_response.wav"
		#("{0:s}_" + str(i) + ".wav").format(filename[0:-4], s[0], s[1])
	wavfile.write(strOut, Fs, x[int(Fs * segments[0]):int(Fs * segments[1])])

def get_final_vector_for_one_row(F):
	output_vector=[]
	for i in range(8,18):
		output_vector.extend(F[i])
	return output_vector


def clean_response_from_user(filename):
	#This method takes in the original question and answer recording and saves just the response
	from pyAudioAnalysis import audioBasicIO as aIO
	from pyAudioAnalysis import audioSegmentation as aS
	import scipy.io.wavfile as wavfile
	
	#filename="data/practice_run1.wav"
	[Fs, x] = aIO.readAudioFile(filename)
	segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow = 0.2, Weight = 0.65, plot = False)
	segments = segments[0]
	strOut = "user_response.wav"
		#("{0:s}_" + str(i) + ".wav").format(filename[0:-4], s[0], s[1])
	wavfile.write(strOut, Fs, x[int(Fs * segments[0]):int(Fs * segments[1])])


#get_response_segments("input_sound.wav")
"""
TODO:

-play around with the multiplications of Fs to analysis method to get returned array around 100
-make it in a for loop 
-add rahul's
-add Derek's code
-Get the final vector
-make model with dynamic numpy's
-Rahul connects the method and makes front end look good
-Train on ourselves
-implement the test function into a button
-Dub out

"""

