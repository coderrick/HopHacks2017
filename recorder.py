def record_on_button_press(path):
	import pyaudio
	import wave

	print path
	print "DFASFDAF"	

	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	CHUNK = 2048
	RECORD_SECONDS = 6

	audio = pyaudio.PyAudio()
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
	                rate=RATE, input=True, output=True,
	                frames_per_buffer=CHUNK)
	print "recording..."

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	print "finished recording"
	 
	 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	print "Audio has been terminated"
	waveFile = wave.open(path, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()
	print "Closed"
