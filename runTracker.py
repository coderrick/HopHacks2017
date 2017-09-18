def get_eye_track_data(seconds):
	import eyetracker as eye
	import time as t


	start_time = t.time()

	answers = eye.start_alone(seconds)

	return answers

get_eye_track_data(10000)