#Identify pupils. Based on beta 1

import numpy as np
import cv2
import time
import threading
import recorder

start_time = time.time()
prev_time = 0

STOP = False
	

eyeMovement = []

def showImage(tempVar, pupilFrame):
	for i in tempVar[0,:]:
			cv2.circle(pupilFrame,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(pupilFrame,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow("frame", pupilFrame)

def getDarkness(img):
	data = np.asarray(img)
	darkness = 0
	for row in range(0, len(data)):
		for col in range(0, len(data[0])):
			if(data[row][col] < 90):
				darkness += 1
	return darkness

def process(quadrants):
	gradients = []
	for row in quadrants:
		left = getDarkness(row[0])
		right = getDarkness(row[1])
		gradients.append([left, right])
	return gradients

def replace(img, curr, max):
	data = np.asarray(img)
	color = ((curr+0.0)/(max+0.0))*255.0
	for row in range(0, len(data)):
		for col in range(0, len(data[0])):
			data[row][col] = color
	return data

def end():
	global eyeMovement
	temp = eyeMovement
	eyeMovement = []
	cv2.destroyAllWindows()
	return temp

def start(seconds, fileName):
	thread = threading.Thread(target=recorder.record_on_button_press, args=("input_sound.wav",))
	thread.daemon = True                            # Daemonize thread
	thread.start()
	cap = cv2.VideoCapture(0) 	#640,480
	w = 640
	h = 480
	testSize = 15
	curr_time = 0
	count = 0
	baseline = np.array([[0,0],[0,0]])
	while(cap.isOpened() and curr_time < seconds):
		ret, frame = cap.read()
		#print curr_time
		#print "-------------"
		if ret==True:
			#downsample
			#frameD = cv2.pyrDown(cv2.pyrDown(frame))
			#frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)
	
			#detect face
			frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
			frame = frame[0:700, 200:1000]
			faces = cv2.CascadeClassifier('haarcascade_eye.xml')
			
			
			detected = faces.detectMultiScale(frame, 1.3, 5)
			#faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# 		detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
			
			pupilFrame = frame
			pupilO = frame
			#
			windowClose = np.ones((5,5),np.uint8)
			windowOpen = np.ones((2,2),np.uint8)
			windowErode = np.ones((2,2),np.uint8)

			# draw square
			for (x,y,w,h) in detected:
				cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)
				cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)
				cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
				cv2.rectangle(frame, (x+(w/5), y+(h)), (x+(w*4/5), y+(h*3/4)), (0,0,255), 1)
				#cv2.imshow('frame2',frame)
				pupilFrame = cv2.equalizeHist(frame[int(y+(h*.25)):(y+h), x:(x+w)])
				cv2.line(frame, (x+(w/2),0), (x+(w/2),y+h), (0,0,255),1)
				cv2.line(frame, (0,y+(h/2)), (x+w,y+(h/2)), (0,0,255),1)
				eye = frame[(y+(h/3)):(y+(h*2/3)), (x+(w/4)):(x+(w*3/4))]
				eye = cv2.equalizeHist(eye)
				h, w = eye.shape
				eye = cv2.flip(eye, 1)
				cv2.imshow("Eye", eye)
				topleft = eye[0:h/2, 0:w/2]
				topright = eye[0:h/2, w/2:w]
				bottomleft = eye[h/2:h, 0:w/2]
				bottomright = eye[h/2:h, w/2:w]
				#cv2.imshow("eye", eye)
				cv2.moveWindow("Eye", 100, 100)
# 				print eye[0, 0]
# 				print eye[h-1,w-1]
# 				print eye[0,w-1]
# 				print eye[h-1,0]
# 				print "------------"
				
				startx = 0
				starty = 0
				# cv2.imshow("top left", topleft)
				# cv2.moveWindow("top left", startx, starty)
				# cv2.imshow("top right", topright)
				# cv2.moveWindow("top right", w/2, starty)
				# cv2.imshow("bottom left", bottomleft)
				# cv2.moveWindow("bottom left", startx, h*2)
				# cv2.imshow("bottom right", bottomright)
				# cv2.moveWindow("bottom right", w/2, h*2)
				quadrants = [[topleft, topright],
							 	[bottomleft, bottomright]]
				if(count<testSize):
					baseline += np.array(process(quadrants))
				elif(count == testSize):
					baseline/=testSize
					print "Done with Baseline"
				else:
					difference = np.array(process(quadrants))-np.array(baseline)
					#difference = np.absolute(difference)
					max = np.amax(difference)
					index = np.where(difference==max)
					difference = list(difference)
					index = list(index)
					max = difference[index[0][0]][index[1][0]]
					#print max
					if(max<75):
						#print "Straight Ahead"
						eyeMovement.append(0)
						continue
					for j in range(0, len(quadrants)):
						for k in range(0, len(quadrants[0])):
							quadrants[j][k] = replace(quadrants[j][k], difference[j][k], max)
					if(index[0].any() == 0):
						if(index[1].any() == 1):
							eyeMovement.append(2)
							#print "Top Right"
							k = 1
						else:
							k = 1
							eyeMovement.append(1)
							#print "Top Left"
					else:
						if(index[1].any() == 1):
							k = 1
							eyeMovement.append(3)
							#print "Bottom Right"
						else:
							k = 1
							eyeMovement.append(4)
							#print "Bottom Left"
				count+=1
				h*=2
				starty+=h
				starty+=(h/2)
				# cv2.imshow("top left1", quadrants[0][0])
				# cv2.moveWindow("top left1", startx, starty)
				# cv2.imshow("top right1", quadrants[0][1])
				# cv2.moveWindow("top right1", w/2, starty)
				# cv2.imshow("bottom left1", quadrants[1][0])
				# cv2.moveWindow("bottom left1", startx, h*2)
				# cv2.imshow("bottom right1", quadrants[1][1])
				# cv2.moveWindow("bottom right1", w/2, h*2)
		curr_time = time.time() - start_time
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	thread.join()
	return eyeMovement

def start_alone(seconds):
	cap = cv2.VideoCapture(0) 	#640,480
	w = 640
	h = 480
	testSize = 15
	curr_time = 0
	count = 0
	baseline = np.array([[0,0],[0,0]])
	while(cap.isOpened() and curr_time < seconds):
		print curr_time
		ret, frame = cap.read()
		#print curr_time
		#print "-------------"
		if ret==True:
			#downsample
			#frameD = cv2.pyrDown(cv2.pyrDown(frame))
			#frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)
	
			#detect face
			frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
			frame = frame[0:700, 200:1000]
			faces = cv2.CascadeClassifier('haarcascade_eye.xml')
			
			
			detected = faces.detectMultiScale(frame, 1.3, 5)
			#faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# 		detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
			
			pupilFrame = frame
			pupilO = frame
			#
			windowClose = np.ones((5,5),np.uint8)
			windowOpen = np.ones((2,2),np.uint8)
			windowErode = np.ones((2,2),np.uint8)

			# draw square
			for (x,y,w,h) in detected:
				cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)
				cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)
				cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
				cv2.rectangle(frame, (x+(w/5), y+(h)), (x+(w*4/5), y+(h*3/4)), (0,0,255), 1)
				cv2.imshow('frame2',frame)
				pupilFrame = cv2.equalizeHist(frame[int(y+(h*.25)):(y+h), x:(x+w)])
				cv2.line(frame, (x+(w/2),0), (x+(w/2),y+h), (0,0,255),1)
				cv2.line(frame, (0,y+(h/2)), (x+w,y+(h/2)), (0,0,255),1)
				eye = frame[(y+(h/3)):(y+(h*2/3)), (x+(w/4)):(x+(w*3/4))]
				eye = cv2.equalizeHist(eye)
				h, w = eye.shape
				eye = cv2.flip(eye, 1)
				#cv2.imshow("framss", eye)
				topleft = eye[0:h/2, 0:w/2]
				topright = eye[0:h/2, w/2:w]
				bottomleft = eye[h/2:h, 0:w/2]
				bottomright = eye[h/2:h, w/2:w]
				#cv2.imshow("eye", eye)
				#cv2.moveWindow("framss", 600, 600)
# 				print eye[0, 0]
# 				print eye[h-1,w-1]
# 				print eye[0,w-1]
# 				print eye[h-1,0]
# 				print "------------"
				
				startx = 0
				starty = 0
				# cv2.imshow("top left", topleft)
				# cv2.moveWindow("top left", startx, starty)
				# cv2.imshow("top right", topright)
				# cv2.moveWindow("top right", w/2, starty)
				# cv2.imshow("bottom left", bottomleft)
				# cv2.moveWindow("bottom left", startx, h*2)
				# cv2.imshow("bottom right", bottomright)
				# cv2.moveWindow("bottom right", w/2, h*2)
				quadrants = [[topleft, topright],
							 	[bottomleft, bottomright]]
				if(count<testSize):
					baseline += np.array(process(quadrants))
				elif(count == testSize):
					baseline/=testSize
					print "Done with Baseline"
				else:
					difference = np.array(process(quadrants))-np.array(baseline)
					#difference = np.absolute(difference)
					max = np.amax(difference)
					index = np.where(difference==max)
					difference = list(difference)
					index = list(index)
					max = difference[index[0][0]][index[1][0]]
					#print max
					if(max<75):
						#print "Straight Ahead"
						eyeMovement.append(0)
						continue
					for j in range(0, len(quadrants)):
						for k in range(0, len(quadrants[0])):
							quadrants[j][k] = replace(quadrants[j][k], difference[j][k], max)
					if(index[0].any() == 0):
						if(index[1].any() == 1):
							eyeMovement.append(2)
							#print "Top Right"
							k = 1
						else:
							k = 1
							eyeMovement.append(1)
							#print "Top Left"
					else:
						if(index[1].any() == 1):
							k = 1
							eyeMovement.append(3)
							#print "Bottom Right"
						else:
							k = 1
							eyeMovement.append(4)
							#print "Bottom Left"
				count+=1
				h*=2
				starty+=h
				starty+=(h/2)
				# cv2.imshow("top left1", quadrants[0][0])
				# cv2.moveWindow("top left1", startx, starty)
				# cv2.imshow("top right1", quadrants[0][1])
				# cv2.moveWindow("top right1", w/2, starty)
				# cv2.imshow("bottom left1", quadrants[1][0])
				# cv2.moveWindow("bottom left1", startx, h*2)
				# cv2.imshow("bottom right1", quadrants[1][1])
				# cv2.moveWindow("bottom right1", w/2, h*2)
		curr_time = time.time() - start_time
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	return eyeMovement