import cv2
import numpy as np
import datetime

cv2.namedWindow("test")
cap = cv2.VideoCapture(0)
success,frame = cap.read()
faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyeClassifier = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
leftEyeClassifier = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
rightEyeClassifier = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")

 
while success:
	success, frame = cap.read()
	size = frame.shape[:2]
	image = np.zeros(size , dtype = np.float16)
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	divisor = 10
	h,w = size
	minSize = (w/divisor,h/divisor)
	minEyeSize = (w/20,h/20)
	maxEyeSize = (w/10,h/10)
	faceRects = faceClassifier.detectMultiScale(image,1.2,2,cv2.CASCADE_SCALE_IMAGE,minSize)
	if len(faceRects) > 0:
		for faceRect in faceRects:
			x,y,w,h=faceRect
			#Face
			cv2.circle(frame,(x + w/2,y + h/2),min(w/2,h/2),(255,0,0))
			#Mouse
			cv2.rectangle(frame,(x+3*w/8,y+3*h/4),(x+5*w/8,y+7*h/8),(255,0,0))
			eyeRects = eyeClassifier.detectMultiScale(image,1.2,2,cv2.CASCADE_SCALE_IMAGE,minEyeSize,maxEyeSize)
			if len(eyeRects) > 0:
				for eyeRect in eyeRects:
					ex,ey,ew,eh=eyeRect
					cv2.circle(frame,(ex + ew/2,ey + eh/2),min(ew/2,eh/2),(0,0,255))
			else: 
				print(datetime.datetime.now().strftime('%Y/%m/%d %H:%m:%S.%f')[:-3] + " Not Found eyes")

			#cv2.circle(frame,(x+w/4,y+h/3),min(w/8,h/8),(255,0,0))
			#cv2.circle(frame,(x+3*w/4,y+h/3),min(w/8,h/8),(255,0,0))
			
	cv2.imshow("test",frame) 
	key = cv2.waitKey(10)
	c = chr(key&255)
	if c in ['q','Q',chr(27)]:
		break
cv2.destroyWindow("test")