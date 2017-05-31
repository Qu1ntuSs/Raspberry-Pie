import numpy as np
from PIL import ImageGrab
import cv2
import time

#http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/


#bbox specifies specific region (bbox= x,y,width,height)
bbox_youtube_small = (100, 120, 865, 550)
bbox_youtube_2 = (36,127,890,606)
bbox_full_screen = (0,0,1366,768)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)


def fps(last_time, current_time, img):
	fps = None
	loop_time = current_time - last_time
	
	try:
		fps = 1/(loop_time)
	except ZeroDivisionError:
		pass
	
	message = 'FPS: {}'.format(str(round(fps, 1)))
	#message = 'Loop: {}'.format(loop_time)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, message, (25,460), font, 1, (105,105,105), 4, cv2.LINE_AA)	

	return img
	

def write(message):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, message, (170,460), font, 2, (255,0,0), 4, cv2.LINE_AA)

def find_face(img, place_message=False, message=None, find_the_eyes=False):
	font = cv2.FONT_HERSHEY_SIMPLEX
	font2 = cv2.FONT_HERSHEY_TRIPLEX
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	found_face = False
	#full_body = full_body_cascade.detectMultiScale(gray, 1.3, 5)
	
	#for (x,y,w,h) in full_body:
	#	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		found_face = True
		#cv2.putText(img, 'Human detected', (170,460), font2, 2, (255,0,0), 4, cv2.LINE_AA)
		
		if find_the_eyes==True:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
		else:
			pass
			
		if place_message==True:
			
			cv2.putText(img, message, (x,y-10), font, 1, (0,0,255), 4, cv2.LINE_AA)	
		else:
			pass
		
		
		
	return img, found_face
	




if vc.isOpened(): # try to get the first frame
    rval, img = vc.read()
else:
    rval = False

last_time = time.time()
while rval:
	img, found_face = find_face(img, place_message=True, message='KILL', find_the_eyes=True)
	current_time = time.time()
	img = fps(last_time, current_time, img)
	cv2.imshow("preview", img)
	rval, img = vc.read()
	if found_face == True:
		write('found face')
		#lcd('KILL')
	#else:
		#lcd(clear=True)
		
	last_time = time.time()
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break
cv2.destroyWindow("preview")


