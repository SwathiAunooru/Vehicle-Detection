import cv2
import dlib
import time
import math
import pytesseract

carCascade = cv2.CascadeClassifier('myhaar.xml')
numCascade = cv2.CascadeClassifier('numplate.xml')
video = cv2.VideoCapture('cars.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2,fps):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	#ppm = 8.8
	ppm = 20
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6  # to get the km/h we have devided 3600 sec with 1000 meters(3.6)
	return speed
	

def trackMultipleObjects():
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	#carloc1_img = {}
	carLocation2 = {}
	#carloc2_img = {}
	speed = [None] * 1000
	
	# Write output to video file
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))

	counter = 0
	st_time = time.time()
	while True:
		start_time = time.time()
		rc, image = video.read()
		height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print("---------------------------HEIGHT---------")
		print(height)
		if type(image) == type(None):
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		#print(image)
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1

		
		carIDtoDelete = []

		counter+=1
		if (time.time() - st_time) > 1 :
			print("-----------------------------------FPS----------------------: ", counter / (time.time() - start_time))
			counter = 0
			start_time = time.time()

		for carID in carTracker.keys():
			#print(carTracker)
			trackingQuality = carTracker[carID].update(image)
			#print(carTracker[carID])

			#print(trackingQuality)
			
			if trackingQuality < 5:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print ('Removing carID ' + str(carID) + ' from list of trackers.')
			print ('Removing carID ' + str(carID) + ' previous location.')
			print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 2 , 13 , 18, (30, 30))    #originally given vals gray, 2, 13, 18, (30, 30)
			#print(cars)
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentCarID))
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
		
		#cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)


		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
			
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]
		
		end_time = time.time()
		
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		cv2.putText(resultImage, 'Frame Counter: ' + str(int(frameCounter)), (700, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
		
		cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


		for i in carLocation1.keys():	
			if frameCounter % 1 == 0:
				

				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]
		
				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
				carLocation1[i] = [x2, y2, w2, h2]
		
				# print 'new previous location: ' + str(carLocation1[i])
				d_pixels = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
				print('-----------------')
				print(d_pixels)
				print("-------------------")
				# if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
				if(d_pixels > 10):
					#if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
					if (speed[i] == None or speed[i] == 0):

						speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2],fps)
						
						if(int(speed[i]) > 10):
							print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
							print ('CarID ' + str(i) + ' Location1: ' + str([x1, y1, w1, h1]) + ' Location2: ' + str([x2, y2, w2, h2]) + ' speed is ' +  str(speed[i]) + ' km/h.\n')
							trackedPosition = carTracker[i].get_position()
					
							p_x = int(trackedPosition.left())
							p_y = int(trackedPosition.top())
							p_w = int(trackedPosition.width())
							p_h = int(trackedPosition.height())
							img = resultImage
							crop_img = img[p_y:p_y+p_h, p_x:p_x+p_w]
							# cv2.imshow("cropped", crop_img)
							ii = "Cropped"+str(i)+".jpg"
							cv2.imwrite(ii, crop_img)
							numgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
							numthresh = cv2.adaptiveThreshold(numgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
							numberplate = numCascade.detectMultiScale(numthresh, 1.1,2)

							for (n_x,n_y,n_w,n_h) in numberplate:
								cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,255,255),2)
								detect = crop_img[n_y:n_y+n_h,n_x:n_x+n_w]
								cv2.imshow("detect",detect)
								text = pytesseract.image_to_string(numthresh)
								print("------------------------CAR NUMBER-------------------")
								print(text)
								print("-----------------------------------------------------")




					#if y1 > 275 and y1 < 285:
					#print(speed[i])
					print(y1)
					if speed[i] != None and y1 >= 180:
						cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)					
						print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
						
							
					else:
						cv2.putText(resultImage, "CarID " + str(i), (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
						print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' +  str(speed[i]) + ' km/h.\n')
					
		cv2.imshow('result', resultImage)
		# Write the frame into the file 'output.avi'
		out.write(resultImage)


		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()
