# -*- coding: utf-8 -*-
from win32api import GetSystemMetrics
from PyQt5.QtWidgets import QWidget,QApplication,QLabel
from PyQt5 import QtCore, QtGui
import sys
from PyQt5.QtGui import QPixmap
from collections import deque
import cv2
import dlib
import time
import math
import pytesseract
#import os
import base64
import requests
import json


class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title = 'Traffic voilation - safevision.ai'
        

        self.left = 0
        self.top = 0
        self.width = GetSystemMetrics(0) #640
        self.height = GetSystemMetrics(1) #480
        self.height = self.height - 50
        self.carCascade = cv2.CascadeClassifier('myhaar.xml')
        self.numCascade = cv2.CascadeClassifier('numplate.xml')
        self.video = cv2.VideoCapture('cars_client1.avi')
        self.initUI()
    
    def estimateSpeed(self,location1, location2,fps):
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        #ppm = 8.8
        ppm = 20
        d_meters = d_pixels / ppm
        #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        speed = d_meters * fps * 3.6  # to get the km/h we have devided 3600 sec with 1000 meters(3.6)
        return speed
        
            
    def service_call(self,data):
        service_url = "http://13.126.21.160:5000/trafficViolation"
        headers={'Content-Type': "application/json",'cache-control': "no-cache"}
        response = requests.request("POST", service_url, data=data, headers=headers)
        print(response)

    
    

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height) 
        #self.resize(1200, 650) #1800,1200 width & height

        w = self.width
        print(w)
        w_frame = (w/100)*80
        h = self.height
        print(h)
        h_frame = (h/100)*100
        
        w_sub_frame = (w/100)*20
        h_sub_frame = (h/100)*20
        
        
#        layout = QWidget(self)
        label = QLabel(self)
        label_1 = QLabel(self)
        label_2 = QLabel(self)
        label_3 = QLabel(self)
        label_4 = QLabel(self)
        label_5 = QLabel(self)

        label_img = QLabel(self)
        pixmap = QPixmap('logo.png')
        label_img.setPixmap(pixmap)
        label_img.setScaledContents(True)
        label_img.setGeometry(QtCore.QRect(0, 0, 140, 50))

        label_img1 = QLabel(self)
        pixmap1 = QPixmap('byteforce_logo.jpeg')
        label_img1.setPixmap(pixmap1)
        label_img1.setScaledContents(True)
        label_img1.setGeometry(QtCore.QRect(1500, 10, 140, 50))
#        
      
        stack = deque(["0","1","2","3","4"])
        stack.append("new")
        stack.popleft()
        
        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentCarID = 0
        fps = 0

        carTracker = {}
#        carNumbers = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000

        WIDTH = 1200
        HEIGHT = 720
        
       
        while True:
            start_time = time.time()
            rc, image = self.video.read()
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print("---------------------------HEIGHT---------")
            # print(height)
            if type(image) == type(None):
                break
            
            image = cv2.resize(image, (WIDTH, HEIGHT))
            #print(image)
            resultImage = image.copy()
            
            
            frameCounter = frameCounter + 1

            
            carIDtoDelete = []

            for carID in carTracker.keys():
                #print(carTracker)
                trackingQuality = carTracker[carID].update(image)
                #print(carTracker[carID])

                #print(trackingQuality)
                print(trackingQuality)
                if trackingQuality < 10:
                    carIDtoDelete.append(carID)
                    
            for carID in carIDtoDelete:
                print ('Removing carID ' + str(carID) + ' from list of trackers.')
                print ('Removing carID ' + str(carID) + ' previous location.')
                print ('Removing carID ' + str(carID) + ' current location.')
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)
            
            if not (frameCounter % 2):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cars = self.carCascade.detectMultiScale(gray, 1.3, 18, 18, (60, 60))  #gray, 2, 13, 18, (40, 30)
#               rects = cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=4,minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
                
        
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
            
            cv2.putText(resultImage, '#: ' + str(currentCarID), (0, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


            for i in carLocation1.keys():	
                if frameCounter % 1 == 0:
                    

                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]
            
                    # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                    carLocation1[i] = [x2, y2, w2, h2]
            
                    # print 'new previous location: ' + str(carLocation1[i])
                    d_pixels = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
                    # print('-----------------')
                    # print(d_pixels)
                    # print("-------------------")
                    # if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if(d_pixels > 5):
                        # if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        if (speed[i] == None or speed[i] == 0):

                            speed[i] = self.estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2],fps)
                            vnum = ''
                            if(int(speed[i]) > 15):
                                trackedPosition = carTracker[i].get_position()
                                p_x = int(trackedPosition.left())
                                p_y = int(trackedPosition.top())
                                p_w = int(trackedPosition.width())
                                p_h = int(trackedPosition.height())
                                img = resultImage
                                crop_img = img[p_y:p_y+p_h, p_x:p_x+p_w]
                                cv2.putText(crop_img, 'Speed: ' + str(int(speed[i])) + "Km/Hr", (0, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                # cv2.imshow("cropped", crop_img)
                                ii = "Cropped"+str(i)+".jpg"
                                
                                cv2.imwrite(ii, crop_img)
                                stack.append(ii)
                                stack.popleft()
#                                numgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#                                numthresh = cv2.adaptiveThreshold(numgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                                numberplate = self.numCascade.detectMultiScale(crop_img, 1.1,4,10)  #(crop_img, 1.1,4,10)
#                                rects = cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=4,minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
                                
                                

                                for (n_x,n_y,n_w,n_h) in numberplate:
                                    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,255,255),2)
                                    detect = crop_img[n_y:n_y+n_h,n_x:n_x+n_w]
                                    cv2.imshow("detect",detect)
                                    
                                    alpha = 1.5 # Contrast control (1.0-3.0)
                                    beta = 0 # Brightness control (0-100)
                                    detect = cv2.convertScaleAbs(detect, alpha=alpha, beta=beta)
                                    
                                    detect = cv2.fastNlMeansDenoisingColored(detect, None, 10, 10, 7, 15) 
                                    
                                    scale_percent = 420 # percent of original size increase of it value will increases the size of the image.
                                    width = int(detect.shape[1] * scale_percent / 100)
                                    height = int(detect.shape[0] * scale_percent / 100)
                                    dim = (width, height)   
                                    
                                    detect = cv2.resize(detect, dim, interpolation = cv2.INTER_AREA)
                                    
                                    gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
                                    cv2.imshow("Final detection",gray)  
                                    vnum = ''
                                    vnum = pytesseract.image_to_string(gray)
                                    print("------------------------CAR NUMBER-------------------")
                                    print(vnum)
                                    print("-----------------------------------------------------")
                                
                                
                                retval, img = cv2.imencode('.jpg', crop_img)
                                vimg = base64.b64encode(img)
                                vimg = vimg.decode('utf8')
                                vspeed= round(speed[i],2)
                                
                                fields = {
                                        "CamId":1,
                                        "VechileNumber":vnum,
                                        "speed":vspeed,
                                        "VechileImage":vimg
                                        }
                                
                                
                                op = json.dumps(fields)
                                self.service_call(op)
                                vnum = ''
                                
                                #if y1 > 275 and y1 < 285:
                        #print(speed[i])
                        #print(y1)
                        if speed[i] != None and y1 >= 180:
                            cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        
                            print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
                            
                                
                        else:
                            cv2.putText(resultImage, "CarID " + str(i), (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' +  str(speed[i]) + ' km/h.\n')
                        
            # cv2.imshow('result', resultImage)
            # detection = True
            rgbImage = cv2.cvtColor(resultImage, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                             QtGui.QImage.Format_RGB888)
            convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
            pixmap = QPixmap(convertToQtFormat)
            #resizeImage = pixmap.scaled(900, 600) #changing 640,480 , QtCore.Qt.KeepAspectRatio
            resizeImage = pixmap.scaled(w_frame, h_frame)
            QApplication.processEvents()
            label.setPixmap(resizeImage)
            self.show()

            #label_1.setGeometry(QtCore.QRect(920, 0, 250, 160)) #Here x,y,width,height
            label_1.setGeometry(QtCore.QRect(5+w_frame, 0, w_sub_frame, h_sub_frame))
            img1 = stack[0]
            label_1.setPixmap(QtGui.QPixmap(img1))
            label_1.setScaledContents(True)
            

            #label_2.setGeometry(QtCore.QRect(920, 165, 250, 160))
            label_2.setGeometry(QtCore.QRect(5+w_frame, 1*h_sub_frame, w_sub_frame, h_sub_frame))
            img2 = stack[1]
            label_2.setPixmap(QtGui.QPixmap(img2))
            label_2.setScaledContents(True)
            
            #label_3.setGeometry(QtCore.QRect(920, 330, 250, 160))
            label_3.setGeometry(QtCore.QRect(5+w_frame, 2*h_sub_frame, w_sub_frame, h_sub_frame))
            img3 = stack[2]
            label_3.setPixmap(QtGui.QPixmap(img3))
            label_3.setScaledContents(True)
            
            #label_4.setGeometry(QtCore.QRect(920, 495, 250, 160))
            label_4.setGeometry(QtCore.QRect(5+w_frame, 3*h_sub_frame, w_sub_frame, h_sub_frame))
            img4 = stack[3]
            label_4.setPixmap(QtGui.QPixmap(img4))
            label_4.setScaledContents(True)

            #label_5.setGeometry(QtCore.QRect(920, 660, 250, 160))
            label_5.setGeometry(QtCore.QRect(5+w_frame, 4*h_sub_frame, w_sub_frame, h_sub_frame))
            img5 = stack[4]
            label_5.setPixmap(QtGui.QPixmap(img5))
            label_5.setScaledContents(True)
            # Write the frame into the file 'output.avi'
            #out.write(resultImage)


            if cv2.waitKey(33) == 27:
                break
        
        cv2.destroyAllWindows()
        




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
