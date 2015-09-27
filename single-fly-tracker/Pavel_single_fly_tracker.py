# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:49:03 2015

@author: Nicholas Mei
"""
import numpy as np
import cv2
import time
import math
import serial
import types
import datetime
import csv

import roi

from collections import deque

def get_elapsed_time(start_time):
    return time.clock()-start_time
    
def get_dist((x1,y1), (x2,y2)):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def init_arduino():
    #Initialize the arduino!
    #Doing it this way prevents the serial reset that occurs!
    arduino = serial.Serial()
    arduino.port = 'COM7'    
    arduino.baudrate = 9600
    arduino.timeout = 0.1
    arduino.setDTR(False)
    arduino.open()  
    time.sleep(1.5)
    
    def turn_on_stim(self, led_freq, led_dur):
        self.write('{freq},{dur}'.format(freq=led_freq, dur=led_dur))

    def turn_off_stim(self):
        self.write('0,0')
    
    arduino.turn_on_stim = types.MethodType(turn_on_stim, arduino)
    arduino.turn_off_stim = types.MethodType(turn_off_stim, arduino)        
    
    return arduino
            
#%%
            
def start_fly_tracking(debug_mode=True, use_arduino = False, write_csv = True):
    
    if use_arduino:
        try:
            arduino = init_arduino()
        except:
            print "There was a problem connecting to the arduino!! Try restarting Python!"
      
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))    
    
    #Note if your laptop or computer has a built in webcam you may need to set
    #the videocapture element to 1 instead of 0
    cam = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    
    #We don't want the camera to try to autogain as it messes up the image
    #These settings apparently don't work with our camera... (ELP-USBFHD01M-L21)
    #cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
    #cam.set(cv2.CAP_PROP_GAIN, 0.0)
    #So instead let's have the camera capture some frames and adapt the exposure accordingly
    for x in range(40):
        cam.read()
    
    bg = cv2.createBackgroundSubtractorKNN(100,300,False)
    
    fly_location_array = []
    last_fly_locations = deque(maxlen=2)
    last_displacement = deque(maxlen=1)
    #start our background subtractor off with an automatically determined 
    #learning rate
    learning_rate=-1    
    
    expt_start_time = time.clock()
    expt_start_date = datetime.datetime.now().strftime("%Y-%m-%d %H.%M")
    while True:
        _, img = cam.read()
        time_ellapsed = get_elapsed_time(expt_start_time)
        
        fgmask = bg.apply(img, learningRate=learning_rate)        
        #cv2.imshow('preview', img)      
        filtered = cv2.medianBlur(fgmask,5)                     
        dilate = cv2.dilate(filtered, kernel1)
        
        if debug_mode:
            cv2.imshow('background subtracted', fgmask) 
            cv2.imshow('morphology', dilate)
                   
        image, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img, contours, -1, (255,0,0), 2) 
        
        for cont_indx, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)

            if debug_mode:
                print "Contour {} in the given frame has area: {}".format(cont_indx, contour_area)
            
            #enforce a minimum contour area to further remove any false positives
            if contour_area > 10.0:
                moments = cv2.moments(contour)
                
                if moments['m00'] > 0:
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])            
                    #print cx
                    #print cy               
                    cv2.circle(img, (cx, cy), 5, (255,255,255))
                    
                    fly_location_array.append((time_ellapsed, (cx, cy)))
                    last_fly_locations.append(np.array([cx,cy]))
                    
                    if len(last_fly_locations) > 1:
                        dist = get_dist(last_fly_locations[0],last_fly_locations[1])
                        last_displacement.append(dist)
                        
                        if debug_mode:
                            print "Fly's distance displaced from last detection is: {}".format(dist)
        
        if last_displacement:
            #If the fly isn't moving, then we should stop updating the background subtractor
            #as we don't want the fly to become "Background"
            if last_displacement[0] <= 3.0:
                learning_rate = 0
                if debug_mode:
                    print "Setting learning rate to 0!"
            else:
                learning_rate = -1  
                if debug_mode:
                    print "learning rate is auto"
        
        cv2.imshow('contours drawn', img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    if write_csv:
        with open("{}.csv".format(expt_start_date), "wb") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Time Elapsed (sec)", "Fly (x,y)"])
            writer.writerows(fly_location_array)  
    
    #print fly_location_array
    if use_arduino:
        arduino.close()
    cam.release()
    cv2.destroyAllWindows()