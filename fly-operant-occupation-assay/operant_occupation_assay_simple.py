# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:49:03 2015

@author: Nicholas Mei
"""
import sys
import os
import time
import json
import datetime
import math
import serial
import types
import csv
from collections import deque

import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
import cv2

#If we are using python 2.7 or under
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
      
#If we are using python 3.0 or above
elif sys.version_info[0] > 3:
    import tkinter as tk
    import tkinter.filedialog as filedialog

import roi

#Location of your ffmpeg.exe file in order to write video out
FFMPEG_BIN = u'C:/FFMPEG/bin/ffmpeg.exe'

def get_elapsed_time(start_time):
    return time.clock()-start_time
    
def get_dist((x1,y1), (x2,y2)):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def preview_camera(calibration_data = None):   
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        cv2.imshow('Camera preview: press "Esc" to close', frame)                    
        if calibration_data:
            mtx = calibration_data["camera_matrix"]
            dist = calibration_data["dist_coeff"]          
            h,  w = frame.shape[:2]           
            #apply undistortion
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            unwarped = cv2.undistort(frame, mtx, dist, None, newcameramtx)         
            # crop the image
            x,y,w,h = roi
            unwarped = unwarped[y:y+h, x:x+w]           
            #image comparisons
            cv2.imshow('Calibrated camera preview: press "Esc" to close',unwarped)        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break          
    cam.release()
    cv2.destroyAllWindows()

def correct_distortion(input_frame, calib_mtx, calib_dist):
    """
    Function that applies correction for radial "fisheye" lens distortion
    make sure you've already loaded the relevant calibration correction matrices
    with the 'read_cam_calibration_file()' function at some point...
    """
    h, w = input_frame.shape[:2]        
    #apply undistortion
    newcameramtx, region = cv2.getOptimalNewCameraMatrix(calib_mtx,calib_dist,(w,h),1,(w,h))
    corrected_frame = cv2.undistort(input_frame, calib_mtx, calib_dist, None, newcameramtx)         
    return corrected_frame
    
def choose_file(custom_text):
    root = tk.Tk()
    try:
        root.tk.call('console','hide')
    except tk.TclError:
        pass
    file_path = filedialog.askopenfilename(parent=root, title=custom_text)
    root.destroy()
    return file_path
    
def load_cam_calib_file(filepath=None):
    """
    Function to read in a camera calibration file which contains the 
    reprojection error, camera_matrix, and distance_coefficient produced by
    the cv2.calibrateCamera() function
    """
    if not filepath:
        filepath = choose_file("Please choose the camera calibration file you wish to load!")

    if os.path.exists(filepath):
        filename, file_extens = os.path.splitext(filepath)                        
        if file_extens == ".json":          
            with open(filepath, 'r') as data_file:
                data = json.load(data_file)                
            if data:
                try:
                    calibration_data = {"reprojection_error": data["reprojection_error"], 
                                        "camera_matrix": np.array(data["camera_matrix"]),
                                        "dist_coeff": np.array(data["dist_coeff"])}                                          
                    print("Camera calibration .json file was successfully loaded!")
                    sys.stdout.flush()
                    
                    return calibration_data
                except:
                    print("It looks like you might have accidentally selected an ROI save file! Try again!")
                    sys.stdout.flush()
        else:
            print("It looks like you didn't select a valid camera calibration file!")
            sys.stdout.flush()
    else:
        print("Loading camera calibration file failed! Check if the file exists at: {}".format(filepath))
        sys.stdout.flush()

def init_arduino():
    #Initialize the arduino!
    #Doing it this way prevents the serial reset that occurs!
    arduino = serial.Serial()
    arduino.port = 'COM7'    
    arduino.baudrate = 115200
    arduino.timeout = 0.05
    arduino.setDTR(False)
    arduino.open()  
    time.sleep(1.5)
    #When serial connection is made, arduino opto-blink script sends an initial
    #"OFF" signal. We'll just read the line and empty the serial buffer
    arduino.readline()
    arduino.is_on = False
    
    def turn_on_stim(self, led_freq, led_dur):
        self.write('{freq},{dur}'.format(freq=led_freq, dur=led_dur))
        arduino_state = self.readline()
        if str(arduino_state) == 'ON':
            self.is_on = True

    def turn_off_stim(self):
        self.write('0,0')
        arduino_state = self.readline()
        if str(arduino_state) == 'OFF':
            self.is_on = False
    
    arduino.turn_on_stim = types.MethodType(turn_on_stim, arduino)
    arduino.turn_off_stim = types.MethodType(turn_off_stim, arduino)        
    
    return arduino
           
def create_rect_contour(start_point, end_point):
    #create points of rectangle in clockwise fashion
    point_list = [[start_point[0], start_point[1]],
                  [end_point[0], start_point[1]],
                  [end_point[0], end_point[1]],
                  [start_point[0], end_point[1]],
                  [start_point[0], start_point[1]]]
                  
    rect_contour = np.array(point_list).reshape((-1,1,2)).astype(np.int32)
    
    return [rect_contour]
    
    
def crop_image(img, roi_object):
    #each position is in array([x,y]) format        
    start_pos, end_pos = roi_object.roi  
    #Image cropping works by img[y: y + h, x: x + w]
    #This cropping is just a slice into the existing "current frame" *NOT* an array copy
    cropped_img = img[start_pos[1]:end_pos[1], start_pos[0]:end_pos[0]]
                                
    return cropped_img
    
def init_video_writer(frame_height, frame_width, fps_cap, save_path, timestring):
    fname = "video--" + timestring
                
    ffmpeg_command = [ FFMPEG_BIN,
                      '-f', 'rawvideo',
                      '-pix_fmt', 'bgr24',
                      '-s', '{}x{}'.format(frame_width,frame_height), # size of one frame
                      '-r', '{}'.format(fps_cap), # frames per second
                      '-i', '-', # The imput comes from a pipe
                      '-an', # Tells FFMPEG not to expect any audio
                      '-vcodec', 'libx264',
                      '-preset', 'fast',
                      #'-qp', '0', #"-qp 0" specifies lossless output
                      os.path.join(save_path, "{}.avi".format(fname))]
                               
    #Note to self, don't try to redirect stout or sterr to sp.PIPE as filling the pipe up will cause subprocess to hang really bad :(
    video_writer = sp.Popen(ffmpeg_command, stdin=sp.PIPE) 
    
    return video_writer
            
#%%
def start_fly_tracking(expt_dur = 900, led_freq = 30, led_pw=5, fps_cap = 30, 
                       debug_mode=False, use_arduino = True, 
                       write_video = True, write_csv = True, 
                       cam_calib_file = "Camera_calibration_matrices.json",
                       default_save_dir = "C:\\Users\\Nicholas\\Desktop\\Operant Occupancy Assay"):
    if use_arduino:
        try:
            arduino = init_arduino()
        except:
            print "There was a problem connecting to the arduino!! Try restarting Python!"
  
    calibration_data = load_cam_calib_file(cam_calib_file)  
    
    calib_mtx = calibration_data["camera_matrix"]
    calib_dist = calibration_data["dist_coeff"]
  
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
        ret, temp_frame = cam.read()
        
    temp_frame = correct_distortion(temp_frame, calib_mtx, calib_dist)
        
    #Set up arena and occupancy rois
    arena_roi = roi.set_roi('blue', temp_frame)
    arena_roi.wait_for_roi()
    occupancy_roi = roi.set_roi('green', crop_image(temp_frame, arena_roi))
    occupancy_roi.wait_for_roi()
    
    occupancy_contour = create_rect_contour(occupancy_roi.roi[0], occupancy_roi.roi[1])
  
    bg = cv2.createBackgroundSubtractorKNN(100,150,False)
    
    fly_location_array = []
    last_fly_locations = deque(maxlen=2)
    last_displacement = deque(maxlen=1)
    #start our background subtractor off with an automatically determined 
    #learning rate
    learning_rate=-1    
    
    expt_start_time = time.clock()
    expt_start_date = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    
    save_path = os.path.join(default_save_dir, expt_start_date + " - {} Hz {} PW".format(led_freq, led_pw))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if write_video:
        cropped_temp_frame = crop_image(temp_frame, arena_roi) 
        vid_height, vid_width = cropped_temp_frame.shape[:2]
        
        raw_video_writer = init_video_writer(vid_height, 
                                             vid_width, 
                                             fps_cap, 
                                             save_path, 
                                             expt_start_date)
        
        tracking_video_writer = init_video_writer(vid_height, 
                                                  vid_width, 
                                                  fps_cap, save_path, 
                                                  expt_start_date + ' - tracking')
           
    while True:
        ret, current_frame = cam.read()
        time_ellapsed = get_elapsed_time(expt_start_time)
        
        if time_ellapsed > expt_dur:
            break
       
        if ret:
            current_frame = correct_distortion(current_frame, calib_mtx, calib_dist)
         
        cropped_current_frame = crop_image(current_frame, arena_roi)   
        
        if write_video:
            raw_video_writer.stdin.write(cropped_current_frame.tostring())
      
        cropped_copy = cropped_current_frame.copy()
        cv2.drawContours(cropped_copy, occupancy_contour, -1, color=(255, 0, 0), thickness=1)                                
             
        fgmask = bg.apply(cropped_current_frame, learningRate=learning_rate)        
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
            if contour_area > 15.0:
                moments = cv2.moments(contour)               
                if moments['m00'] > 0:
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])            
                    #print cx
                    #print cy               
                    cv2.circle(cropped_copy, (cx, cy), 8, (255,255,255), thickness=1)

                    if use_arduino:
                        if getattr(arduino, "is_on"):
                            cv2.circle(cropped_copy, (cx, cy), 11, (0,0, 255), thickness=2)
                    
                    #cv2.circle(current_frame, (cx, cy), 5 (255, 255, 255))
                    
                    last_fly_locations.append(np.array([cx,cy]))               
                    in_rewd = cv2.pointPolygonTest(occupancy_contour[0], (cx, cy), False)
                    
                    #print "\n\ninPolygon test gives: {}\n\n".format(in_contour)
                    
                    if in_rewd == 1:
                        rewd_contour_status = True
                        if debug_mode:
                            print "The fly *IS* in the occupancy region!!"                        
                        if use_arduino and not getattr(arduino, "is_on"):
                            arduino.turn_on_stim(led_freq,led_pw) 
                                                                                           
                            #if get_dist(occupancy_roi.roi[0]+obtained_rewards, occupancy_roi.roi[1]-obtained_rewards) > 5:                                          
                            #    occupancy_contour = create_rect_contour(occupancy_roi.roi[0]+obtained_rewards, occupancy_roi.roi[1]-obtained_rewards)
                    else:
                        rewd_contour_status = False                       
                        if debug_mode:
                            print "The fly is *NOT* in the occupancy region!!"                       
                        if use_arduino:
                            if getattr(arduino, "is_on"):
                                arduino.turn_off_stim()    
                            

                    fly_location_array.append((time_ellapsed, cx, cy, rewd_contour_status))
                    
                    if len(last_fly_locations) > 1:
                        dist = get_dist(last_fly_locations[0],last_fly_locations[1])
                        last_displacement.append(dist)
                        
                        if debug_mode:
                            print "Fly's distance displaced from last detection is: {}".format(dist)
        
        if last_displacement:
            #If the fly isn't moving, then we should stop updating the background subtractor
            #as we don't want the fly to become "Background"
            if last_displacement[0] <= 2.5:
                learning_rate = 0
                if debug_mode:
                    print "Learning rate is 0!\n\n"
            else:
                learning_rate = -1  
                if debug_mode:
                    print "Learning rate is auto!\n\n"                    
                    
        if write_video:
            tracking_video_writer.stdin.write(cropped_copy.tostring())
        
        cv2.imshow('contours drawn', cropped_copy)        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    if write_csv:
        with open(os.path.join(save_path, "{}.csv".format(expt_start_date)), "wb") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Time Elapsed (sec)", "Fly x", "Fly y", "Fly in rewd region?"])
            writer.writerows(fly_location_array)  
    
    #print fly_location_array
    if use_arduino:
        arduino.turn_off_stim()
        arduino.close()
        
    if write_video:
        raw_video_writer.stdin.close()
        raw_video_writer.wait()
        
        tracking_video_writer.stdin.close()
        tracking_video_writer.wait()
    cam.release()
    cv2.destroyAllWindows()
    