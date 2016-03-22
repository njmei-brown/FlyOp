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
import serial
import csv
from collections import deque

import subprocess as sp
import numpy as np
import multiprocessing as mp
import cv2
import roi

#from profilehooks import profile

 #If we are using python 2.7 or under
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
      
#If we are using python 3.0 or above
elif sys.version_info[0] > 3:
    import tkinter as tk
    import tkinter.filedialog as filedialog

#Location of your ffmpeg.exe file in order to write video out
FFMPEG_BIN = u'C:/FFMPEG/bin/ffmpeg.exe'

#%%


def get_elapsed_time(start_time):
    return time.clock()-start_time  
    
def correct_distortion(input_frame, calib_mtx, calib_dist, crop_frame=False):
    """
    Function that applies correction for radial "fisheye" lens distortion
    make sure you've already loaded the relevant calibration correction matrices
    with the 'read_cam_calibration_file()' function at some point...
    """
    h, w = input_frame.shape[:2]        
    #apply undistortion
    newcameramtx, region = cv2.getOptimalNewCameraMatrix(calib_mtx,calib_dist,(w,h),1,(w,h))
    corrected_frame = cv2.undistort(input_frame, calib_mtx, calib_dist, None, newcameramtx)  
    
    if crop_frame :
        # crop the image
        x,y,w,h = region
        corrected_frame = corrected_frame[y:y+h, x:x+w]   
       
    return corrected_frame

def crop_image(img, roi_object):
    """
    Function that crops an image according to roi coordinates of an roi_object
    that is possed to it.
    """ 
    #each position is in array([x,y]) format        
    start_pos, end_pos = roi_object.roi  
    #Image cropping works by img[y: y + h, x: x + w]
    #This cropping is just a slice into the existing "current frame" *NOT* an array copy
    #Because img is a numpy array and slicing in numpy returns a view, not a copy
    cropped_img = img[start_pos[1]:end_pos[1], start_pos[0]:end_pos[0]]
                                
    return cropped_img
    
def init_video_writer(frame_width, frame_height, fps_cap, save_path, timestring, roi_name, encode_preset):
    fname = "video - {} - {}".format(timestring, roi_name)       
    ffmpeg_command = [ FFMPEG_BIN,
                      '-f', 'rawvideo',
                      '-pix_fmt', 'bgr24',
                      '-s', '{}x{}'.format(frame_width,frame_height), # size of one frame
                      '-r', '{}'.format(fps_cap), # frames per second
                      '-i', '-', # The imput comes from a pipe
                      '-an', # Tells FFMPEG not to expect any audio
                      '-vcodec', 'libx264rgb',
                      '-preset', encode_preset, #tells ffmpeg how much processor time to spend encoding
                      #'-qp', '0', #"-qp 0" specifies lossless output
                      os.path.join(save_path, "{}.avi".format(fname))]
                               
    #Note to self, don't try to redirect stout or sterr to sp.PIPE as filling the pipe up will cause subprocess to hang really bad :(
    video_writer = sp.Popen(ffmpeg_command, stdin=sp.PIPE) 
    
    return video_writer  

def video_capture(child_conn_obj, data_q_obj, 
                  calib_mtx, calib_dist, fps_cap, 
                  roi_list, vid_params, save_path, write_video):
     """
     This function contains the camera read() loop. 
     
     Because a fast camera loop is better, we start this function as 
     its own process using multiprocessing (mp).
     
     1) function waits for msg from parent process which either indicates
         a) a time stamp to align saved filenames
         b) a start command which initiates video capture
     2) USB camera is initialized
     3) While loop which will continuously:
         a) grab frames from camera
         b) crop frames into ROIs specified during startup of the parent process
         c) send cropped frames back to parent process for processing and tracking
         d) write cropped frames to .AVI (x264rgb) files *OPTIONAL* 
     """   
     
     #Loop optimization by removing '.' function references
     child_conn_obj_poll = child_conn_obj.poll
     child_conn_obj_recv = child_conn_obj.recv
     data_q_obj_put_nowait = data_q_obj.put_nowait  
     time_clock = time.clock   
   
     #Wait for the start signal from the parent process to begin grabbing frames
     #Note that 'while 1' is faster than 'while True'
     while 1:
         #This will block until it receives the message it was waiting for
         msg = child_conn_obj_recv()                
         #The parent process will send a timestamp right before sending the 
         #'Start' signal. This allows all file names to be synchronized to when
         #the expt.start_expt() command is called.
         if 'Time' in msg:            
             timestring = msg.split(":")[-1]            
             if write_video: 
                 raw_video_writers = [init_video_writer(vid_width, vid_height, fps_cap, save_path, timestring, arena_name, encode_preset='fast') for vid_height, vid_width, arena_name in vid_params]
                 dot_optimized_writers = [video_writer.stdin.write for video_writer in raw_video_writers]
                 
         if msg == 'Start!':
             break 
         
     #initilize the video capture object
     cam  = cv2.VideoCapture(cv2.CAP_DSHOW + 0) 
     cam_read = cam.read
     #We don't want the camera to try to autogain as it messes up the image
     #So start acquiring some frames to avoid the autogaining frames
     for x in xrange(55):
         ret, temp = cam_read()     
     #start the clock!!
     expt_start_time = time_clock() 
     fps_cap_timer = time_clock()
     
     #camera read and video write loop
     #Will continue running until it receives the 'Shutdown!' message from
     #parent process
     while 1:        
         #poll to see if there is any msg to read, we don't want this to block
         if child_conn_obj_poll():
             msg = child_conn_obj_recv()
             if msg == 'Shutdown!':
                 break
         
         #enforce an FPS cap such that camera read speed cannot be faster than the cap
         if get_elapsed_time(fps_cap_timer) >= 1/float(fps_cap):   
             fps_cap_timer = time_clock()            
             ret, raw_frame = cam_read()     
             
             frame = correct_distortion(raw_frame, calib_mtx, calib_dist)    
             cropped_frames = [crop_image(frame, roi) for roi in roi_list]                             
             # Use the multiprocessing Queue to send a timestamp, and video frame
             # to the post-processing and analysis portion of script             
             data_q_obj_put_nowait((get_elapsed_time(expt_start_time), cropped_frames))
             
             if write_video:
                 for indx, vid_writer in enumerate(dot_optimized_writers):
                     vid_writer(cropped_frames[indx].tostring())
             
     #Release the camera before closing process    
     #parent process will take care of closing connection objects and data_q
     cam.release()
     
     #Close all raw video writer instances
     for vid_writer in raw_video_writers:
         vid_writer.stdin.close()
         vid_writer.wait()

def get_dist(a, b):
    return np.linalg.norm(a-b)
    
def get_disp_rate(a, b):
    dist = get_dist(a[0], b[0])
    dt = b[1] - a[1]
    result = dist/dt
    return result

def preview_camera(calibration_data = None):   
    cam = cv2.VideoCapture(0)
    
    while 1:
        ret, frame = cam.read()
        cv2.imshow('Camera preview: press "Esc" to close', frame)                    
        if calibration_data:     
            calib_mtx = calibration_data["camera_matrix"]
            calib_dist = calibration_data["dist_coeff"]         
            #apply undistortion
            unwarped = correct_distortion(frame, calib_mtx, calib_dist)          
            #image comparisons
            cv2.imshow('Calibrated camera preview: press "Esc" to close',unwarped)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break          
    cam.release()
    cv2.destroyAllWindows()
    
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
        
#%%

class InitArduino:
    """
    InitArduino class initializes an arduino serial port connection instance.
    The class is configured with methods to communicate with Arduinos 
    loaded with the "opto-blink" or "opto-blink_and_solenoid" sketches
    """
    def __init__(self, port='COM4', baudrate=115200, timeout=0.02):
        #Initialize the arduino!
        #Doing it this way prevents the serial reset that occurs!
        self.arduino = serial.Serial()
        self.arduino.port = port   
        self.arduino.baudrate = baudrate
        self.arduino.timeout = timeout
        self.arduino.setDTR(False)
        self.arduino.open()  
        time.sleep(0.5)
        #When serial connection is made, arduino opto-blink script sends an initial
        #"OFF" signal. We'll just read the line and empty the serial buffer
        self.arduino.readline()
        self.is_on = False
        
        #Arduino state consists of 6 values (LED_freq,LED_PW,SOL1,SOL2,SOL3,SOL4)
        self.state = '0.00,0.00,0.00,0.00,0.00,0.00'
        self.desired_state = None
        
        #Loop optimizations
        self.arduino_readline = self.arduino.readline
        self.np_fromstring = np.fromstring
    
    #method used to determine a desired solenoid state
    def update_desired_state(self, new_state, roi_id):
        if self.desired_state:
            prior_state = self.np_fromstring(self.desired_state, dtype=float, sep=',')
        else:
            #first convert state string of floats into numpy array
            prior_state = self.np_fromstring(self.state, dtype=float, sep=',')
            
        state_indx = roi_id + 1    
        
        if int(prior_state[state_indx] - new_state) != 0:
            prior_state[state_indx] = new_state         
            self.desired_state = ",".join(map(str, prior_state))        
    
    #method to tell Arduino to achieve the desired state
    def write_desired_state(self):
        if self.desired_state:
            self.write(self.desired_state)
            self.desired_state = None
    
    def report_state(self):
        return self.np_fromstring(self.state, dtype=float, sep=',')
        
    #Avoid using this method on its own, update_desired_state()/write_desired_state() are far safer!!
    def write(self, values):
        self.arduino.write(values)
        self.state = self.arduino_readline()
    
    def turn_on_stim(self, led_freq, led_dur):
        self.arduino.write('{freq},{dur}'.format(freq=led_freq, dur=led_dur))
        self.state = self.arduino_readline()
        #if str(arduino_state) == 'ON':
            #self.is_on = True

    def turn_off_stim(self):
        self.arduino.write('0,0')
        self.state = self.arduino_readline()
        #if str(arduino_state) == 'OFF':
            #self.is_on = False
        
    def turn_off_solenoids(self):
        self.arduino.write('0,0,0,0,0,0')
        self.state = self.arduino_readline()
            
    def close(self):
        '''
        Closes the serial connection to the arduino. 
        
        Note: Make sure to close serial connection when finished with arduino 
        otherwise subsequent attempts to connect to the arduino will be blocked!!
        '''
        self.turn_off_solenoids()
        self.arduino.close()
#%%

def create_rect_contour(start_point, end_point):
    #create points of rectangle in clockwise fashion
    point_list = [[start_point[0], start_point[1]],
                  [end_point[0], start_point[1]],
                  [end_point[0], end_point[1]],
                  [start_point[0], end_point[1]],
                  [start_point[0], start_point[1]]]
                  
    rect_contour = np.array(point_list).reshape((-1,1,2)).astype(np.int32)    
    return [rect_contour]    

#%%
class Arena():
    """
    Arena class where each instance of the class (an arena) is tracking 1 fly. 
    
    Each Arena instance contains the following important variables:
    1) A KNN cv2 background subtractor (and a bgs learning rate)
    2) An arena ROI instance
    *Optionally* an occupancy ROI instance
    3) A video writer to write video covering the arena ROI along with tracking annotations
    4) A list of fly locations over time (fly_location_array)
    """
    def __init__(self, arena_label, arena_id, input_frame, get_occupancy_roi=False, 
                 get_displacement=False,
                 write_video=False, debug_mode=False, fps_cap=30, use_arduino=False,
                 arduino_obj = None):
        self.name = arena_label
        self.write_video = write_video
        self.debug_mode = debug_mode
        self.get_occupancy_roi = get_occupancy_roi
        self.get_displacement = get_displacement
        self.sample_frame = input_frame
        self.fps_cap = fps_cap
        self.arena_id = arena_id       
        self.arduino = arduino_obj
        self.use_arduino = use_arduino
        
        if use_arduino:
            self.arduino_update_desired_state = self.arduino.update_desired_state
           
        #Prompt user for set up of the arena bounds
        roi_msg = "Press the 'n' key on your keyboard when you are happy\n with the region of interest for {}".format(self.name)
        self.arena_roi = roi.set_roi('blue', self.sample_frame, 
                                     roi_selection_msg = roi_msg)
        self.arena_roi.wait_for_roi()
        self.arena_roi.name = arena_label
        
        if self.get_occupancy_roi:
            self.occupancy_roi = roi.set_roi('green', crop_image(self.sample_frame, self.arena_roi))
            self.occupancy_roi.wait_for_roi()          
            self.occupancy_contour = create_rect_contour(self.occupancy_roi.roi[0], self.occupancy_roi.roi[1])
            
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(50,150,False)
        self.bg_apply = self.bg_subtractor.apply
        #start our background subtractor (bgs) off with an automatically 
        #determined learning rate  
        self.bgs_learning_rate = -1
                
        self.fly_location_array = []
        self.last_fly_locations = deque(maxlen=2)
        self.last_displacement = deque(maxlen=1)    
        self.last_rewd_status = deque(maxlen=1)
        
        self.bgs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))  
                                                      
    def init_video_writer(self, save_path, expt_start_date):
        if self.write_video:
            cropped_sample_frame = crop_image(self.sample_frame, self.arena_roi) 
            self.vid_height, self.vid_width = cropped_sample_frame.shape[:2]
                      
            self.tracking_video_writer = init_video_writer(self.vid_width, 
                                                      self.vid_height, 
                                                      self.fps_cap, save_path, 
                                                      expt_start_date + ' - tracking',
                                                      self.name,
                                                      encode_preset='faster')
    def close_video_writer(self):
        self.tracking_video_writer.stdin.close()
        self.tracking_video_writer.wait()
    
    def preprocess_frame(self, cropped_current_frame, time_ellapsed):
        """
        Function to preprocess frame by:
        1) cropping input frame to only include the arena roi area
        2) creating a copy of the cropped frame (1 version gets annotated, the other is fed into cv2.findContours)
        2) applying the background subtractor
        3) filtering and applying morphology operation to consolidate detections
        """       
        cropped_copy = cropped_current_frame.copy()
        
        if self.get_occupancy_roi:
            cv2.drawContours(cropped_copy, self.occupancy_contour, -1, color=(255, 0, 0), thickness=1)                                
             
        fgmask = self.bg_apply(cropped_current_frame, learningRate=self.bgs_learning_rate)        
        #cv2.imshow('preview', img)      
        filtered = cv2.medianBlur(fgmask,5)                     
        dilate = cv2.dilate(filtered, self.bgs_kernel)
        
        if self.debug_mode:
            cv2.imshow('{} background subtracted'.format(self.name), fgmask) 
            cv2.imshow('{} median blur'.format(self.name), filtered)
            cv2.imshow('{} morphology'.format(self.name), dilate)
            
        return (dilate, cropped_copy, time_ellapsed)

    def track(self, preprocess_output):

        dilate, cropped_copy, time_ellapsed = preprocess_output                 
        image, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img, contours, -1, (255,0,0), 2)             
            
        if contours:            
            #If there is more than 1 detected contour we need some way of figuring
            #out which is is actually the fly. 
            if len(contours) > 1:
                contour_areas = np.array([cv2.contourArea(contour) for contour in contours])               
                #Filter contours by area and make sure area is between 30 and 190 units (empirical)
                masked_contour_areas = contour_areas*np.logical_and([contour_areas > 30],[contour_areas < 190])
                max_indx = masked_contour_areas.argmax()
                contour_area = contour_areas[max_indx]  
                contour = contours[max_indx]                
            if len(contours) == 1:
                contour = contours[0]
                contour_area = cv2.contourArea(contour)           

            if self.debug_mode:
                print "First contour in the given frame has area: {}".format(contour_area)            

            moments = cv2.moments(contour)               
            if moments['m00'] > 0:
                cx, cy = int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])  
                cv2.circle(cropped_copy, (cx, cy), 8, (255,255,255), thickness=1)

                self.last_fly_locations.append((np.array([cx,cy]), time_ellapsed)) 
                
                #---------- Code for if we want to do an occupancy ROI experiment ------------------
                if self.get_occupancy_roi:                    
                    in_rewd = cv2.pointPolygonTest(self.occupancy_contour[0], (cx, cy), False)                         
                    #print "\n\ninPolygon test gives: {}\n\n".format(in_contour)                          
                    if in_rewd == 1:
                        
                        cv2.circle(cropped_copy, (cx, cy), 11, (0,0, 255), thickness=2)
                        self.last_rewd_status.append('True')
                        if self.debug_mode:
                            print("The fly *IS* in the occupancy region!!")
                        
                        if self.use_arduino:
                            self.arduino_update_desired_state(1, self.arena_id)
                                                                                           
                            #if get_dist(occupancy_roi.roi[0]+obtained_rewards, occupancy_roi.roi[1]-obtained_rewards) > 5:                                          
                            #    occupancy_contour = create_rect_contour(occupancy_roi.roi[0]+obtained_rewards, occupancy_roi.roi[1]-obtained_rewards)
                    else:
                        self.last_rewd_status.append('False')                      
                        if self.debug_mode:
                            print("The fly is *NOT* in the occupancy region!!")
                            
                        if self.use_arduino:
                            self.arduino_update_desired_state(0, self.arena_id)
                        
                #---------- Code for if we want to do a movement based experiment -------------
                elif self.get_displacement:
                    try:
                        displacement_rate = get_disp_rate(self.last_fly_locations[0], self.last_fly_locations[1])
                        if displacement_rate >= 90:
                            self.last_rewd_status.append('True')
                            cv2.circle(cropped_copy, (cx, cy), 11, (0,0, 255), thickness=2)
                            if self.debug_mode:
                                print("The fly *IS* being rewarded!")
                            if self.use_arduino:
                                self.arduino_update_desired_state(1, self.arena_id)
                        else:
                            self.last_rewd_status.append('False')
                            if self.debug_mode:
                                print("The fly is *NOT* being rewarded")
                            if self.use_arduino:
                                self.arduino_update_desired_state(0, self.arena_id)
                        
                    except IndexError:
                        self.last_rewd_status.append('N/A')
                        pass
                
                #--------- If we're not doing either type of experiment just append an 'N/A' to status
                else:
                    self.last_rewd_status.append('N/A')
                            
                self.fly_location_array.append((time_ellapsed, cx, cy, self.last_rewd_status[-1]))    
            
            #use try instead of "if len(self.last_fly_locations) > 1:" as there is
            #less of a timecost incurred and the exception only occurs once!
            try:
                dist = get_dist(self.last_fly_locations[0][0],self.last_fly_locations[1][0])
                displacement_rate = get_disp_rate(self.last_fly_locations[0], self.last_fly_locations[1])
                self.last_displacement.append(dist)                       
                if self.debug_mode:
                    print("Fly's distance displaced from last detection is: {}".format(dist))
                    print("Fly dd/dt is: {}".format(displacement_rate))      
                    print("Arduino status is: {}".format(self.arduino.report_state()))
            except IndexError:
                pass
            
        #If no contour was found
        else:                      
            try:
                #We should assume that the current contour position is where the fly previously was
                temp_cx, temp_cy = self.last_fly_locations[-1][0]        
                cv2.circle(cropped_copy, (temp_cx, temp_cy), 8, (255,255,255), thickness=1)      
                        
                if self.last_rewd_status:
                    temp_rewd_status = self.last_rewd_status[-1]
                    if temp_rewd_status == 'True':
                        cv2.circle(cropped_copy, (temp_cx, temp_cy), 11, (0,0, 255), thickness=2)
                else:
                    temp_rewd_status = "N/A"         
                self.fly_location_array.append((time_ellapsed, temp_cx, temp_cy, temp_rewd_status))        
            except IndexError:
                pass
        #Finally, regardless of whether there is a found contour or not we should
        #always be updating the bg_subtractor learning rate
        try:
            #If the fly isn't moving, then we should stop updating the background subtractor
            #as we don't want the fly to become "background"
            if self.last_displacement[-1] <= 2.0:
                self.bgs_learning_rate = 0
                if self.debug_mode:
                    print "Learning rate is 0!\n\n"
            else:
                self.bgs_learning_rate = -1  
                if self.debug_mode:
                    print "Learning rate is auto!\n\n"                            
        except IndexError:
            pass   
            
        if self.write_video:
            self.tracking_video_writer.stdin.write(cropped_copy.tostring())
            
        cv2.imshow('{} contours drawn'.format(self.name), cropped_copy) 
                       
    def write_data_out(self, save_path, filename):
        with open(os.path.join(save_path, "{}.csv".format(filename)), "wb") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Time Elapsed (sec)", "Fly x", "Fly y", "Fly being rewarded?"])
            writer.writerows(self.fly_location_array)  

#we can attach an optional profiler to figure out where and how to optimize
#code efficiency
#@profile
def preprocess_and_track(preproc_dict, track_dict, arena_dict_keys, cropped_frame_list, time_stamp, arduino_obj):   
    #pre-processing step. See: preproces_frame()
    preproc_output = [preproc_dict[key](cropped_frame_list[indx], time_stamp) for indx, key in enumerate(arena_dict_keys)]
    #tracking step. See: track()
    for indx, key in enumerate(arena_dict_keys):
        track_dict[key](preproc_output[indx])
    if arduino_obj:
        arduino_obj.write_desired_state()
    
#%%
def start_fly_tracking(expt_dur = 900, led_freq = 5, led_pw=5, fps_cap = 30, 
                       debug_mode=False, use_arduino = False, 
                       write_video = True, write_csv = False, num_arenas=4,
                       define_occupancy_roi = False, movement_assay = False, 
                       cam_calib_file = "Camera_calibration_matrices.json",
                       default_save_dir = "C:\\Users\\Mixologist\\Desktop\\Karen's Operant Data"):
    if use_arduino:
        try:
            arduino = InitArduino()
        except:
            print "There was a problem connecting to the arduino!! Try restarting Python!"
    else:   
        arduino=None
  
    calibration_data = load_cam_calib_file(cam_calib_file)   
    calib_mtx = calibration_data["camera_matrix"]
    calib_dist = calibration_data["dist_coeff"]
      
    #Note if your laptop or computer has a built in webcam you may need to set
    #the videocapture element to 1 instead of 0
    temp_cam = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    #We don't want the camera to try to autogain as it messes up the image
    #These settings apparently don't work with our camera... (ELP-USBFHD01M-L21)
    #cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
    #cam.set(cv2.CAP_PROP_GAIN, 0.0)
    #So instead let's have the camera capture some frames and adapt the exposure accordingly
    for x in xrange(45):
        ret, temp_frame = temp_cam.read()       
    temp_frame = correct_distortion(temp_frame, calib_mtx, calib_dist)
    temp_cam.release()
     
    #Initialize Arena instances (and set up arena and occupancy rois)
    #We will create a dictionary where the each "Arena X" label will be the "key"
    #for each 'Arena()' class instance object
    arena_dict = {'Arena {}'.format(arena):Arena('Arena {}'.format(arena), arena, temp_frame, 
                                                 get_occupancy_roi=define_occupancy_roi, 
                                                 get_displacement=movement_assay,
                                                 write_video=write_video, 
                                                 debug_mode=debug_mode, 
                                                 fps_cap=fps_cap, use_arduino=use_arduino,
                                                 arduino_obj=arduino)
                  for arena in xrange(1, 1+num_arenas)}
                      
    arena_dict_keys = sorted(arena_dict.keys())
    
    arena_roi_list = [arena_dict[key].arena_roi for key in arena_dict_keys]
       
    vid_params = []
    if write_video:
        #Initialize raw video writers
        for arena_roi in arena_roi_list:
            cropped_sample_frame = crop_image(temp_frame, arena_roi)    
            vid_height, vid_width = cropped_sample_frame.shape[:2]
            vid_params.append((vid_height, vid_width, arena_roi.name))
                       
    expt_start_date = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")   
    save_path = os.path.join(default_save_dir, expt_start_date)   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if write_video:
        #Initialize annotated tracking video writers
        for key in arena_dict_keys:
            arena_dict[key].init_video_writer(save_path, expt_start_date)
                                                                                                                                                      
    #Initialize the multiprocess communication pipe and queue
    parent_conn, child_conn = mp.Pipe()
    data_q = mp.Queue()    
    #Initialize and start the video capture process
    proc_args = (child_conn, data_q, calib_mtx, calib_dist, fps_cap, 
                 arena_roi_list, vid_params, save_path, write_video)             
    video_cap_process = mp.Process(target=video_capture, args=proc_args)                                    
    video_cap_process.start()
    
    #Loop optimization by removing '.' function references
    get_data_q = data_q.get
    get_q_size = data_q.qsize
    print_flush = sys.stdout.flush
    np_ndarray = np.ndarray
    cv2_waitKey = cv2.waitKey
    #unpack the .proprocess frame method for each arena into a dictionary
    preproc_dict = {key: arena_dict[key].preprocess_frame for key in arena_dict_keys}
    #unpack the .track methods for each arena into a dictionary
    track_dict = {key: arena_dict[key].track for key in arena_dict_keys}
    
    #arduino.turn_on_stim(15, 5)
    #time.sleep(5)
    #arduino.turn_off_stim()
        
    parent_conn.send('Time:{}'.format(expt_start_date))      
    parent_conn.send('Start!')
    #give a bit of time for the child process to get started
    time.sleep(0.3)   
    prev_time_stamp = 0
          
    while 1:
        time_stamp, cropped_frame_list = get_data_q()
        
        if time_stamp > expt_dur:
            #let's close everything down
            parent_conn.send('Shutdown!')
            time.sleep(1)
            cv2.destroyAllWindows()
            #clean up the expt control process
            data_q.close()
            data_q.join_thread()
            child_conn.close()
            parent_conn.close()
            video_cap_process.terminate()
            break      
        
        #check if the frame we're getting is not a blank
        if type(cropped_frame_list[0]) is np_ndarray:                
            #print frame.dtype, frame.size
            #print (time_stamp, stim_bool)            
            fps = 1/(time_stamp-prev_time_stamp)
            prev_time_stamp = time_stamp           
            print('Lagged frames: {} fps: {}'.format(int(get_q_size()),fps))
            print_flush()

            #------------------ Heavy Lifting functions ---------------------
            #The majority of processing time and power go into these two functions
            preprocess_and_track(preproc_dict, track_dict, arena_dict_keys, cropped_frame_list, time_stamp, arduino)
                
        k = cv2_waitKey(30) & 0xff
        if k == 27:
            #let's close everything down
            parent_conn.send('Shutdown!')
            time.sleep(1)
            cv2.destroyAllWindows()
            #clean up the expt control process
            data_q.close()
            data_q.join_thread()
            child_conn.close()
            parent_conn.close()
            video_cap_process.terminate()
            break

    #print fly_location_array
    if use_arduino:
        arduino.turn_off_stim()
        arduino.turn_off_solenoids()
        arduino.close()
        
    if write_csv:
        for key in arena_dict_keys:
            arena_dict[key].write_data_out(save_path, "{} - {}".format(expt_start_date, key))
            
    if write_video:
        for key in arena_dict_keys:
            arena_dict[key].close_video_writer()
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':    
    start_fly_tracking(expt_dur = 600, write_video=True, 
                       define_occupancy_roi=True, movement_assay=False, 
                       use_arduino=True, write_csv=True, num_arenas=4, debug_mode=True)
    