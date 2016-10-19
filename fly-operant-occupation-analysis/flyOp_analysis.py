# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:48:46 2015

@author: Nicholas Mei

Summarized plotting script that takes an experiment condition folder 
and plot relevant behavioral data 

"""

"""
Installing rpy2 and R

1) Get R: https://cran.r-project.org/bin/windows/
2) Get rpy2 from: http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2
3) Use pip to install the rpy2 .whl from step 2
4) Now you need to go to the system environment variables and add:
    4a) "C:\Program Files\R\R-3.2.2\bin" to the PATH variable
    4b) "R_USER" "Computer User Name" to environment variable
    see: http://stackoverflow.com/questions/24414540/rpy2-error-wac-a-mole-r-user-not-defined
"""

import sys
import os
import math  
import glob

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

#code to make .pdf versions of plots importable with illustrator
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42

from scipy.stats import gaussian_kde
import rpy2.robjects as robjects
import cv2

#If we are using python 2.7 or under
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
      
#If we are using python 3.0 or above
elif sys.version_info[0] >= 3:
    import tkinter as tk
    import tkinter.filedialog as filedialog
   
#%%
#Helper functions   
   
def chooseDir(cust_text):
    root = tk.Tk()
    try:
        root.tk.call('console','hide')
    except tk.TclError:
        pass
    
    baseFilePath = "C:/Users/Mixologist/Desktop/Operant Occupancy Assay"
    directoryPath = filedialog.askdirectory(parent = root, title=cust_text, initialdir= baseFilePath)
    root.destroy()
    
    return directoryPath

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    
    Written by: Unutbu http://stackoverflow.com/a/25941474
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    
    Written by: Unutbu http://stackoverflow.com/a/25941474
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
    
#%%
#Data visualization functions
    
def plot_fly_occupancy_over_time(path_to_data_dir=None):
    rois_to_analyze = ['Arena 1', 'Arena 2', 'Arena 3', 'Arena 4']
    
    if not path_to_data_dir:
        path_to_data_dir = chooseDir("Please select the experiment condition you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_data_dir)) 
        
    if os.path.isdir(path_to_data_dir):
        
        fig = plt.figure()
        fig.suptitle("Fly ROI Occupancy Over Time", fontsize=16, fontweight='bold')
            
        gs = gridspec.GridSpec(4, 1)   
        
        for indx, roi in enumerate(rois_to_analyze):
            ax = fig.add_subplot(gs[indx])  
            ax.tick_params(axis='x', which='both', bottom='off', top='off')    
            ax.set_title(roi, fontsize=12, fontweight='bold')
            
            file_to_analyze = glob.glob('{basedir}/*{roi_name}.csv'.format(basedir = path_to_data_dir, roi_name = roi))[0]         
            roi_df = pd.read_csv(file_to_analyze)
            
            roi_occupancy = roi_df['Fly being rewarded?']
            time_points = roi_df['Time Elapsed (sec)']
            
            ax.plot(time_points.values, roi_occupancy.values)
            ax.fill_between(time_points.values, 0, roi_occupancy.values)
            ax.yaxis.set_visible(False)
            
        ax.set_xlabel("Time (secs)")    
            
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
             
            
def plot_position_densities(path_to_data_dir=None, title=None):
    rois_to_analyze = ['Arena 1', 'Arena 2', 'Arena 3', 'Arena 4']
    
    if not path_to_data_dir:
        path_to_data_dir = chooseDir("Please select the experiment condition you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_data_dir))
    
    if os.path.isdir(path_to_data_dir):
        
        fig = plt.figure()
        fig.suptitle("Fly Position Density\n{}".format(title), fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(1, 4)      
        
        for indx, roi in enumerate(rois_to_analyze):
            
            ax = fig.add_subplot(gs[indx])             
            
            file_to_analyze = glob.glob('{basedir}/*{roi_name}.csv'.format(basedir = path_to_data_dir, roi_name = roi))[0]
            video_to_analyze = glob.glob('{basedir}/*- tracking - {roi_name}.avi'.format(basedir = path_to_data_dir, roi_name = roi))[0]      
            #print file_to_analyze
            #print video_to_analyze
            vid = cv2.VideoCapture(video_to_analyze)
            ret, first_frame = vid.read()
            if ret:
                vid.release()
            else:
                print("Error could not read {}".format(video_to_analyze))
            
            roi_df = pd.read_csv(file_to_analyze)
            x = roi_df['Fly x']
            y = roi_df['Fly y']
             
            # Calculate the point density
            xy = np.vstack([x,y])
            # This is super slow!!
            z = gaussian_kde(xy)(xy)
            
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            
            ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')    
            ax.set_title(roi, fontsize=14)
            ax.imshow(first_frame)
            ax.scatter(x, y, c=z, s=10, edgecolor='', cmap='viridis')
                       
        plt.subplots_adjust(top=0.75)
            
    else:
        print("Error, invalid *directory* path provided: {}".format(path_to_data_dir))

#test csv file: 
#testfile = u'C:/Users/Mixologist/Desktop/Operant Occupancy Assay/2015-11-09 14.11.40 - 15 min\\2015-11-09 14.11.40 - Arena 1.csv'
#testvid = u'C:/Users/Mixologist/Desktop/Operant Occupancy Assay/2015-11-09 14.11.40 - 15 min\\video - 2015-11-09 14.11.40 - Arena 1.avi'


def plot_positions(path_to_data_dir=None, title=None):
    rois_to_analyze = ['Arena 1', 'Arena 2', 'Arena 3', 'Arena 4']
    
    if not path_to_data_dir:
        path_to_data_dir = chooseDir("Please select the experiment condition you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_data_dir))  
    
    if os.path.isdir(path_to_data_dir):
        
        fig = plt.figure()
        fig.suptitle("Fly Position Over Time (dark to bright)\n{}".format(title), fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(1, 4)      
        
        for indx, roi in enumerate(rois_to_analyze):
            
            ax = fig.add_subplot(gs[indx])             
            
            file_to_analyze = glob.glob('{basedir}/*{roi_name}.csv'.format(basedir = path_to_data_dir, roi_name = roi))[0]
            video_to_analyze = glob.glob('{basedir}/*- tracking - {roi_name}.avi'.format(basedir = path_to_data_dir, roi_name = roi))[0]      
            #print file_to_analyze
            #print video_to_analyze
            vid = cv2.VideoCapture(video_to_analyze)
            ret, first_frame = vid.read()
            if ret:
                vid.release()
            else:
                print("Error could not read {}".format(video_to_analyze))
            
            roi_df = pd.read_csv(file_to_analyze)
            
            x = roi_df['Fly x']
            y = roi_df['Fly y']
            z = np.linspace(0, 1, len(x))
            
            ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')           
            ax.imshow(first_frame)  
            ax.set_title(roi, fontsize=14)
            colorline(ax, x, y, z, cmap=plt.get_cmap('viridis'), linewidth=1)
            
        plt.subplots_adjust(top=0.75)
            
    else:
        print("Error, invalid *directory* path provided: {}".format(path_to_data_dir))
 
#%%
#Quantitation/statistics and plotting of group averages
def calculate_occupancy_statistics(path_to_data_dir=None):
    rois_to_analyze = ['Arena 1', 'Arena 2', 'Arena 3', 'Arena 4']
      
    if not path_to_data_dir:
        path_to_data_dir = chooseDir("Please select the experiment condition you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_data_dir)) 

    if os.path.isdir(path_to_data_dir):
        
        arena_dict = {}
        
        for indx, roi in enumerate(rois_to_analyze):          
            
            occupancy_dict = {}            
            
            file_to_analyze = glob.glob('{basedir}/*{roi_name}.csv'.format(basedir = path_to_data_dir, roi_name = roi))[0]         
            roi_df = pd.read_csv(file_to_analyze)
            
            roi_occupancy = roi_df['Fly being rewarded?']
            time_points = roi_df['Time Elapsed (sec)']
            
            #time_points_in_occupancy = roi_occupancy.dropna().sum()
            #trial_percent_occupancy = time_points_in_occupancy/float(len(time_points))
            
            # Insert a [False] at both the start and end of roi_occupancy boolean array
            # This will help catch edge cases where an entry has no exit
            # Or when the fly starts in the occupation roi and ends in the occupation roi
            roi_occupancy = roi_occupancy.values               
            roi_occupancy = np.append([False], roi_occupancy)
            roi_occupancy = np.append(roi_occupancy, [False])
                       
            # subtract the roi_occupancy with roi_occupancy shifted by 1 timepoint as integers
            # EX: F,F,F,F,T,T,T,T,T,F,F,F - F,F,F,F,F,T,T,T,T,T,F,F   
            #     0,0,0,0,1,1,1,1,1,0,0,0 - 0,0,0,0,0,1,1,1,1,1,0,0 = 0,0,0,0,1,0,0,0,0,-1,0,0
            diff = roi_occupancy.astype(int) - np.roll(roi_occupancy.astype(int), 1)      
            
            # Remove inserted [False] elements at start and end so that 
            # index of diff will line back up with "time_points"
            diff = np.delete(diff, 0)
            diff = np.delete(diff, -1)
            
            occupancy_dict['Occupancy ROI Entries'] = np.sum(diff == 1)
            
            entry_indices = np.where(diff == 1)[0]
            exit_indices = np.where(diff == -1)[0]

            # Check if the entry and exit indices length match
            # In theory you might have more entries than exits if expt finishes
            # while fly is still in occupancy ROI, But you should never have 
            # more exits than entries
            if len(entry_indices) > len(exit_indices):
                entry_indices = entry_indices[0:len(exit_indices)]
                print("Warning! There were more entry indices than exit indices for {}!".format(roi))
            elif len(exit_indices) > len(entry_indices):
                print("Warning! Something is seriously wrong as you have more exits than entries for {}!".format(roi))           
            try:            
                occupancy_durations = time_points[exit_indices].values - time_points[entry_indices].values
            except ValueError as error:
                print("Error! Number of entry indices does not match number of exit indices for {}!".format(roi))
                print(error)         
                
            sys.stdout.flush()
            
            occupancy_dict['Occupancy ROI Dwell Durations'] = occupancy_durations
            
            occupancy_dict['Trial Occupancy ROI Dwell Time'] = occupancy_durations.sum()
            occupancy_dict['Trial Occupancy ROI Dwell Time Fraction'] = occupancy_durations.sum()/time_points.iloc[-1]
            
            occupancy_dict['Average Occupancy ROI Dwell Time'] = occupancy_durations.mean()
            occupancy_dict['SEM of Occupancy ROI Dwell Time'] = sp.stats.sem(occupancy_durations)

            arena_dict[roi] = occupancy_dict
            
        return pd.DataFrame(arena_dict)
    
    else:
        print("Error, invalid *directory* path provided: {}".format(path_to_data_dir))        
        
def add_staggered_points(ax, x_anchor, y_values, spread=0.04, color='b', marker='o', alpha=0.3, linestyle='-'):
    x = np.random.normal(x_anchor, spread, size=len(y_values))
    ax.plot(x, y_values, c=color, marker=marker, ms=4, alpha=alpha, linestyle=linestyle)
    return x
    
def plot_train_test_occupancy_entries(train_df, test_df, title=None):
    
    train_occupancy_entries = train_df.loc['Occupancy ROI Entries'].values
    test_occupancy_entries = test_df.loc['Occupancy ROI Entries'].values
    
    # Note about R wilcox.test - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
    # if both x and y are given and paired is TRUE, a Wilcoxon signed rank test of the null that the distribution of x - y (in the paired two sample case) is symmetric about mu is performed.
    wilcox_test = robjects.r['wilcox.test']
    p_val = wilcox_test(robjects.IntVector(train_occupancy_entries), robjects.IntVector(test_occupancy_entries), paired = True).rx2('p.value')[0]
    p_val_statement = "Day 1 vs. Day 2 Occupancy ROI Entries P-Value: {}".format(p_val)
    print(p_val_statement)    
    
    #Occupancy ROI Entries graph        
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle(title, fontsize=16, fontweight='bold')    
    ax = plt.subplot()  
    ax.set_title(p_val_statement)
    ax.set_ylabel("Number of entries into occupancy ROI")
    ax.yaxis.labelpad = 20
    ax.set_xlabel("Experimental condition")        
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Day 1', 'Day 2'])
    ax.tick_params(top="off",right="off", bottom="off") 
    ax.tick_params(axis='both', pad=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    
    (plotlines, caplines, barlines) = ax.errorbar(x=[1,2], y=[train_df.loc['Occupancy ROI Entries'].mean(), test_df.loc['Occupancy ROI Entries'].mean()], yerr=[train_df.loc['Occupancy ROI Entries'].sem(), test_df.loc['Occupancy ROI Entries'].sem()], capsize=0, lw=2.5, elinewidth=1.5, color="black")
    
    x1 = add_staggered_points(ax, 1, train_df.loc['Occupancy ROI Entries'].values, color='b', marker='o', alpha=0.3, linestyle='None')             
    x2 = add_staggered_points(ax, 2, test_df.loc['Occupancy ROI Entries'].values, color='b', marker='o', alpha=0.3, linestyle='None')
    
    line_x_coords = zip(x1, x2)
    line_y_coords = zip(train_df.loc['Occupancy ROI Entries'].values, test_df.loc['Occupancy ROI Entries'].values)
    
    for x_set, y_set in zip(line_x_coords, line_y_coords):
        ax.plot(x_set, y_set, c='b', alpha=0.5)
    
    if p_val < 0.1:                
        y_min, y_max = ax.get_ylim()
        ax.plot([1, 2], [y_max+5,y_max+5], 'k-', lw=2)      
        ax.set_ylim(y_min, y_max+20)
        
        if p_val < 0.001:
            ax.text(1.5, y_max+5.5, '***')
        elif p_val < 0.01:
            ax.text(1.5, y_max+5.5, '**')
        elif p_val < 0.05:
            ax.text(1.5, y_max+5.5, '*')
        elif p_val < 0.1:
            ax.text(1.5, y_max+5.5, '+')    

    fig.add_subplot(ax)
    
def plot_train_test_dwell_times(train_df, test_df, title=None):
    
    color_cycle = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] * int(math.floor(train_df.shape[1]/4))
       
    train_df_hstack = np.hstack(train_df.loc['Occupancy ROI Dwell Durations'].values)
    test_df_hstack = np.hstack(test_df.loc['Occupancy ROI Dwell Durations'].values)
    
    #This is the Mann-Whitney test because paired is set to "False"
    wilcox_test = robjects.r['wilcox.test']
    p_val = wilcox_test(robjects.FloatVector(train_df_hstack), robjects.FloatVector(test_df_hstack), paired=False).rx2('p.value')[0]
    p_val_statement = "Day 1 vs. Day 2 Occupancy ROI Dwell Time Per Entry P-Value: {}".format(p_val)
    print(p_val_statement)  
    
    train_df_mean = train_df_hstack.mean()
    train_df_sem = sp.stats.sem(train_df_hstack)
    
    test_df_mean = test_df_hstack.mean()
    test_df_sem = sp.stats.sem(test_df_hstack)
    
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    ax = plt.subplot()  
    ax.set_title(p_val_statement)
    ax.set_ylabel("Occupancy ROI Dwell Time (Sec)")
    ax.yaxis.labelpad = 20
    ax.set_xlabel("Experimental condition") 
    ax.set_xlim(0.5, 2.75)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Day 1', 'Day 2'])
    ax.tick_params(top="off",right="off", bottom="off") 
    ax.tick_params(axis='both', pad=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 

    for indx, df_label in enumerate(train_df.columns):
        add_staggered_points(ax, 1, train_df.loc['Occupancy ROI Dwell Durations'][df_label], spread=0.08, color=color_cycle[indx], marker='o', alpha=1, linestyle='None')
        add_staggered_points(ax, 2, test_df.loc['Occupancy ROI Dwell Durations'][df_label], spread=0.08, color=color_cycle[indx], marker='o', alpha=1, linestyle='None')

    (plotlines, caplines, barlines) = ax.errorbar(x=[1,2], y=[train_df_mean, test_df_mean], yerr=[train_df_sem, test_df_sem], capsize=0, lw=2.5, elinewidth=1.5, color="black")

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min-10, y_max)
    
    if p_val < 0.1:                
        y_min, y_max = ax.get_ylim()
        ax.plot([1, 2], [y_max+5,y_max+5], 'k-', lw=2)      
        ax.set_ylim(y_min, y_max+15)
        
        if p_val < 0.001:
            ax.text(1.5, y_max+5.5, '***')
        elif p_val < 0.01:
            ax.text(1.5, y_max+5.5, '**')
        elif p_val < 0.05:
            ax.text(1.5, y_max+5.5, '*')
        elif p_val < 0.1:
            ax.text(1.5, y_max+5.5, '+')              
            
    #legend            
    patches = [mpatches.Patch(color=color_cycle[indx], label='{}'.format(roi_name)) for indx, roi_name in enumerate(train_df.columns)]
    plt.legend(handles=patches)
   
    fig.add_subplot(ax)     
    
def plot_train_test_total_dwell_time(train_df, test_df, title=None):
    total_train_dwell_times = train_df.loc['Trial Occupancy ROI Dwell Time'].values
    total_test_dwell_times = test_df.loc['Trial Occupancy ROI Dwell Time'].values
        
    wilcox_test = robjects.r['wilcox.test']
    p_val = wilcox_test(robjects.FloatVector(total_train_dwell_times), robjects.FloatVector(total_test_dwell_times), paired=True).rx2('p.value')[0]
    p_val_statement = "Day 1 vs. Day 2 Occupancy ROI Total Dwell Time P-Value: {}".format(p_val)
    print(p_val_statement)    
    
    #Occupancy ROI Entries graph        
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle(title, fontsize=16, fontweight='bold')    
    ax = plt.subplot()  
    ax.set_title(p_val_statement)
    ax.set_ylabel("Total dwell time in occupancy ROI (Sec)")
    ax.yaxis.labelpad = 20
    ax.set_xlabel("Experimental condition")        
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Day 1', 'Day 2'])
    ax.tick_params(top="off",right="off", bottom="off") 
    ax.tick_params(axis='both', pad=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    
    (plotlines, caplines, barlines) = ax.errorbar(x=[1,2], y=[train_df.loc['Trial Occupancy ROI Dwell Time'].mean(), test_df.loc['Trial Occupancy ROI Dwell Time'].mean()], yerr=[train_df.loc['Trial Occupancy ROI Dwell Time'].sem(), test_df.loc['Trial Occupancy ROI Dwell Time'].sem()], capsize=0, lw=2.5, elinewidth=1.5, color="black")    
    
    x1 = add_staggered_points(ax, 1, train_df.loc['Trial Occupancy ROI Dwell Time'].values, color='b', marker='o', alpha=0.3, linestyle='None')             
    x2 = add_staggered_points(ax, 2, test_df.loc['Trial Occupancy ROI Dwell Time'].values, color='b', marker='o', alpha=0.3, linestyle='None')
    
    line_x_coords = zip(x1, x2)
    line_y_coords = zip(train_df.loc['Trial Occupancy ROI Dwell Time'].values, test_df.loc['Trial Occupancy ROI Dwell Time'].values)
    
    for x_set, y_set in zip(line_x_coords, line_y_coords):
        ax.plot(x_set, y_set, c='b', alpha=0.5)
    
    if p_val < 0.1:                
        y_min, y_max = ax.get_ylim()
        ax.plot([1, 2], [y_max+5,y_max+5], 'k-', lw=2)      
        ax.set_ylim(y_min, y_max+20)
        
        if p_val < 0.001:
            ax.text(1.5, y_max+5.5, '***')
        elif p_val < 0.01:
            ax.text(1.5, y_max+5.5, '**')
        elif p_val < 0.05:
            ax.text(1.5, y_max+5.5, '*')
        elif p_val < 0.1:
            ax.text(1.5, y_max+5.5, '+')    

    fig.add_subplot(ax)

def multi_train_test_occupancy_comparisons(path_to_parent_train_dir=None, path_to_parent_test_dir=None):
    """
    This function concatenates (groups together) all the data for Day 1 and concatenates all the data for Day 2 and performs comparisons
    
    Note: Choose the folder that contains each experimental folder
    """
    if not path_to_parent_train_dir:
        path_to_parent_train_dir = chooseDir("Please select the training directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_parent_train_dir))
        
    if not path_to_parent_test_dir:
        path_to_parent_test_dir = chooseDir("Please select the test directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_parent_test_dir))
    
    if path_to_parent_train_dir and path_to_parent_test_dir:
        
        #first determine all the individual experiment folder names in our "condition folder" (condition being like 90% EtOH or 80% EtOH, etc...)
        train_dirs = glob.glob('{basedir}/*'.format(basedir=path_to_parent_train_dir))
        test_dirs = glob.glob('{basedir}/*'.format(basedir=path_to_parent_test_dir))
        
        train_dfs = [calculate_occupancy_statistics(train_dir) for train_dir in train_dirs]
        test_dfs = [calculate_occupancy_statistics(test_dir) for test_dir in test_dirs]
        
        def join_dfs(df_list):      
            joined_df = df_list[0].copy()
            for indx, df in enumerate(df_list[1:], start=1):
                joined_df = joined_df.join(df, lsuffix="-{}".format(indx), rsuffix="-{}".format(indx+1))
            return joined_df
            
        joined_train_df = join_dfs(train_dfs)
        joined_test_df = join_dfs(test_dfs)
        
        plot_train_test_occupancy_entries(joined_train_df, joined_test_df, title='')       
        plot_train_test_dwell_times(joined_train_df, joined_test_df, title='')
        plot_train_test_total_dwell_time(joined_train_df, joined_test_df, title='') 

def train_test_occupancy_comparisons(path_to_train_dir=None, path_to_test_dir=None):
    """
    This function is for comparing day 1 (train) vs. day 2 (test) for one experiment set.
    """
    
    if not path_to_train_dir:
        path_to_train_dir = chooseDir("Please select the training directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_train_dir))
        
    if not path_to_test_dir:
        path_to_test_dir = chooseDir("Please select the test directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_test_dir))
    
    if path_to_train_dir and path_to_test_dir:
        #plot_positions(path_to_train_dir)
        #plot_positions(path_to_test_dir)
        
        train_df = calculate_occupancy_statistics(path_to_train_dir)
        test_df = calculate_occupancy_statistics(path_to_test_dir)
        
        #Get the title 
        title  = "{}".format(path_to_train_dir.split('/')[-1])
              
        plot_train_test_occupancy_entries(train_df, test_df, title=title)       
        plot_train_test_dwell_times(train_df, test_df, title=title)
        plot_train_test_total_dwell_time(train_df, test_df, title=title)        
        
        plot_positions(path_to_train_dir, title=title + ' - Day 1')
        plot_positions(path_to_test_dir, title=title + ' - Day 2')
        
        #plot_position_densities(path_to_train_dir, title=title + ' - Train')
        #plot_position_densities(path_to_test_dir, title=title + ' - Test')

        #use train_df.loc['label'] to slice by row instead of column

    else:
        print("One of the directories you specified was invalid:\n {}\n{}".format(path_to_train_dir,path_to_test_dir))


##
#Analysis of total distance moved figure out #entries/distance moved
##

def analyze_within_fly_statistics(path_to_parent_train_dir=None, path_to_parent_test_dir=None):
    """
    This function will compare a fly's performance on Day 1 with its performance on Day 2.
    Specifically, it will make this comparison over all experiments in a "condition folder"
    """
    
    if not path_to_parent_train_dir:
        path_to_parent_train_dir = chooseDir("Please select the training directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_parent_train_dir))
        
    if not path_to_parent_test_dir:
        path_to_parent_test_dir = chooseDir("Please select the test directory you would like to have analyzed")    
        print("The data directory you've chose is: {}".format(path_to_parent_test_dir))
    
    if path_to_parent_train_dir and path_to_parent_test_dir:
        #first determine all the individual experiment folder names in our "condition folder" (condition being like 90% EtOH or 80% EtOH, etc...)
        train_dirs = glob.glob('{basedir}/*'.format(basedir=path_to_parent_train_dir))
        test_dirs = glob.glob('{basedir}/*'.format(basedir=path_to_parent_test_dir))
        
        paired_dfs = [(calculate_occupancy_statistics(train_dir), calculate_occupancy_statistics(test_dir)) for train_dir, test_dir in zip(train_dirs, test_dirs)]   

        total_occupancy_time_within_fly_differences = []
        total_roi_entry_within_fly_differences = []

        for train_df, test_df in paired_dfs:
            total_occupancy_diff = test_df.loc["Trial Occupancy ROI Dwell Time"] - train_df.loc["Trial Occupancy ROI Dwell Time"]
            total_occupancy_time_within_fly_differences.append(total_occupancy_diff)
            total_entries_diff = test_df.loc["Occupancy ROI Entries"] - train_df.loc["Occupancy ROI Entries"]
            total_roi_entry_within_fly_differences.append(total_entries_diff)
        
        print(total_occupancy_time_within_fly_differences)
        print(total_roi_entry_within_fly_differences)
#plot_positions()