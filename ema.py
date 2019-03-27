#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:19:42 2017

@author: ubuntu
"""
import numpy as np
from numpy import cross,dot
from numpy.linalg import norm
import pandas as pd
import os

def read_ndi_data(mydir, file_name,sensors,subcolumns):
    '''
    Read data produced by NDI WaveFront software
    skip empty columns, sensor not OK data set to 'nan'

    Input 
        mydir- directory where biteplate file will be found 
        file_name - name of a biteplate calibration recording
        sensors - a list of sensors in the recording
        subcolumns - a list of info to be found for each sensor
      
    Output  
        df - a pandas dataframe representation of the whole file
    '''

    fname = os.path.join(mydir, file_name)

    better_head = ['time'] + \
        ['{}_{}'.format(s, c) for s in sensors for c in subcolumns]

    # Deal with empty columns, which have empty header values.
    with open(fname, 'r') as f:
        filehead = f.readline().rstrip()
    headfields = filehead.split('\t')
    indices = [i for i, x in enumerate(headfields) if x == ' ']

    for count, idx in enumerate(indices):
        better_head.insert(idx, 'EMPTY{:}'.format(count))
    
    ncol_file = len(headfields)
    ncol_sens = len(better_head)
    if ncol_file > ncol_sens:
        raise ValueError("too few sensors are specified")
    if ncol_file < ncol_sens:
        raise ValueError("too many sensors are specified")

    df = pd.read_csv(fname, sep='\t', index_col = False,
        header=None,            # The last three parameters
        skiprows=1,             # are used to override
        names=better_head       # the existing file header.
    )

    for s in sensors:   # clean up the data - xyz are nan if state is not ok
        state = '{}_state'.format(s)
        if str(df.loc[0,state])=='nan':  # here skipping non-existant sensors,
            continue            # perhaps a cable not plugged in
        locx = '{}_x'.format(s)
        locz = '{}_z'.format(s)
        cols = list(df.loc[:,locx:locz])
        
        df.loc[df.loc[:,state]!="OK",cols]=[np.nan,np.nan,np.nan]   
    return df

def get_referenced_rotation(df):
    '''
    given a dataframe representation of a biteplate recording, find rotation matrix 
         to put the data on the occlusal plane coordinate system

    Input
        df - a dataframe read from a biteplate calibration recording
            sensor OS is the origin of the occlusal plane coordinate system
            sensor MS is located on the biteplate some distance posterior to OS

    Output 
        OS - the origin of the occlusal plane coordinate system
        m - a rotation matrix
    '''

    MS = df.loc[:, ['MS_x', 'MS_y', 'MS_z']].mean(skipna=True).as_matrix()
    OS = df.loc[:, ['OS_x', 'OS_y', 'OS_z']].mean(skipna=True).as_matrix()
    REF = np.array([0, 0, 0])
        
    ref_t = REF-OS   # the origin of this space is OS, we will rotate around this
    ms_t = MS-OS
       
    z = cross(ms_t,ref_t)  # z is perpendicular to ms and ref vectors
    z = z/norm(z)
    
    y = cross(z,ms_t)        # y is perpendicular to z and ms
    y = y/norm(y)
    
    x = cross(z,y)
    x = x/norm(x)
       
    m = np.array([x, y, z])    # rotion matrix directly

    return OS, m

def get_desired_head_location(df):  
    ''' get the desired positions of three points - nasion, right mastoid, left mastoid (REF, RMA, LMA)
        so that the translation and rotation of these points will correct for head movement, and put
        the data onto an occlusal plane coordinate system.  
        
        The location of the occlusal plane is given by a bite-plate, and the triangle formed by REF (nasion), 
        OS (origin sensor), and MS (molar sensor), which are in the saggital plane.
        
        Input - a dataframe that has points 
            REF, RMA, LMA, OS, and MS
            
        Output - 
            desired positions of REF, RMA, and LMA
    '''
    # The relative locations of these is fixed - okay to operate on means
    MS = df.loc[:, ['MS_x', 'MS_y', 'MS_z']].mean(skipna=True).as_matrix()
    OS = df.loc[:, ['OS_x', 'OS_y', 'OS_z']].mean(skipna=True).as_matrix()
    REF = df.loc[:,['REF_x', 'REF_y', 'REF_z']].mean(skipna=True).as_matrix()
    RMA= df.loc[:, ['RMA_x', 'RMA_y', 'RMA_z']].mean(skipna=True).as_matrix()
    LMA = df.loc[:, ['LMA_x', 'LMA_y', 'LMA_z']].mean(skipna=True).as_matrix()
    
    # 1) start by translating the space so OS is at the origin
    ref_t = REF-OS   
    ms_t = MS-OS
    rma_t = RMA-OS
    lma_t = LMA-OS
    os_t = np.array([0,0,0])
    
    # 2) now find the rotation matrix to the occlusal coordinate system
    z = cross(ms_t,ref_t)  # z is perpendicular to ms and ref vectors
    z = z/norm(z)
    
    y = cross(z,ms_t)        # y is perpendicular to z and ms
    y = y/norm(y)
    
    x = cross(z,y)
    x = x/norm(x)
       
    m = np.array([x, y, z])    # rotion matrix directly
    
    # 3) now rotate the mastoid points - using the rotation matrix
    rma_t = dot(rma_t,m.T) 
    lma_t = dot(lma_t,m.T)
    
    return ref_t, rma_t, lma_t
    
def read_referenced_biteplate(my_dir,file_name,sensors,subcolumns):
    ''' 
    Input 
        mydir- directory where biteplate file will be found 
        file_name - name of a biteplate calibration recording
        sensors - a list of sensors in the recording
        subcolumns - a list of info to be found for each sensor

    Output  
        OS - the origin of the occlusal plane coordinate system
        m - a rotation matrix based on the quaternion
    '''

    bpdata = read_ndi_data(my_dir,file_name,sensors,subcolumns)
    [OS,m] = get_rotation(bpdata)
    return OS, m


def read_3pt_biteplate(my_dir,file_name,sensors,subcolumns):
    ''' 
    Input 
        mydir- directory where biteplate file will be found 
        file_name - name of a biteplate calibration recording
        sensors - a list of sensors in the recording
        subcolumns - a list of info to be found for each sensor

    Output  
        desired positions of the head location sensors: REF, RMA, and LMA
            (nasion, right mastoid, left mastoid)
    '''

    bpdata = read_ndi_data(my_dir,file_name,sensors,subcolumns)
    [REF, RMA, LMA] = get_desired_head_location(bpdata)
    return REF, RMA, LMA

def rotate_referenced_data(df,m,origin, sensors):
    ''' 
    This function can be used when NDI head correction is used.  All we need is a translation vector
    and a rotation matrix, to move the data into the occlusal coordinate system.
    
    Input
        df - a pandas dataframe read by read_ndi_data
        m  - a rotation matrix computed by read_biteplate
        sensors - a list of the sensors to expect in the file
            specifically we exect to find columns with these names plus "_x", "_y" and "_z"
            
    Output
        df - the dataframe with the xyz locations of the sensors translatedd and rotated
    '''

    # TODO:  remove quaternion data, or fix it.
    for s in sensors:  # read xyz one sensor at a time
        locx = '{}_x'.format(s)
        locz = '{}_z'.format(s)
        cols = list(df.loc[:,locx:locz])  # get names of columns to read

        points = df.loc[:,cols].values   # read data
        if s=="REF":
            points = [0,0,0]
        points = points - origin      # translate
        df.loc[:,cols] = dot(points,m.T) # rotate - put back in the dataframe

    return df

def head_correct_and_rotate(df,REF,RMA,LMA):
    '''This function uses the previously calculated desired locations of three sensors 
    on the head -- nasion (REF), right mastoid (RMA), and left mastoid (LMA) and based 
    on the locations of those sensors in each frame, finds a translation and rotation 
    for the frame's data, and then applies those to each sensor in that frame.
    '''
    
    # for each frame
    # 1) find the translation and rotation that will move the head into the occlusal coordinate system
    #              this is where we use Horns direct method of fitting to an ideal triangle
    # 2) apply the translation and rotation to each sensor in the frame.
    
    ''' Question:  should we smooth the head position sensors prior to head correction?
        A reason to do this is that we can then avoid loosing any data due to calibration sensor dropout
        (assuming that missing frames are rare and can be interpolated).  Smoothing might also produce more 
        accurate data because we constrain our estimate of the the location of the head (a very slow moving 
        structure) by neighboring points in time.
    '''
    
def save_rotated(mydir,fname,df,myext = 'ndi'):
    '''
    save the rotated data as *.ndi
    
    Input
        mydir - directory where the data will be found
        fname - the name of the original .tsv file
        df - a pandas dataframe containing the processed/rotated data
    '''

    fname = os.path.join(mydir,fname)
    
    name,ext = os.path.splitext(fname)
    processed = name + '.' + myext
    
    df.to_csv(processed, sep="\t", index=False)
