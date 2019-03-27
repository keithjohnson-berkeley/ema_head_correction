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

def get_rotation(df):
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
 
def read_biteplate(my_dir,file_name,sensors,subcolumns):
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

def rotate_data(df,m,origin, sensors):
    ''' 
    Input
        df - a pandas dataframe read by read_ndi_data
        m  - a rotation matrix computed by read_biteplate
        sensors - a list of the sensors to expect in the file
            specifically we exect to find columns with these names plus "_x", "_y" and "_z"
            
    Output
        df - the dataframe with the xyz locations of the sensors rotated
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
