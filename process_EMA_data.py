#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 14:25:38 2017

@author: Keith Johnson

"""
from PyQt5.QtWidgets import (QMainWindow, QGroupBox,QApplication, QLineEdit, 
                             QAction, QComboBox, QLabel, QHBoxLayout,
                             QFileDialog, QPushButton, QDialog, QVBoxLayout)
from PyQt5.QtGui import QDoubleValidator

import numpy as np
import os, sys, re
from itertools import cycle
import ema

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Main(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.bpname = 'select a BitePlate file'
        self.palname = 'select a Palate trace file'
        self.dataname = 'select'
        self.base_directory = '/media'
        self.nchannels = '16'
        self.sensors = ["REF","UL","LL","JW","TT","TB","TD","TL","LC","UI","J","OS","MS","PL"]
        self.bpsensors = self.sensors
        self.PAL_sensors = self.sensors
        self.subcolumns = ["ID","frame","state","q0","qx","qy","qz","x","y","z"]
        self.pal_start = 0
        self.pal_end = 15
        self.setGeometry(100,100,300,600)
 
        self.initUI()
        
        
    def initUI(self):               
        lo = QVBoxLayout()
        frame = QGroupBox()
        self.setCentralWidget(frame)


        frame.setLayout(lo)
        
        menubar = self.menuBar()
        self.statusBar().showMessage('Ready')
        
        # -------- this section makes the menu bar --------------------
        exitAct = QAction('&Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(app.quit)
        
        one_Act = QAction('process one file', self) 
        one_Act.setShortcut('Ctrl+O')
        one_Act.setStatusTip('Open and process one file')
        one_Act.triggered.connect(self.process_one_file)
        
        all_Act = QAction('process &All files',self)
        all_Act.setShortcut('Ctrl+A')
        all_Act.setStatusTip('Process and save the results for all files')
        all_Act.triggered.connect(self.process_lots_of_files)
  
        numchannels = QAction('set the number of &Channels',self)
        numchannels.setShortcut('Ctrl+C')
        numchannels.setStatusTip('Set the number of channels in the EMA recordings')
        #numchannels.triggered.connect()  # need a dialog box
      
        biteplate = QAction('read &Biteplate file',self)
        biteplate.setShortcut('Ctrl+B')
        biteplate.setStatusTip('Get the calibration data from a biteplate recording')
        biteplate.triggered.connect(self.BP_FileDialog)

        palate = QAction('read &Palate trace file',self)
        palate.setShortcut('Ctrl+P')
        palate.setStatusTip('Read in the Palate trace data')
        palate.triggered.connect(self.PL_FileDialog)


        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)

        setup_Menu = menubar.addMenu('&Setup')
        setup_Menu.addAction(biteplate)
        setup_Menu.addAction(palate)
 
       
        run_Menu = menubar.addMenu('&Run')
        run_Menu.addAction(one_Act)
        run_Menu.addAction(all_Act)
        
        #  ----------------- this section lays out the window contents -----------
       
        #grid.setSpacing(10)


        nchan = QLabel('# EMA channels')
        lo.addWidget(nchan)

        nchan_box = QComboBox(self)
        items = ['8','16']
        nchan_box.addItems(items)
        nchan_box.setCurrentIndex(items.index(self.nchannels))
        nchan_box.activated[str].connect(self.change_channels) 
        lo.addWidget(nchan_box)
       
        lo.addStretch(1)
        
        bp_label = QLabel('Biteplate Sensors:')
        lo.addWidget(bp_label)
        
        self.bp_edit = QLineEdit(self)
        bp_string = ' '.join(self.bpsensors)
        self.bp_edit.setText(bp_string)
        lo.addWidget(self.bp_edit)
        self.bp_edit.editingFinished.connect(self.alter_text) 
         
        self.BPbutton = QPushButton(self.bpname,self)
        self.BPbutton.clicked.connect(self.BP_FileDialog)
        lo.addWidget(self.BPbutton)
        
        lo.addStretch(1)

        pl_label = QLabel('Palate Sensors:')
        lo.addWidget(pl_label)
        
        self.pl_edit = QLineEdit(self)
        pl_string = ' '.join(self.PAL_sensors)
        self.pl_edit.setText(pl_string)
        self.pl_edit.editingFinished.connect(self.alter_text) 
        lo.addWidget(self.pl_edit)
 
        time_label = QLabel('midSagittal Times:')
        lo.addWidget(time_label)
        
        times_lo = QHBoxLayout()
        
        
        self.start_edit = QLineEdit(self)
        self.start_edit.setText(str(self.pal_start))
        self.start_edit.setValidator(QDoubleValidator(0.99,99.99,2))
        self.start_edit.editingFinished.connect(self.alter_times) 
        times_lo.addWidget(self.start_edit)
        
        self.end_edit = QLineEdit(self)
        self.end_edit.setText(str(self.pal_end))
        self.end_edit.setValidator(QDoubleValidator(0.99,99.99,2))
        self.end_edit.editingFinished.connect(self.alter_times) 
        times_lo.addWidget(self.end_edit)
        
        lo.addLayout(times_lo)
        
        self.PLbutton = QPushButton(self.palname,self)
        self.PLbutton.clicked.connect(self.PL_FileDialog)
        lo.addWidget(self.PLbutton)
        
        lo.addStretch(1)
 
        ds_label = QLabel('Data Sensors:')
        lo.addWidget(ds_label)
        
        self.ds_edit = QLineEdit(self)
        ds_string = ' '.join(self.sensors)  
        self.ds_edit.setText(ds_string)
        self.ds_edit.editingFinished.connect(self.alter_text) 
        lo.addWidget(self.ds_edit)
        
        SC_label = QLabel('Subcolumns:')
        lo.addWidget(SC_label)
        
        self.SC_edit = QLineEdit(self)
        SC_string = ' '.join(self.subcolumns)  
        self.SC_edit.setText(SC_string)
        self.SC_edit.editingFinished.connect(self.alter_text) 
        lo.addWidget(self.SC_edit)   

        dir_label = QLabel('Base Directory:')
        lo.addWidget(dir_label)
        
        self.base_button = QPushButton(self.base_directory,self)
        self.base_button.clicked.connect(self.base_FileDialog)
        lo.addWidget(self.base_button)
        
        self.runone_button = QPushButton("process one file",self)
        self.runone_button.clicked.connect(self.process_one_file)
        lo.addWidget(self.runone_button)
                                     
        self.setWindowTitle('Process EMA')    
        self.show()

    def alter_text(self):   #heavy-handed - anytime one changes - update all
        text = self.SC_edit.text()
        self.subcolumns = text.split(' ')
        text = self.ds_edit.text()
        self.sensors = text.split(' ')
        text = self.bp_edit.text()
        self.bpsensors = text.split(' ')
        text = self.pl_edit.text()
        self.PAL_sensors = text.split(' ')
        self.statusBar().showMessage('Sensor list updated')
      
    def alter_times(self):   #heavy-handed - anytime one changes - update all
        text = self.start_edit.text()
        self.pal_start = float(text)
        text = self.end_edit.text()
        self.pal_end = float(text)
        try:  # if palate data has already been read, then update this
            self.tracetimes = (self.pdata.time > self.pal_start) & (self.pdata.time < self.pal_end)
        except:
            pass
        
        self.statusBar().showMessage('Mid-sagittal trace from {} to {}'.format(self.pal_start,self.pal_end))
          
    def change_channels(self, text):
        nc = int(text)
        self.nchannels = nc
        if (nc == 8):
            self.sensors = ["REF","UL","LL","JW","TT","TB","TD"]
            self.bpsensors = ["REF","I","J","K","L","OS","MS"]
            self.PAL_sensors = ["REF","I","J","K","L","PL"]
        else:
            self.sensors = ["REF","UL","LL","JW","TT","TB","TD","TL","LC","UI","J","OS","MS","PL"]
            self.bpsensors = self.sensors
            self.PAL_sensors = self.sensors

        self.subcolumns = ["ID","frame","state","q0","qx","qy","qz","x","y","z"]
        self.ds_edit.setText(' '.join(self.sensors))
        self.bp_edit.setText(' '.join(self.bpsensors))
        self.pl_edit.setText(' '.join(self.PAL_sensors))
        self.SC_edit.setText(' '.join(self.subcolumns))
        
    def BP_FileDialog(self):
        fname, wcard = QFileDialog.getOpenFileName(self, 'Open BitePlate file', self.base_directory,
		'TSV Files (*.tsv);;All Files (*)')
        self.base_directory,self.bpname = os.path.split(fname)
        self.base_button.setText(self.base_directory)
        self.BPbutton.setText(self.bpname)
        
        try:
            self.origin, self.m = ema.read_biteplate(
                    self.base_directory,self.bpname,
                    self.bpsensors, self.subcolumns)
            self.statusBar().showMessage('origin: {}'.format(self.origin))
        except ValueError as err:
            self.statusBar().showMessage(err.args[0])
            

    def PL_FileDialog(self):
        fname, wcard = QFileDialog.getOpenFileName(self, 'Open Palate file', self.base_directory,
        	'TSV Files (*.tsv);;All Files (*)')
        self.base_directory,self.palname = os.path.split(fname)
        self.base_button.setText(self.base_directory)
        self.PLbutton.setText(self.palname)
        
        try:
            self.pdata = ema.read_ndi_data(self.base_directory,self.palname,
                                self.PAL_sensors,self.subcolumns)
            self.statusBar().showMessage('Palate trace: success')
        except ValueError as err:
            self.statusBar().showMessage(err.args[0])
            
        try:
            self.pdata = ema.rotate_data(self.pdata,self.m,self.origin,self.PAL_sensors)
        except:
            self.statusBar().showMessage('No rotation applied')

        self.tracetimes = (self.pdata.time > self.pal_start) & (self.pdata.time < self.pal_end)
        ema.save_rotated(self.base_directory,self.palname,self.pdata)

        
    def base_FileDialog(self):
        self.base_directory = QFileDialog.getExistingDirectory(self,"Open a folder",self.base_directory,
        	QFileDialog.ShowDirsOnly)
        self.base_button.setText(self.base_directory)
        
    def process_one_file(self):
        fname, wcard = QFileDialog.getOpenFileName(self, 'Open TSV file', self.base_directory,
        	'TSV Files (*.tsv);;All Files (*)')
        self.base_directory,onename = os.path.split(fname)

        try:
            self.data = ema.read_ndi_data(self.base_directory,onename,self.sensors,self.subcolumns)
        except ValueError as err:
            self.statusBar().showMessage(err.args[0])
            return

        try:
            self.data = ema.rotate_data(self.data,self.m,self.origin,self.sensors)
            self.statusBar().showMessage('Showing rotated data')

        except:
            self.statusBar().showMessage('No rotation applied')

        
        win = Window(self)
        win.move(0,0)
        win.show()
    
    def process_lots_of_files(self):
#def rotate_all(base_directory,sensors, subcolumns, origin, m):
    # loop over tsv files, read them, rotate them, save the old as file.raw
    # save the rotated as file.tsv 

        for root, dirs,files in os.walk(self.base_directory):
            for f in files: 
                base,ext = os.path.splitext(f)
                if ext != '.tsv':
                    continue
                if re.match("\w+BitePlate.*",f):
                    continue
                if re.match("\w+Palate.*",f):
                    continue
                try:
                    data = ema.read_ndi_data(self.base_directory, f, self.sensors,self.subcolumns)
                except ValueError as err:
                    self.statusBar().showMessage(err.args[0])

                try:
                    data=ema.rotate_data(data,self.m,self.origin,self.sensors)
                    ema.save_rotated(self.base_directory,f,data)
                    self.statusBar().showMessage('Processed {}'.format(f))
                except:
                    self.statusBar().showMessage('rotation not applied')
        # file dialog with wild card to specify files to be processed
        #  - or simply walk the base directory and process all nonBP and nonPAL files?

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.plot(parent)

        
    def plot(self, parent):
        
        data = parent.data
        sensors = parent.sensors

        ax1 = self.figure.add_subplot(121)
        ax1.set_title('Side view')
        ax1.set_aspect(1)
        ax1.grid(True)
        ax2 = self.figure.add_subplot(122)
        ax2.set_title('Front view')
        ax2.set_aspect(1)
        ax2.grid(True)
        
        psym = cycle(["r,", "b,", "m,", "c,", "k,","y,"])
        
        for s in sensors:  # read xyz one sensor at a time
            if s == "REF":
                continue
            locx = '{}_x'.format(s)
            locy = '{}_y'.format(s)
            locz = '{}_z'.format(s)
            sym = next(psym)
            
            #   and plot the data
            ax1.plot(data.loc[:,locx],data.loc[:,locy],sym)
            ax2.plot(data.loc[:,locz],data.loc[:,locy],sym)
      
        try:
            pdata = parent.pdata
            tracetimes = parent.tracetimes
            ax1.plot(pdata[tracetimes].PL_x,pdata[tracetimes].PL_y,"g,")
            ax2.plot(pdata[tracetimes].PL_z,pdata[tracetimes].PL_y,"g,")
        except:
            pass
        

        self.canvas.draw()
        
np.seterr(all='raise')

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())
