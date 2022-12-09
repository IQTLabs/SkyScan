#!/usr/bin/env python
# coding: utf-8
import time
import os
import shutil
import socket
hostname = socket.gethostname()

directory = '/flash/raw'
dumpdirectory = '/flash/unprocessed'

path = ''
fname = ''


def processFile(path):
#    try:
    if True:
        fileList = os.listdir(path)
        for fname in fileList:
            if os.path.isdir(os.path.join(path, fname)):
                print('PATH IS A DIRECTORY: '+os.path.join(path, fname))
                processFile(os.path.join(path, fname))
            else:
                newfname = fname.split('.')[0] + '_' + hostname + '.' + fname.split('.')[1]
                shutil.move(os.path.join(path, fname), os.path.join(dumpdirectory, newfname))
            time.sleep(0.01)
        os.rmdir(path)
#    except:
#        print('Ran into error | path: '+path+' | file: '+fname + ' |')
#        time.sleep(1)


while True:
    dir_list = [x[0] for x in os.walk(directory)]
    dir_list[:] = (value for value in dir_list if value != directory)
    for path in dir_list:
        print('Processing Path: '+path)
        processFile(path)        
    time.sleep(5)


