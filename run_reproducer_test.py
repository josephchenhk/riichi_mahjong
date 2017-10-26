# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:58:52 2017

@author: joseph.chen
"""

import subprocess
import os
import time

from config.config import abs_data_path

if __name__=="__main__":
        
    # read file names 
    mjlog_names = os.listdir(abs_data_path+"/raw_data")
    #print(mjlog_names)
    tic = time.time()
    print("Start time:{}".format(tic))
    
    for n,fn in enumerate(mjlog_names[:]): # (mjlog_names[0:30000]):
        if "mjlog" not in fn:
            continue
        
        #cmd = "python reproducer_test.py -m ..\..\data\{} -d".format(fn)
        mjlog = os.path.join(abs_data_path, "raw_data", fn)
        cmd = "python reproducer_test_for_scores.py -m {} -d".format(mjlog) 
        
        os.system(cmd)
        #subprocess.call(['python', 'reproducer_test.py', '-m', '..\..\data\{} -d'.format(fn), '-d' ])
    toc = time.time()
    print("End time:{}\nElapsed time:{}".format(toc, toc-tic))
#   
#    for n,fn in enumerate(file_names[:]):
#        #print(subprocess.check_output(['tlu','download',fn, fn+".mjlog"]))
#        print(n)
#        subprocess.call(['tlu','download',fn, fn+".mjlog"])