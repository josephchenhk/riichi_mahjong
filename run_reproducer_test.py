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
    
    CREATE_NO_WINDOW = 0x08000000
    
    for n,fn in enumerate(mjlog_names[:]): # (mjlog_names[0:100]): #
        if "mjlog" not in fn:
            continue
        
        #cmd = "python reproducer_test.py -m ..\..\data\{} -d".format(fn)
        mjlog = os.path.join(abs_data_path, "raw_data", fn)
        
        # For HS
        #cmd = "python reproducer_test_for_scores.py -m {} -d".format(mjlog) 
        
        # For HS_WFW
        cmd = "python reproducer_test_for_scores_wfw.py -m {} -d".format(mjlog) 

        #os.system(cmd)
        p = subprocess.Popen(cmd, shell=False, 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL,
                             creationflags=CREATE_NO_WINDOW)
        #p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        p.terminate()
        
        if n%7000==0:
            print("{} mjlogs completed.".format(n+1))
        
        #subprocess.call(['python', 'reproducer_test.py', '-m', '..\..\data\{} -d'.format(fn), '-d' ])
    toc = time.time()
    print("End time:{}\nElapsed time:{}".format(toc, toc-tic))
#   
#    for n,fn in enumerate(file_names[:]):
#        #print(subprocess.check_output(['tlu','download',fn, fn+".mjlog"]))
#        print(n)
#        subprocess.call(['tlu','download',fn, fn+".mjlog"])