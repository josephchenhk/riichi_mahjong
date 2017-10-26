# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:41:50 2017

@author: joseph.chen
"""
from config.config import abs_data_path

if __name__=="__main__":
    
    # Total 28,904,266 lines. So we take 700,000*40 = 28,000,000 samples
#    for n in range(40):
#        n1 = 700000*n       # 0
#        n2 = 700000*(n+1)   # 600000
        
    # Total 106259 lines. So we take 10000*10 = 100,000 samples
    for n in range(10):
        n1 = 10000*n       # 0
        n2 = 10000*(n+1)   # 600000
        
        # Slice the data and save the pieces to hard disk
        k = 0
        with open(abs_data_path+"/debuglogs/testscores{}_{}_lines[{}_{}].debuglog".format(n+1,n2-n1,n1,n2), "w") as f2:        
            #with open(abs_data_path+"/debuglogs/tmp/full_waiting.debuglog") as f1:
            with open(abs_data_path+"/debuglogs/tmp/full_scores.debuglog") as f1:    
                while k<n2:
                    data = next(f1)
                    
                    if k==n2-1:
                        data = data.replace("\n", "")
                    if k>=n1:
                        #print("({}).{}".format(k,data))
                        f2.write(data)
                    k += 1
        print(k)
 
    
            