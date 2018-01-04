# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:38:18 2017

@author: joseph.chen
"""
import time
import logging
from mahjong.ai.Monte_Carlo import MonteCarlo
from tenhou.decoder import TenhouDecoder


from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld


#import mahjong as mahjong_std
#from mahjong_std.tile import TilesConverter

logger = logging.getLogger('MClogger')

logger.setLevel(logging.INFO)

if __name__=="__main__":
    MC = MonteCarlo()
    
    tic = time.time()
    for n in range(50):
        print("Sim %d"%(n+1))
        MC._check2()
    toc = time.time()
    
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    
    print("Elapsed time: {:.3f} seconds".format(toc-tic))


    

