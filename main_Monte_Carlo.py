# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:38:18 2017

@author: joseph.chen
"""
import time
import logging
from mahjong.ai.Monte_Carlo import MonteCarlo

from tenhou.decoder import TenhouDecoder


logger = logging.getLogger('MClogger')

logger.setLevel(logging.INFO)

if __name__=="__main__":
    MC = MonteCarlo()
    
    tic = time.time()
    for n in range(100):
        print("Sim %d"%(n+1))
        MC._check2()
    toc = time.time()
    
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    
    print("Elapsed time: {:.3f} seconds".format(toc-tic))


#    decoder = TenhouDecoder()
#    tag = '<N who="2" m="25600" /> <DORA hai="32" /> <V/>'
#    meld = decoder.parse_meld(tag)
#    
#    """
#    who = None
#    tiles = []
#    type = None
#    from_who = None
#    called_tile = None
#    # we need it to distinguish opened and closed kan
#    opened = True
#    """
#    print(meld)
#    
#    print("---------")
#    print(meld.who, meld.from_who, meld.opened)
#    print(meld.tiles)
#    print(meld.type)
#    print(meld.called_tile)