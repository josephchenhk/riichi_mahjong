# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:20:24 2017

@author: joseph.chen
"""
import pickle

from config.config import abs_data_path

def P_is_winning(tile, opponent_info):
    """probability of tile being the winning tile.
    param tile: int 0-33, the tile to be checked
    return prob_winning_tile: float, the probability of tile being a winning tile
    """
    path = abs_data_path+"\\train_model\\trained_models\\waiting_tile_{}.sav".format(tile)
    clf = pickle.load(open(path, "rb"))
    prob_winning_tile = clf.predict_proba(opponent_info)[0][1]  
    return prob_winning_tile  
    

def ODEV():
    """One-Depth Expected Value function
    """
    p = 1 # probability of reading current states
    value = 0
    PWFW = [0, 0, 0, 0] # probability of winning-from-the-wall
    SWFW = 0 # score of winning-from-the-wall
    SWD = 0 # score of winning-by-a-discard about the program
    # P(get_tiles) # probability of getting the tiles
    for i in range(4): # all players (opponent players and program)
        for j in range(34): # all kinds of tiles
            PWFW[i] += P 
                
                
                
if __name__=="__main__":
    print("OMG")
    P_is_winning(1)