# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:20:24 2017

@author: joseph.chen
"""
import os
import errno
import pickle
import numpy as np
import scipy as sp

from config.config import abs_data_path

def P_is_winning(tile, opponent_info):
    """Probability of tile being the winning tile.
    param tile: int 0-33, the tile to be checked
    param opponent_info: scipy.sparse.csr.csr_matrix, matrix that stores the info 
          of opponent.
    return prob_winning_tile: float, the probability of tile being a winning tile
    """
    path = abs_data_path+"\\train_model\\trained_models\\waiting_tile_{}.sav".format(tile)
    clf = pickle.load(open(path, "rb"))
    prob_winning_tile = clf.predict_proba(opponent_info)[0][1]  
    return prob_winning_tile 

def P_get_tiles(tile):
    """Probability of getting a specific tile.
    param tile: int 0-33, the tile we want to get
    return prob_get_tiles: float, the probability of getting the tile we want
    """
    return
    
def WAITING(Player):
    if Player.waiting==True:
        return 1
    else:
        return 0
    
def FOLD(Player):
    if Player.fold==True:
        return 1
    else:
        return 0
    
#TODO: finish the probability functions    
def P(is_waiting_player=None, winning_player=None, winning_tile=None, get_tiles=None):
    """Probabilities in different situations
    param is_waiting_player: int 0-3, player index 
    param winning_player: int 0-3, player index 
    param winning_tile: int 0-33, the tile that can complete a player's hand
    param get_tiles: int 0-33, the tile that the player will draw
    """
    if (is_waiting_player!=None):
        return 
    if (winning_player!=None and winning_tile!=None):
        return
    if (get_tiles!=None):
        return
   
#TODO: finish the hand score function (winning from wall)    
def HS_WFW(player=None, tile=None):
    return

#TODO: finish the hand score function (winning from opponent's discards)
def HS(player=None, tile=None):
    return

def EL(player=None, tile=None):
    """Expected loss to `player` if program discards `tile`
    param player: int 0-3, player index of the opponent
    param tile: int 0-33, tile that the program discards
    return: expected loss score
    """
    return LP(player, tile)*HS(player, tile)

def LP(player=None, tile=None):
    """Losing Probability is the probability that `player` is waiting and 
    `tile` is the winning tile for this player.
    param player: int 0-3, the player index
    param tile: int 0-33, the tile that can complete a hand for the player
    return: probability of the losing
    """
    return P(is_waiting_player=player)*P(winning_player=player, winning_tile=tile)

def ODEV(tile):
    """One-Depth Expected Value function
    tile: int 0-33, tile that was chosen to discard by the program
    return fold: bool, true means fold, false means not fold
    """
    p = 1 # probability of reading current states
    value = 0
    PWFW = [0, 0, 0, 0] # probability of winning-from-the-wall
    SWFW = [0, 0, 0, 0] # score of winning-from-the-wall
    SWD = 0 # score of winning-by-a-discard about the program
    # P(get_tiles) # probability of getting the tiles
    
    for i in range(4): # all players (opponent players and program)
        for j in range(34): # all kinds of tiles
            PWFW[i] += P(winning_player=i, winning_tile=j)*P(get_tiles=j)
            SWFW[i] += (HS_WFW(winning_player=i, winning_tile=j) *
                       P(winning_player=i, winning_tile=j) *
                       P(get_tiles=j)
                       )
            
    for j in range(34): # all kinds of tiles
        SWD += (HS(player=0, winning_tile=j) * 
               P(winning_player=0, winning_tile=j) *
               P(get_tiles=j)
               )
        
    for i in range(1,4): # for each opponent
        value -= EL(player=i, tile=tile) 
        
    for i in range(1,4): # for each opponent
        value -= p * WAITING(Player[i]) * PWFW[i] * SWFW[i]
        p *= (1 - WAITING(Player[i])*PWFW[i])
        value += p * (1-FOLD(Player[i])) * PWFW[0] * SWD
        for k in range(4): # for all players
            if k!=i:
                p *= (1 - PWFW[k]*(1-FOLD(Player[i])))
    
    value += p * SWFW[0]
    # PED: probability of the exhastive draw (no player wins after taking all
    # tiles from the wall).
    value -= p * PED * fold_value
    
    fold = False
    if value<=0:
        fold = True
    return fold
        
    
                  
        

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])                

class MajongAI(object):
    
    def __init__(self):        
        features_path=abs_data_path+"/train_model/data/waiting_tiles/test/waiting_tiles_sparse_features.npz"
        target_path=abs_data_path+"/train_model/data/waiting_tiles/test/waiting_tiles_sparse_targets.npz"
            
        # try to load data for testing purpose
        try:
            self.sparse_features = load_sparse_csr(features_path)
            self.sparse_targets = load_sparse_csr(target_path)
        except:
            filename="waiting_tiles_sparse_features.npz OR waiting_tiles_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)         
    
    def evaluate(self, hand, opponent_info, opponent_waiting_tiles):
        """
        hand: tile in 34 format
        """
        prob = []
        for tile in hand:
            target = opponent_waiting_tiles[tile]        
            clf = pickle.load(open(abs_data_path+"/train_model/trained_models/waiting_tile_{}.sav".format(tile), "rb"))
            prob_winning_tile = clf.predict_proba(opponent_info)[0][1]
            #predict = clf.predict(opponent_info)
            #print(tile, prob.shape, clf.classes_)
            #print(prob, predict, target)
            prob.append((prob_winning_tile, tile, target))
        sorted_prob = sorted(prob, key=lambda tup: tup[0])
        #print(sorted_prob)
        num_kinds_of_tiles = len(set([p[1] for p in sorted_prob]))
        #print(num_kinds_of_tiles)
        num_kinds_of_winning_tiles = len(set([p[1] for p in sorted_prob if p[2]==1]))
        #print(num_kinds_of_winning_tiles)
        smallest_rank_among_winning_tiles = [p[2] for p in sorted_prob].index(1) # index the first element with value 1 
        #print(smallest_rank_among_winning_tiles)
        
        Clear = smallest_rank_among_winning_tiles
        AT = num_kinds_of_tiles
        WT = num_kinds_of_winning_tiles
        return Clear*1.0/(AT - WT)

    def gen_random_hand(self, revealed_tiles, num_tiles=14):
        unrevealed_tiles = [4-t for t in revealed_tiles]
        unrevealed_tiles_full_list = []
        for n in range(34):
            while unrevealed_tiles[n]>0:
                unrevealed_tiles_full_list.append(n)
                unrevealed_tiles[n] -= 1
        random.shuffle(unrevealed_tiles_full_list)
        if len(unrevealed_tiles_full_list)>num_tiles: # if not, return None
            return unrevealed_tiles_full_list[0:num_tiles+1]
    
    def accuracy_of_prediction(self):
        
        evaluation_values = []
        for n in range(self.sparse_features.shape[0]): #range(30): #
            opponent_info = self.sparse_features[n]
            opponent_waiting_tiles = self.sparse_targets[n].todense().tolist()[0]
            revealed_tiles = opponent_info[:,-68:-34].todense().tolist()[0]
            # Each time we generate the hand randomly, so every time you run
            # the simulation, you will get a different result. However, as the sample
            # size gets bigger (number of n in this for loop), the average 
            # result should converge.
            hand = self.gen_random_hand(revealed_tiles)
            if (hand is None) or (not any(opponent_waiting_tiles)):
                continue
            
            #print(hand)
            #print([t for t,n in enumerate(opponent_waiting_tiles) if n!=0])
            sample_valid = False
            for t, is_waiting in enumerate(opponent_waiting_tiles):
                if is_waiting and (t in hand):
                    sample_valid = True
                    break
                
            if sample_valid:
                value = self.evaluate(hand, opponent_info, opponent_waiting_tiles)
                evaluation_values.append(value)
                #print("----------------------------\n")
        
        print("Number of effective samples: {}".format(len(evaluation_values)))
        if evaluation_values:
            return sum(evaluation_values)*1.0/(len(evaluation_values))      

    def P_is_winning(self, tile, opponent_info):
        """probability of tile being the winning tile.
        param tile: int 0-33, the tile to be checked
        return prob_winning_tile: float, the probability of tile being a winning tile
        """
        path = abs_data_path+"\\train_model\\trained_models\\waiting_tile_{}.sav".format(tile)
        clf = pickle.load(open(path, "rb"))
        prob_winning_tile = clf.predict_proba(opponent_info)[0][1]  
        return prob_winning_tile   

    def test_P_is_winning(self):
        for n in range(3): # range(self.sparse_features.shape[0]): #
            opponent_info = self.sparse_features[n]
            print(type(opponent_info))
            prob_is_winning = self.P_is_winning(0, opponent_info)
            print(prob_is_winning)
                
            
if __name__=="__main__":
    print("OMG")
    AI = MajongAI()
    AI.test_P_is_winning()
 