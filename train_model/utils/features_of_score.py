# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:11:04 2017

@author: joseph.chen
"""

class FeaturesOfScore(object):
    """Generate some features of score prediction.
    """
    def __init__(self, discarded_tiles_34_array,
                 revealed_melded_tiles_34_array,
                 table_revealed_tiles_34_array):
        self.discarded_tiles_34_array = discarded_tiles_34_array
        self.revealed_melded_tiles_34_array = revealed_melded_tiles_34_array
        self.table_revealed_tiles_34_array = table_revealed_tiles_34_array
        
    def is_discards_all_simples(self):
        is_all_simples = (not any(self.discarded_tiles_34_array[27:]))
        return 1 if is_all_simples else 0
    
    def num_revealed_melds_of_wind_tiles(self):
        east,south,west,north = self.revealed_melded_tiles_34_array[27:31]
        count = 0
        if east>0: count += 1
        if south>0: count += 1
        if west>0: count += 1
        if north>0: count += 1
        return count
    
    def num_revealed_melds_of_dragon_tiles(self):
        white,blue,red = self.revealed_melded_tiles_34_array[31:34]
        count = 0
        if white>0: count += 1
        if blue>0: count += 1
        if red>0: count += 1
        return count
    
    def can_hands_be_all_pons(self):
        for t in self.revealed_melded_tiles_34_array:
            if (t!=0 and t!=3):
                return 0
        return 1
    
    def can_hands_be_flush(self):
        man = self.revealed_melded_tiles_34_array[0:9]
        pin = self.revealed_melded_tiles_34_array[9:18]
        suo = self.revealed_melded_tiles_34_array[18:27]
        sum_man = sum(man)
        sum_pin = sum(pin)
        sum_suo = sum(suo)
        if (sum_man+sum_pin+sum_suo==max(sum_man, sum_pin, sum_suo)):
            return 1
        else:
            return 0
        
    def can_hands_be_all_simples(self):
        can_be_all_simples = (not any(self.revealed_melded_tiles_34_array[27:]))
        return 1 if can_be_all_simples else 0 
    
if __name__=="__main__":
    discarded_tiles_34_array = [0]*34
    discarded_tiles_34_array[-2] = 2
    revealed_melded_tiles_34_array = [0]*34
    revealed_melded_tiles_34_array[1] = 3
    revealed_melded_tiles_34_array[10] = 3                              
    table_revealed_tiles_34_array = [0]*34
    features_of_score = FeaturesOfScore(discarded_tiles_34_array,
                                        revealed_melded_tiles_34_array,
                                        table_revealed_tiles_34_array)
    print(features_of_score.is_discards_all_simples())
    print(features_of_score.num_revealed_melds_of_wind_tiles())
    print(features_of_score.can_hands_be_all_pons())
    print(features_of_score.can_hands_be_flush())