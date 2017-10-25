# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:48:03 2017

@author: joseph.chen
"""

class Features(object):
    is_waiting = [0,1]
    is_riichi = [0,1]
    num_of_revealed_melds = [0,1,2,3,4]
    num_of_discarded_tiles = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] 
    
    # this category should be same as discarded tiles?
    # Possible interpretation: not each time the dealer discards is counted as a `turn`.
    # Dealer must wait until all others has discarded at least one tile, then the `turn`
    # number adds one. If for some reason a player interrupts the order, the `turn`
    # does not change. 
    num_of_turns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] 
    
    # Possible interpretation: not all discarded tiles are changed tiles, only those
    # that `changed` a player's hand is called changed tiles.
    num_of_changed_tiles = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    
    tiles_34 = range(34)
    tiles_34_plus_null = range(-1,34)
    tiles_37 = range(37) 
    tiles_136 = range(136)
    kinds_of_revealed_melds = range(136)
    is_red_dora_discarded = [0,1]
    is_dora_discarded = range(34)
    
    revealed_melds = ['',
                      '123m','234m','345m','456m','567m','678m','789m',
                      '123p','234p','345p','456p','567p','678p','789p',
                      '123s','234s','345s','456s','567s','678s','789s',
                      '111m','222m','333m','444m','555m','666m','777m','888m','999m',
                      '111p','222p','333p','444p','555p','666p','777p','888p','999p',
                      '111s','222s','333s','444s','555s','666s','777s','888s','999s',
                      '111z','222z','333z','444z','555z','666z','777z',
                      '1111m','2222m','3333m','4444m','5555m','6666m','7777m','8888m','9999m',
                      '1111p','2222p','3333p','4444p','5555p','6666p','7777p','8888p','9999p',
                      '1111s','2222s','3333s','4444s','5555s','6666s','7777s','8888s','9999s',
                      '1111z','2222z','3333z','4444z','5555z','6666z','7777z']
    
    melds_kinds = ['chi','pon','kan']
    revealed_melds_kinds = ['']
    revealed_melds_kinds += melds_kinds
    for m in melds_kinds:
        for n in melds_kinds:
            revealed_melds_kinds  += [m+"-"+n]
    for m in melds_kinds:
        for n in melds_kinds:
            for i in melds_kinds:
                revealed_melds_kinds  += [m+"-"+n+"-"+i]
    for m in melds_kinds:
        for n in melds_kinds:
            for i in melds_kinds:
                for j in melds_kinds:
                    revealed_melds_kinds  += [m+"-"+n+"-"+i+"-"+j]
    
    @staticmethod
    def gen_one_hot_features(feature_value:list, feature:list)->list:
        assert set(feature_value)<set(feature), "values in feature_value {} must be subset of feature {}".format(feature_value, feature)
        one_hot_feature = [0]*len(feature)
        for v in feature_value:
            one_hot_feature[feature.index(v)] = 1
        return one_hot_feature
    
    @staticmethod
    def join_two_features(feature_value1:list, feature1:list, 
                          feature_value2:list, feature2:list)->list:
        assert set(feature_value1)<set(feature1), "values in feature_value1 {} must be subset of feature1 {}".format(feature_value1, feature1)
        assert set(feature_value2)<set(feature2), "values in feature_value2 {} must be subset of feature2 {}".format(feature_value2, feature2)
        joint_feature = [0]*len(feature1)*len(feature2)
        N1 = len(feature1)
        for f2 in feature_value2:
            for f1 in feature_value1:
                n2 = feature2.index(f2)
                n1 = feature1.index(f1)
                joint_feature[n2*N1+n1] = 1
        return joint_feature


if __name__=="__main__":
    features = Features()
    f = features.join_two_features([1], 
                                   features.is_riichi, 
                                   [3], 
                                   features.num_of_revealed_melds)
    print(f)
