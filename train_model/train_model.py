# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:40:26 2017

@author: joseph.chen
"""
import os
import errno
import sys
import numpy as np
import scipy as sp
import pandas as pd
import pickle
import time
import copy
import ast
from functools import reduce
import shutil
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from train_model.utils.features import Features 
from train_model.utils.tile import TilesConverter 
from train_model.utils.features_of_score import FeaturesOfScore
from config.config import abs_data_path


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_and_process_is_waiting_data(debuglog_name="", num_lines=None, chunk_size=1000):
    
    tic = time.time()
    target_list = []
    feature_list = []
    with open(abs_data_path+"/debuglogs/{}.debuglog".format(debuglog_name)) as f:
        m = 1
        
        existing_files = sorted([int(name.split("_")[-1].split(".")[0]) for name in os.listdir(abs_data_path+"/train_model/data/is_waiting/chunk")])
        if existing_files:
            k = existing_files[-1] + 1
        else:
            k = 1
            
        for n, line in enumerate(f):
            if n<num_lines:
                d1, d2 = line.split(";")
                #d1 = d1.replace(",,",",-1,").replace("True","1").replace("False","0")
                #d2 = d2.replace(",,",",-1,").replace("True","1").replace("False","0")

                (table_count_of_honba_sticks,
                 table_count_of_remaining_tiles,
                 table_count_of_riichi_sticks,
                 table_round_number,
                 table_round_wind,
                 table_turns,
                 table_dealer_seat,
                 table_dora_indicators,
                 table_dora_tiles,
                 table_revealed_tiles) = ast.literal_eval(d1)
                
                (player_winning_tiles,                   
                 player_discarded_tiles, 
                 player_dealer_seat,
                 player_in_riichi,
                 player_is_dealer,
                 player_is_open_hand,              
                 player_melds,                
                 player_name,
                 player_position,
                 player_rank,
                 player_scores,
                 player_seat,
                 player_uma) = ast.literal_eval(d2)
                
                features = gen_is_waiting_features(table_count_of_honba_sticks,
                                                    table_count_of_remaining_tiles,
                                                    table_count_of_riichi_sticks,
                                                    table_round_number,
                                                    table_round_wind,
                                                    table_turns,
                                                    table_dealer_seat,
                                                    table_dora_indicators,
                                                    table_dora_tiles,
                                                    table_revealed_tiles,
                                                    player_winning_tiles,                   
                                                    player_discarded_tiles, 
                                                    player_dealer_seat,
                                                    player_in_riichi,
                                                    player_is_dealer,
                                                    player_is_open_hand,              
                                                    player_melds,                
                                                    player_name,
                                                    player_position,
                                                    player_rank,
                                                    player_scores,
                                                    player_seat,
                                                    player_uma)
                f1, f2, f3, f4, f5, f6, f7, f8, f9 = features
                #print(n, [len([f1]), len([f2]), len([f3]), len([f4]), len([f5]), len(f6), len(f7), len(f8), len(f9)])
                target = gen_is_waiting_targets(player_winning_tiles)
                target_list.append(target)
                feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+f8+f9)
                
                if m==chunk_size:
                    print(k)
                    m = 0
                    sparse_features = sp.sparse.csr_matrix(feature_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/is_waiting/chunk/is_waiting_sparse_feature_{}".format(k), sparse_features) 
                    feature_list = []
                    k += 1

              
            else:
                break
            
            m += 1
    
    sparse_targets = sp.sparse.csr_matrix(target_list)
    #sparse_targets = sparse_targets.T
    print("sparse_target size: {}".format(sparse_targets.shape))
    save_sparse_csr(abs_data_path+"/train_model/data/is_waiting/full/is_waiting_sparse_target", sparse_targets) 
    
    if feature_list:
        sparse_features = sp.sparse.csr_matrix(feature_list)
        save_sparse_csr(abs_data_path+"/train_model/data/is_waiting/chunk/is_waiting_sparse_feature_{}".format(k), sparse_features) 
    
    toc = time.time()
    print("Data has been loaded successfully (Time: {:.2f} seconds). Wait for parsing...".format(toc-tic))


def load_and_process_waiting_tiles_data(debuglog_name="", num_lines=None, chunk_size=1000):
    
    tic = time.time()
    target_list = []
    feature_list = []
    with open(abs_data_path+"/debuglogs/{}.debuglog".format(debuglog_name)) as f:
        m = 1
        
        existing_files = sorted([int(name.split("_")[-1].split(".")[0]) for name in os.listdir(abs_data_path+"/train_model/data/waiting_tiles/chunk")])
        if existing_files:
            k = existing_files[-1] + 1
        else:
            k = 1
            
        for n, line in enumerate(f):
            if n<num_lines:
                d1, d2 = line.split(";")
                #d1 = d1.replace(",,",",-1,").replace("True","1").replace("False","0")
                #d2 = d2.replace(",,",",-1,").replace("True","1").replace("False","0")

                (table_count_of_honba_sticks,
                 table_count_of_remaining_tiles,
                 table_count_of_riichi_sticks,
                 table_round_number,
                 table_round_wind,
                 table_turns,
                 table_dealer_seat,
                 table_dora_indicators,
                 table_dora_tiles,
                 table_revealed_tiles) = ast.literal_eval(d1)
                
                (player_winning_tiles,                   
                 player_discarded_tiles, 
                 player_dealer_seat,
                 player_in_riichi,
                 player_is_dealer,
                 player_is_open_hand,              
                 player_melds,                
                 player_name,
                 player_position,
                 player_rank,
                 player_scores,
                 player_seat,
                 player_uma) = ast.literal_eval(d2)
    
                
                features = gen_waiting_tiles_features(table_count_of_honba_sticks,
                                                        table_count_of_remaining_tiles,
                                                        table_count_of_riichi_sticks,
                                                        table_round_number,
                                                        table_round_wind,
                                                        table_turns,
                                                        table_dealer_seat,
                                                        table_dora_indicators,
                                                        table_dora_tiles,
                                                        table_revealed_tiles,
                                                        player_winning_tiles,                   
                                                        player_discarded_tiles, 
                                                        player_dealer_seat,
                                                        player_in_riichi,
                                                        player_is_dealer,
                                                        player_is_open_hand,              
                                                        player_melds,                
                                                        player_name,
                                                        player_position,
                                                        player_rank,
                                                        player_scores,
                                                        player_seat,
                                                        player_uma)
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = features
                #print(n, [len([f1]), len([f2]), len([f3]), len([f4]), len([f5]), len(f6), len(f7), len(f8), len(f9), len(f10), len(f11)])
                target = gen_waiting_tiles_targets(player_winning_tiles)
                target_list.append(target)
                feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+f8+f9+f10+f11)
                
                if m==chunk_size:
                    print(k)
                    m = 0
                    sparse_features = sp.sparse.csr_matrix(feature_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/chunk/waiting_tiles_sparse_feature_{}".format(k), sparse_features) 
                    feature_list = []
                    
                    sparse_targets = sp.sparse.csr_matrix(target_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/chunk/waiting_tiles_sparse_target_{}".format(k), sparse_targets) 
                    target_list = []
                    
                    k += 1

              
            else:
                break
            
            m += 1
    
    if target_list:
        sparse_targets = sp.sparse.csr_matrix(target_list)
        print("sparse_target size: {}".format(sparse_targets.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/chunk/waiting_tiles_sparse_target_{}".format(k), sparse_targets) 
        
    if feature_list:
        sparse_features = sp.sparse.csr_matrix(feature_list)
        print("sparse_features size: {}".format(sparse_features.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/chunk/waiting_tiles_sparse_feature_{}".format(k), sparse_features) 
    
    toc = time.time()
    print("Data has been loaded successfully (Time: {:.2f} seconds). Wait for parsing...".format(toc-tic))
 
def load_and_process_scores_data(debuglog_name="", num_lines=None, chunk_size=1000):
    
    tic = time.time()
    target_list = []
    feature_list = []
    with open(abs_data_path+"/debuglogs/{}.debuglog".format(debuglog_name)) as f:
        m = 1
        
        existing_files = sorted([int(name.split("_")[-1].split(".")[0]) for name in os.listdir(abs_data_path+"/train_model/data/scores/chunk")])
        if existing_files:
            k = existing_files[-1] + 1
        else:
            k = 1
            
        for n, line in enumerate(f):
            if n<num_lines:
                # ADD: add discarded tile (d3)
                d1, d2, d3, d4 = line.replace("\n","").split(";")
                #d1 = d1.replace(",,",",-1,").replace("True","1").replace("False","0")
                #d2 = d2.replace(",,",",-1,").replace("True","1").replace("False","0")

                (table_count_of_honba_sticks,
                 table_count_of_remaining_tiles,
                 table_count_of_riichi_sticks,
                 table_round_number,
                 table_round_wind,
                 table_turns,
                 table_dealer_seat,
                 table_dora_indicators,
                 table_dora_tiles,
                 table_revealed_tiles) = ast.literal_eval(d1)
                
                (player_winning_tiles,                   
                 player_discarded_tiles, 
                 player_dealer_seat,
                 player_in_riichi,
                 player_is_dealer,
                 player_is_open_hand,              
                 player_melds,                
                 player_name,
                 player_position,
                 player_rank,
                 player_scores,
                 player_seat,
                 player_uma) = ast.literal_eval(d2)
                
                discarded_tile = int(d3)
                   
                features = gen_scores_features(table_count_of_honba_sticks,
                                                        table_count_of_remaining_tiles,
                                                        table_count_of_riichi_sticks,
                                                        table_round_number,
                                                        table_round_wind,
                                                        table_turns,
                                                        table_dealer_seat,
                                                        table_dora_indicators,
                                                        table_dora_tiles,
                                                        table_revealed_tiles,
                                                        player_winning_tiles,                   
                                                        player_discarded_tiles, 
                                                        player_dealer_seat,
                                                        player_in_riichi,
                                                        player_is_dealer,
                                                        player_is_open_hand,              
                                                        player_melds,                
                                                        player_name,
                                                        player_position,
                                                        player_rank,
                                                        player_scores,
                                                        player_seat,
                                                        player_uma)
                #f3, f4, f5, f6, f7, f8, f9, f10, f11 = features
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13 = features 
                f14 = discarded_tile
                
                #print(n, [len([f1]), len([f2]), len([f3]), len([f4]), len([f5]), len(f6), len(f7), len(f8), len(f9), len(f10), len(f11)])
                target = gen_scores_targets(d4) # scores lost to the winner
                target_list.append(target)
                
                #feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+f8+f9+f10+f11)
                feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+[f8]+[f9]+[f10]+[f11]+[f12]+[f13]+[f14])
                
                if m==chunk_size:
                    print(k)
                    m = 0
                    sparse_features = sp.sparse.csr_matrix(feature_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/scores/chunk/scores_sparse_feature_{}".format(k), sparse_features) 
                    feature_list = []
                    
                    sparse_targets = sp.sparse.csr_matrix(target_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/scores/chunk/scores_sparse_target_{}".format(k), sparse_targets) 
                    target_list = []
                    
                    k += 1

              
            else:
                break
            
            m += 1
    
    if target_list:
        sparse_targets = sp.sparse.csr_matrix(target_list)
        print("sparse_target size: {}".format(sparse_targets.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/scores/chunk/scores_sparse_target_{}".format(k), sparse_targets) 
        
    if feature_list:
        sparse_features = sp.sparse.csr_matrix(feature_list)
        print("sparse_features size: {}".format(sparse_features.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/scores/chunk/scores_sparse_feature_{}".format(k), sparse_features) 
    
    toc = time.time()
    print("Data `Scores` has been loaded successfully (Time: {:.2f} seconds). Wait for parsing...".format(toc-tic))
 
def load_and_process_wfw_scores_data(debuglog_name="", num_lines=None, chunk_size=1000):
    
    tic = time.time()
    target_list = []
    feature_list = []
    with open(abs_data_path+"/debuglogs/{}.debuglog".format(debuglog_name)) as f:
        m = 1
        
        existing_files = sorted([int(name.split("_")[-1].split(".")[0]) for name in os.listdir(abs_data_path+"/train_model/data/wfw_scores/chunk")])
        if existing_files:
            k = existing_files[-1] + 1
        else:
            k = 1
            
        for n, line in enumerate(f):
            if n<num_lines:
                d1, d2, d3 = line.replace("\n","").split(";")
                #d1 = d1.replace(",,",",-1,").replace("True","1").replace("False","0")
                #d2 = d2.replace(",,",",-1,").replace("True","1").replace("False","0")

                (table_count_of_honba_sticks,
                 table_count_of_remaining_tiles,
                 table_count_of_riichi_sticks,
                 table_round_number,
                 table_round_wind,
                 table_turns,
                 table_dealer_seat,
                 table_dora_indicators,
                 table_dora_tiles,
                 table_revealed_tiles) = ast.literal_eval(d1)
                
                (player_winning_tiles,                   
                 player_discarded_tiles, 
                 player_dealer_seat,
                 player_in_riichi,
                 player_is_dealer,
                 player_is_open_hand,              
                 player_melds,                
                 player_name,
                 player_position,
                 player_rank,
                 player_scores,
                 player_seat,
                 player_uma) = ast.literal_eval(d2)
    
                
                features = gen_wfw_scores_features(table_count_of_honba_sticks,
                                                        table_count_of_remaining_tiles,
                                                        table_count_of_riichi_sticks,
                                                        table_round_number,
                                                        table_round_wind,
                                                        table_turns,
                                                        table_dealer_seat,
                                                        table_dora_indicators,
                                                        table_dora_tiles,
                                                        table_revealed_tiles,
                                                        player_winning_tiles,                   
                                                        player_discarded_tiles, 
                                                        player_dealer_seat,
                                                        player_in_riichi,
                                                        player_is_dealer,
                                                        player_is_open_hand,              
                                                        player_melds,                
                                                        player_name,
                                                        player_position,
                                                        player_rank,
                                                        player_scores,
                                                        player_seat,
                                                        player_uma)
                #f3, f4, f5, f6, f7, f8, f9, f10, f11 = features
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13 = features 
                #print(n, [len([f1]), len([f2]), len([f3]), len([f4]), len([f5]), len(f6), len(f7), len(f8), len(f9), len(f10), len(f11)])
                target = gen_wfw_scores_targets(d3) # scores lost to the winner
                target_list.append(target)
                
                #feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+f8+f9+f10+f11)
                feature_list.append([f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+[f8]+[f9]+[f10]+[f11]+[f12]+[f13])
                
                if m==chunk_size:
                    print(k)
                    m = 0
                    sparse_features = sp.sparse.csr_matrix(feature_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/chunk/scores_sparse_feature_{}".format(k), sparse_features) 
                    feature_list = []
                    
                    sparse_targets = sp.sparse.csr_matrix(target_list)
                    save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/chunk/scores_sparse_target_{}".format(k), sparse_targets) 
                    target_list = []
                    
                    k += 1             
            else:
                break
            
            m += 1
    
    if target_list:
        sparse_targets = sp.sparse.csr_matrix(target_list)
        print("sparse_target size: {}".format(sparse_targets.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/chunk/scores_sparse_target_{}".format(k), sparse_targets) 
        
    if feature_list:
        sparse_features = sp.sparse.csr_matrix(feature_list)
        print("sparse_features size: {}".format(sparse_features.shape))
        save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/chunk/scores_sparse_feature_{}".format(k), sparse_features) 
    
    toc = time.time()
    print("Data `Scores` has been loaded successfully (Time: {:.2f} seconds). Wait for parsing...".format(toc-tic))
 
    
def load_and_process_is_waiting_sparse_data(dir_path="data/"):
    
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        if 'sparse_feature_' in name:
            sparse_feature = load_sparse_csr(dir_path+name)
            #print(sparse_feature.shape)
            if 'features' in locals():
                features = sp.sparse.vstack([features, sparse_feature])
            else:
                features = sp.sparse.csr_matrix(sparse_feature)
                
    print("sparse_feature size: {}".format(features.shape))
    return features
 
def load_and_process_waiting_tiles_sparse_data(dir_path="data/"):
    
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        if 'sparse_feature_' in name:
            sparse_feature = load_sparse_csr(dir_path+name)
            #print(sparse_feature.shape)
            if 'features' in locals():
                features = sp.sparse.vstack([features, sparse_feature])
            else:
                features = sp.sparse.csr_matrix(sparse_feature)
                
        if 'sparse_target_' in name:
            sparse_target = load_sparse_csr(dir_path+name)
            #print(sparse_feature.shape)
            if 'targets' in locals():
                targets = sp.sparse.vstack([targets, sparse_target])
            else:
                targets = sp.sparse.csr_matrix(sparse_target)
    print("sparse_target size: {}".format(targets.shape))
    print("sparse_feature size: {}".format(features.shape))
    return targets, features    

def load_and_process_scores_sparse_data(dir_path="data/"):
    
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        if 'sparse_feature_' in name:
            sparse_feature = load_sparse_csr(dir_path+name)
            #print(sparse_feature.shape)
            if 'features' in locals():
                features = sp.sparse.vstack([features, sparse_feature])
            else:
                features = sp.sparse.csr_matrix(sparse_feature)
                
        if 'sparse_target_' in name:
            sparse_target = load_sparse_csr(dir_path+name)
            #print(sparse_feature.shape)
            if 'targets' in locals():
                targets = sp.sparse.vstack([targets, sparse_target])
            else:
                targets = sp.sparse.csr_matrix(sparse_target)
    print("sparse_target size: {}".format(targets.shape))
    print("sparse_feature size: {}".format(features.shape))
    return targets, features


def load_and_process_wfw_scores_sparse_data(dir_path="data/"):  
    # This function should be the same as load_and_process_scores_sparse_data(dir_path="data/")
    return load_and_process_scores_sparse_data(dir_path)
    
def gen_is_waiting_features(table_count_of_honba_sticks,
                            table_count_of_remaining_tiles,
                            table_count_of_riichi_sticks,
                            table_round_number,
                            table_round_wind,
                            table_turns,
                            table_dealer_seat,
                            table_dora_indicators,
                            table_dora_tiles,
                            table_revealed_tiles,
                            player_winning_tiles,                   
                            player_discarded_tiles, 
                            player_dealer_seat,
                            player_in_riichi,
                            player_is_dealer,
                            player_is_open_hand,              
                            player_melds,                
                            player_name,
                            player_position,
                            player_rank,
                            player_scores,
                            player_seat,
                            player_uma):
    
    #ft = Features()
    
    num_revealed_melds = len(player_melds)
    discarded_tiles = [d[0] for d in player_discarded_tiles]
    discarded_tiles_34_array = TilesConverter.to_34_array(discarded_tiles)
    num_discarded_tiles = len(discarded_tiles)
    changed_tiles = [d[0] for d in player_discarded_tiles if d[1]==0]
    num_changed_tiles = len(changed_tiles)
    revealed_melded_tiles = reduce(lambda x,y:x+y, [m[0] for m in player_melds]) if player_melds else []   
    revealed_melded_tiles_34_array = TilesConverter.to_34_array(revealed_melded_tiles)
    revealed_melds_discarded_tiles = [m[2][0] for m in player_melds] 
    revealed_melds_discarded_tiles_34_array = TilesConverter.to_34_array(revealed_melds_discarded_tiles)
    discarded_bonus_tiles = [d for d in discarded_tiles if d in table_dora_tiles]
    discarded_bonus_tiles_34_array = TilesConverter.to_34_array(discarded_bonus_tiles)
                                             
    f1 = player_in_riichi
    f2 = num_revealed_melds
    f3 = num_discarded_tiles
    f4 = table_turns
    f5 = num_changed_tiles
    f6 = revealed_melded_tiles_34_array
    f7 = revealed_melds_discarded_tiles_34_array
    f8 = discarded_bonus_tiles_34_array
    f9 = discarded_tiles_34_array

    return f1, f2, f3, f4, f5, f6, f7, f8, f9


def gen_waiting_tiles_features(table_count_of_honba_sticks,
                            table_count_of_remaining_tiles,
                            table_count_of_riichi_sticks,
                            table_round_number,
                            table_round_wind,
                            table_turns,
                            table_dealer_seat,
                            table_dora_indicators,
                            table_dora_tiles,
                            table_revealed_tiles,
                            player_winning_tiles,                   
                            player_discarded_tiles, 
                            player_dealer_seat,
                            player_in_riichi,
                            player_is_dealer,
                            player_is_open_hand,              
                            player_melds,                
                            player_name,
                            player_position,
                            player_rank,
                            player_scores,
                            player_seat,
                            player_uma):

    #ft = Features()
    
    num_revealed_melds = len(player_melds)
    discarded_tiles = [d[0] for d in player_discarded_tiles]
    discarded_tiles_34_array = TilesConverter.to_34_array(discarded_tiles)
    num_discarded_tiles = len(discarded_tiles)
    changed_tiles = [d[0] for d in player_discarded_tiles if d[1]==0]
    changed_tiles_34_array = TilesConverter.to_34_array(changed_tiles)
    num_changed_tiles = len(changed_tiles)
    revealed_melded_tiles = reduce(lambda x,y:x+y, [m[0] for m in player_melds]) if player_melds else []   
    revealed_melded_tiles_34_array = TilesConverter.to_34_array(revealed_melded_tiles)
    revealed_melds_discarded_tiles = [m[2][0] for m in player_melds] 
    revealed_melds_discarded_tiles_34_array = TilesConverter.to_34_array(revealed_melds_discarded_tiles)
    discarded_bonus_tiles = [d for d in discarded_tiles if d in table_dora_tiles]
    discarded_bonus_tiles_34_array = TilesConverter.to_34_array(discarded_bonus_tiles)
                                             
    
    f1 = player_in_riichi
    f2 = num_revealed_melds
    f3 = num_discarded_tiles
    f4 = table_turns
    f5 = num_changed_tiles
    f6 = revealed_melded_tiles_34_array
    f7 = revealed_melds_discarded_tiles_34_array
    f8 = discarded_bonus_tiles_34_array
    f9 = discarded_tiles_34_array
    f10 = table_revealed_tiles
    f11 = changed_tiles_34_array

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11


def gen_scores_features(table_count_of_honba_sticks,
                            table_count_of_remaining_tiles,
                            table_count_of_riichi_sticks,
                            table_round_number,
                            table_round_wind,
                            table_turns,
                            table_dealer_seat,
                            table_dora_indicators,
                            table_dora_tiles,
                            table_revealed_tiles,
                            player_winning_tiles,                   
                            player_discarded_tiles, 
                            player_dealer_seat,
                            player_in_riichi,
                            player_is_dealer,
                            player_is_open_hand,              
                            player_melds,                
                            player_name,
                            player_position,
                            player_rank,
                            player_scores,
                            player_seat,
                            player_uma):

    #ft = Features()
    
    num_revealed_melds = len(player_melds)
    discarded_tiles = [d[0] for d in player_discarded_tiles]
    discarded_tiles_34_array = TilesConverter.to_34_array(discarded_tiles)
#    num_discarded_tiles = len(discarded_tiles)
#    changed_tiles = [d[0] for d in player_discarded_tiles if d[1]==0]
#    changed_tiles_34_array = TilesConverter.to_34_array(changed_tiles)
#    num_changed_tiles = len(changed_tiles)
    revealed_melded_tiles = reduce(lambda x,y:x+y, [m[0] for m in player_melds]) if player_melds else []   
    revealed_melded_tiles_34_array = TilesConverter.to_34_array(revealed_melded_tiles)
#    revealed_melds_discarded_tiles = [m[2][0] for m in player_melds] 
#    revealed_melds_discarded_tiles_34_array = TilesConverter.to_34_array(revealed_melds_discarded_tiles)
#    discarded_bonus_tiles = [d for d in discarded_tiles if d in table_dora_tiles]
#    discarded_bonus_tiles_34_array = TilesConverter.to_34_array(discarded_bonus_tiles)
                                             
    revealed_melded_tiles = reduce(lambda x,y:x+y, [m[0] for m in player_melds]) if player_melds else []  # 0-135
    bonus_tiles_in_revealed_melds = [t for t in revealed_melded_tiles if t in table_dora_tiles] # 0-135
    
    num_bonus_tiles_in_revealed_melds = len(bonus_tiles_in_revealed_melds)

    ft = FeaturesOfScore(discarded_tiles_34_array,
                         revealed_melded_tiles_34_array,
                         table_revealed_tiles)

    f1 = player_in_riichi
    f2 = player_is_dealer
    f3 = player_is_open_hand
    f4 = num_bonus_tiles_in_revealed_melds
    f5 = num_revealed_melds
    
    f6 = discarded_tiles_34_array
    f7 = revealed_melded_tiles_34_array
    
    f8 = ft.is_discards_all_simples()
    f9 = ft.num_revealed_melds_of_wind_tiles()
    f10 = ft.num_revealed_melds_of_dragon_tiles()
    f11 = ft.can_hands_be_all_pons()
    f12 = ft.can_hands_be_flush()
    f13 = ft.can_hands_be_all_simples()
    
#    f2 = num_revealed_melds
#    f3 = num_discarded_tiles
#    f4 = table_turns
#    f5 = num_changed_tiles
#    f6 = revealed_melded_tiles_34_array
#    f7 = revealed_melds_discarded_tiles_34_array
#    f8 = discarded_bonus_tiles_34_array
#    f9 = discarded_tiles_34_array
#    f10 = table_revealed_tiles
#    f11 = changed_tiles_34_array

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13

def gen_wfw_scores_features(table_count_of_honba_sticks,
                            table_count_of_remaining_tiles,
                            table_count_of_riichi_sticks,
                            table_round_number,
                            table_round_wind,
                            table_turns,
                            table_dealer_seat,
                            table_dora_indicators,
                            table_dora_tiles,
                            table_revealed_tiles,
                            player_winning_tiles,                   
                            player_discarded_tiles, 
                            player_dealer_seat,
                            player_in_riichi,
                            player_is_dealer,
                            player_is_open_hand,              
                            player_melds,                
                            player_name,
                            player_position,
                            player_rank,
                            player_scores,
                            player_seat,
                            player_uma):
    # This function should be the same as gen_scores_features
    return gen_scores_features(table_count_of_honba_sticks,
                            table_count_of_remaining_tiles,
                            table_count_of_riichi_sticks,
                            table_round_number,
                            table_round_wind,
                            table_turns,
                            table_dealer_seat,
                            table_dora_indicators,
                            table_dora_tiles,
                            table_revealed_tiles,
                            player_winning_tiles,                   
                            player_discarded_tiles, 
                            player_dealer_seat,
                            player_in_riichi,
                            player_is_dealer,
                            player_is_open_hand,              
                            player_melds,                
                            player_name,
                            player_position,
                            player_rank,
                            player_scores,
                            player_seat,
                            player_uma)
    
def gen_is_waiting_targets(player_winning_tiles):
    is_waiting = 1 if len(player_winning_tiles)>0 else 0
    return is_waiting

def gen_waiting_tiles_targets(player_winning_tiles):
    
    #return [1 if t>0 else 0 for t in TilesConverter.to_34_array(player_winning_tiles)]
    assert len(set(player_winning_tiles))==len(player_winning_tiles),"Winning tiles should be in 34 format and NOT duplicated."
    waiting_tiles = [0]*34
    for n in player_winning_tiles:
        waiting_tiles[n] = 1
    return waiting_tiles

def gen_scores_targets(score_str):
    """return log(score) as in the literature (for HS)
    Note: score is negative here.
    """
    score = float(score_str)
    return np.log(-score)

def gen_wfw_scores_targets(score_str):
    """return log(score) as in the literature (for HS_WFW)
    Note: score is positive here.
    """
    score = float(score_str)
    return np.log(score)
    
"""
Ref: https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/33259578#33259578

def concatenate_csr_matrices_by_rows(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data),axis=0)
    new_indices = np.concatenate((matrix1.indices, matrix2.indices),axis=0)
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return sp.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))    

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return sp.sparse.csc_matrix((new_data, new_indices, new_ind_ptr))
"""

def save_sparse_csr(filename, array):
    """
    :param filename: full name(with directory) of file to be saved
    :param array: full np.array matrix to be saved
    """
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    
def save_arr(filename, array):
    np.save(filename, array)
    
def load_arr(filename):
    return np.load(filename)


def train_is_waiting_partial_fit(load_classifier=False, save_classifier=False):
    """
    Logistic regression model for waiting prediction.
    
    tile: int (0-33). the tile number we want to train, i.e., our target label
    """
    full_dir = abs_data_path+"/train_model/data/is_waiting/full_for_partial_fit"
    dir_names = os.listdir(full_dir)
    
    # classifier 1: 
    hidden_layers = (100,)*4
    classifier = MLPClassifier(verbose=True, 
                               hidden_layer_sizes=hidden_layers,
                               learning_rate_init=0.001,
                               batch_size=2000)

    # classifier 2:
#    classifier = SGDClassifier(verbose=True,
#                               loss="log",
#                               class_weight={1:0.85, 0:0.15})
    
    # get full target vector
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        #print(directory)
        # try to load data
        try:
            target = load_sparse_csr(directory + "is_waiting_sparse_target.npz")
        except:
            raise("The training data must be ready at data/is_waiting/full/ before training model.")
        # concatenate the target vector to get full vector
        if "target_all" in locals():
            target_all = sp.sparse.hstack((target_all, target))
        else:
            target_all = target
        #print(target_all.shape)
     
    avg_accuracy_scores = []
    avg_auc_scores = []    
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        # try to load data
        try:
            sparse_features = load_sparse_csr(directory + "is_waiting_sparse_features.npz")
            target = load_sparse_csr(directory + "is_waiting_sparse_target.npz")
        except:
            raise("The training data must be ready at {}/train_model/data/is_waiting/full/ before training model.".format(full_dir))
        
        X = sparse_features #[0:600000,:]
        y = target.T #[0:600000,:] # just forgot to make the dimension consistent
        
        y = np.array(y.todense()).ravel()
        y_all = np.array(target_all.T.todense()).ravel()
        
        #print(X.shape, y.shape)
        
        
        classifier.partial_fit(X, y, np.unique(y_all))
        
        features_path = abs_data_path+"/train_model/data/is_waiting/test/is_waiting_sparse_features.npz"
        target_path = abs_data_path+"/train_model/data/is_waiting/test/is_waiting_sparse_target.npz"
        # try to load data
        try:
            sparse_features = load_sparse_csr(features_path)
            target = load_sparse_csr(target_path)
        except:
            raise("The training data must be ready at data/is_waiting/full/ before training model.")
            
        avg_accuracy_score, avg_auc_score = validate_classifier(clf=classifier,
                                                                sparse_features=sparse_features,
                                                                sparse_target=target)
        avg_accuracy_scores.append(avg_accuracy_score)
        avg_auc_scores.append(avg_auc_score)
        
        print("--------------------------------")
    
    if save_classifier:
        classifier_name = "is_waiting.sav"
        pickle.dump(classifier, open(abs_data_path+"/train_model/trained_models/"+classifier_name, 'wb'))
        
    return classifier, avg_accuracy_scores, avg_auc_scores

def train_waiting_tiles_partial_fit(tile=1, load_classifier=False, save_classifier=False):
    """
    Logistic regression model for waiting prediction.
    
    tile: int (0-33). the tile number we want to train, i.e., our target label
    """
    full_dir = abs_data_path+"/train_model/data/waiting_tiles/full_for_partial_fit"
    dir_names = os.listdir(full_dir)
#    classifier = MLPClassifier(verbose=True, 
#                               learning_rate_init=0.001,
#                               batch_size=2000)
    classifier = SGDClassifier(verbose=True,
                               loss="log",
                               class_weight={1:0.99, 0:0.01})
    
    # get full target vector
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        #print(directory)
        # try to load data
        try:
            target = load_sparse_csr(directory + "waiting_tiles_sparse_targets.npz")
        except:
            raise("The training data must be ready at {}/train_model/data/waiting_tiles/full/ before training model.".format(full_dir))
        # concatenate the target vector to get full vector
        if "target_all" in locals():
            target_all = sp.sparse.hstack((target_all, target[:,tile]))
        else:
            target_all = target[:,tile]
        #print(target_all.shape)
     
    avg_accuracy_scores = []
    avg_auc_scores = []    
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        # try to load data
        try:
            sparse_features = load_sparse_csr(directory + "waiting_tiles_sparse_features.npz")
            target = load_sparse_csr(directory + "waiting_tiles_sparse_targets.npz")
            target = target[:,tile]
        except:
            filename="waiting_tiles_sparse_features.npz OR waiting_tiles_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        X = sparse_features #[0:600000,:]
        y = target.T #[0:600000,:] # just forgot to make the dimension consistent
        
        y = np.array(y.todense()).ravel()
        y_all = np.array(target_all.T.todense()).ravel()
        
        #print(X.shape, y.shape)
        
        
        classifier.partial_fit(X, y, np.unique(y_all))
        
        features_path = abs_data_path+"/train_model/data/waiting_tiles/test/waiting_tiles_sparse_features.npz"
        target_path = abs_data_path+"/train_model/data/waiting_tiles/test/waiting_tiles_sparse_targets.npz"
        
        # try to load data
        try:
            sparse_features = load_sparse_csr(features_path)
            sparse_targets = load_sparse_csr(target_path)
            sparse_target=sparse_targets[:,tile]
        except:
            filename="waiting_tiles_sparse_features.npz OR waiting_tiles_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
            
        avg_accuracy_score, avg_auc_score = validate_classifier(clf=classifier,
                                                                sparse_features=sparse_features,
                                                                sparse_target=sparse_target)
        avg_accuracy_scores.append(avg_accuracy_score)
        avg_auc_scores.append(avg_auc_score)
        
        print("--------------------------------")
    
    if save_classifier:
        classifier_name = "waiting_tile_{}.sav".format(tile)
        pickle.dump(classifier, open(abs_data_path+"/train_model/trained_models/"+classifier_name, 'wb'))
        
    return classifier, avg_accuracy_scores, avg_auc_scores

def train_scores_partial_fit(load_classifier=False, save_classifier=False, save_scaler=False):
    """
    Linear regression model for score (HS) prediction.
    
    tile: int (0-33). the tile number we want to train, i.e., our target label
    """
    full_dir = abs_data_path+"/train_model/data/scores/full_for_partial_fit"
    dir_names = os.listdir(full_dir)
    hidden_layers = (100,)*8
    classifier = MLPRegressor(verbose=True, 
                              hidden_layer_sizes=hidden_layers,
                               learning_rate_init=0.001,
                               batch_size="auto")
#    classifier = SGDRegressor(verbose=True,
#                               loss="squared_loss")

    scaler = StandardScaler(with_mean=False)
    
    # get full target vector
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        #print(directory)
        # try to load data
        try:
            target = load_sparse_csr(directory + "scores_sparse_targets.npz")
        except:
            raise("The training data must be ready at data/scores/full/ before training model.")
        # concatenate the target vector to get full vector
        if "target_all" in locals():
            target_all = sp.sparse.hstack((target_all, target[:,:]))
        else:
            target_all = target[:,:]
        #print(target_all.shape)
     
    avg_mse_scores = [] 
    
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        # try to load training data
        try:
            sparse_features = load_sparse_csr(directory + "scores_sparse_features.npz")
            target = load_sparse_csr(directory + "scores_sparse_targets.npz")
            target = target[:,:]
        except:
            filename="scores_sparse_features.npz OR scores_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        X = sparse_features #[0:600000,:]
        y = target.T #[0:600000,:] # just forgot to make the dimension consistent
        
        y = np.array(y.todense()).ravel()
        #y_all = np.array(target_all.T.todense()).ravel()
        
        #print(X.shape, y.shape)
        scaler.partial_fit(X)
        X = scaler.transform(X)
        classifier.partial_fit(X, y)
        
        features_path = abs_data_path+"/train_model/data/scores/test/scores_sparse_features.npz"
        target_path = abs_data_path+"/train_model/data/scores/test/scores_sparse_targets.npz"
        
        # try to load testing data
        try:
            sparse_features = load_sparse_csr(features_path)
            sparse_targets = load_sparse_csr(target_path)
            sparse_target=sparse_targets[:,:]
        except:
            filename="scores_sparse_features.npz OR scores_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
            
        mse_score = validate_regressor(clf=classifier,
                                       sparse_features=sparse_features,
                                       sparse_target=sparse_target,
                                       scaler=scaler)
        avg_mse_scores.append(mse_score)
        
        print("--------------------------------")
    
    if save_classifier:
        classifier_name = "scores.sav"
        pickle.dump(classifier, open(abs_data_path+"/train_model/trained_models/"+classifier_name, 'wb'))
    if save_scaler:
        scaler_name = "scaler_scores.sav"
        pickle.dump(scaler, open(abs_data_path+"/train_model/trained_models/"+scaler_name, 'wb'))
    
    return classifier, avg_mse_scores

def train_wfw_scores_partial_fit(load_classifier=False, save_classifier=False, save_scaler=False):
    """
    Linear regression model for score (HS_WFW) prediction.
    
    tile: int (0-33). the tile number we want to train, i.e., our target label
    """
    full_dir = abs_data_path+"/train_model/data/wfw_scores/full_for_partial_fit"
    dir_names = os.listdir(full_dir)
    hidden_layers = (100,)*8
    classifier = MLPRegressor(verbose=True, 
                              hidden_layer_sizes=hidden_layers,
                               learning_rate_init=0.001,
                               batch_size="auto")
#    classifier = SGDRegressor(verbose=True,
#                               loss="squared_loss")

    scaler = StandardScaler(with_mean=False)
    
    # get full target vector
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        #print(directory)
        # try to load data
        try:
            target = load_sparse_csr(directory + "scores_sparse_targets.npz")
        except:
            raise("The training data must be ready at data/wfw_scores/full/ before training model.")
        # concatenate the target vector to get full vector
        if "target_all" in locals():
            target_all = sp.sparse.hstack((target_all, target[:,:]))
        else:
            target_all = target[:,:]
        #print(target_all.shape)
     
    avg_mse_scores = [] 
    
    for dn in dir_names:
        directory = full_dir + "/" + dn + "/"
        # try to load training data
        try:
            sparse_features = load_sparse_csr(directory + "scores_sparse_features.npz")
            target = load_sparse_csr(directory + "scores_sparse_targets.npz")
            target = target[:,:]
        except:
            filename="scores_sparse_features.npz OR scores_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        X = sparse_features #[0:600000,:]
        y = target.T #[0:600000,:] # just forgot to make the dimension consistent
        
        y = np.array(y.todense()).ravel()
        #y_all = np.array(target_all.T.todense()).ravel()
        
        #print(X.shape, y.shape)
        scaler.partial_fit(X)
        X = scaler.transform(X)
        classifier.partial_fit(X, y)
        
        features_path = abs_data_path+"/train_model/data/wfw_scores/test/scores_sparse_features.npz"
        target_path = abs_data_path+"/train_model/data/wfw_scores/test/scores_sparse_targets.npz"
        
        # try to load testing data
        try:
            sparse_features = load_sparse_csr(features_path)
            sparse_targets = load_sparse_csr(target_path)
            sparse_target=sparse_targets[:,:]
        except:
            filename="scores_sparse_features.npz OR scores_sparse_targets.npz"
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
            
        mse_score = validate_regressor(clf=classifier,
                                       sparse_features=sparse_features,
                                       sparse_target=sparse_target,
                                       scaler=scaler)
        avg_mse_scores.append(mse_score)
        
        print("--------------------------------")
    
    if save_classifier:
        classifier_name = "wfw_scores.sav"
        pickle.dump(classifier, open(abs_data_path+"/train_model/trained_models/"+classifier_name, 'wb'))
    if save_scaler:
        scaler_name = "scaler_wfw_scores.sav"
        pickle.dump(scaler, open(abs_data_path+"/train_model/trained_models/"+scaler_name, 'wb'))
        
    return classifier, avg_mse_scores

def validate_classifier(clf, sparse_features, sparse_target):
    
    X_test = sparse_features #[0:600000,:]
    y_test = sparse_target.T #[0:600000,:] # just forgot to make the dimension consistent
    
    y_test = np.array(y_test.todense()).ravel()
    
    auc_scores = []
    accuracy_scores = []
    
    for i in range(1):
        #print("random_state is ", i,", and accuracy metrics are:")
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i) 
        
        # ADD: ravel to avoid warning  
        
        y_pred = clf.predict(X_test)
        accuracy_score = metrics.accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy_score)
        #print("accuracy score: {}".format(metrics.accuracy_score(y_test, y_pred)))
        try:
            auc_score = metrics.roc_auc_score(y_test, y_pred)
            auc_scores.append(auc_score)
        except ValueError:
            # Only one class present in y_true. ROC AUC score is not defined in that case.
            pass
        
        num_test_data = len(y_test)
        num_test_ones = 0
        num_pred_ones = 0
        num_test_pred_both_ones = 0
        for j in range(num_test_data):
            if y_test[j]==1:
                num_test_ones += 1
            if y_pred[j]==1:
                num_pred_ones += 1
            if (y_test[j]==1) and (y_pred[j]==1):
                num_test_pred_both_ones += 1
                    
        print("\t({}) num_test_data:{}, num_test_ones:{}, \n\t\tnum_pred_ones:{}, both_ones:{}".format(
                i, num_test_data, num_test_ones, num_pred_ones, num_test_pred_both_ones)
        )
        
    if accuracy_scores:        
        avg_accuracy_score = sum(accuracy_scores)/len(accuracy_scores)
        print("Average accuracy score: {:.4f}".format(avg_accuracy_score)) 
    if auc_scores:
        avg_auc_score = sum(auc_scores)/len(auc_scores)
        print("Average AUC score: {:.4f}".format(avg_auc_score))
    print("Accuracy scores: "+", ".join("{:.4f}".format(k) for k in accuracy_scores))  
    print("AUC scores: "+", ".join("{:.4f}".format(k) for k in auc_scores))
    
    if accuracy_scores and auc_scores:
        return avg_accuracy_score, avg_auc_score
    elif accuracy_scores:
        print("AUC score is not available.")
        return avg_accuracy_score, -1
    elif auc_scores:
        print("Accuracy score is not available.")
        return -1, avg_auc_score
    else:
        return -1, -1


def validate_regressor(clf, sparse_features, sparse_target, scaler):
    
    X_test = sparse_features #[0:600000,:]
    y_test = sparse_target.T #[0:600000,:] # just forgot to make the dimension consistent
    y_test = np.array(y_test.todense()).ravel()
    
    mse_scores = []
    for i in range(1): 
        X_test = scaler.transform(X_test)
#        print(X_test[1:4,:].todense(),"!!!")
#        print(X_test.shape)
        y_pred = clf.predict(X_test)
        mse_score = metrics.mean_squared_error(y_test, y_pred)
        mse_scores.append(mse_score)
        print("MSE score: {}".format(mse_score))
    print("*Check*: {}, {}".format(y_test[0:5], y_pred[0:5]))
        
    if mse_scores:
        return sum(mse_scores)*1.0/len(mse_scores)
    else:
        print("MSE score is not available.")
        return -1


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def plot_scores(avg_accuracy_scores, avg_auc_scores, save_path, save_name):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(range(len(avg_accuracy_scores)), avg_accuracy_scores, marker="o")
    plt.xlabel("sample batch")
    plt.ylabel("accuracy score")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(range(len(avg_auc_scores)), avg_auc_scores, marker="o", color="r")
    plt.xlabel("sample batch")
    plt.ylabel("AUC score")
    
    fig1.savefig(save_path + save_name[0])
    fig2.savefig(save_path + save_name[1])
    

def is_waiting_data_preprocessing():
    debuglogs = os.listdir(abs_data_path+"/debuglogs/")
    debuglogs_cp = debuglogs[:]
    for debuglog in debuglogs:
        if "testwaiting" not in debuglog:
            debuglogs_cp.remove(debuglog)
    debuglogs = debuglogs_cp      
    log_index = [int(d) for d in [log.split("_")[0][11:] for log in debuglogs]]
    
    for n in range(1,len(debuglogs)+1):
        log_idx = log_index.index(n)
        debuglog = debuglogs[log_idx]
        #print(debuglog)
    
        debuglog_name = debuglog.split(".")[0]
        print("{}. -----------------------------".format(n))
        print(debuglog_name)
        
        # remove files in chunk and full
        directory = abs_data_path+"/train_model/data/is_waiting/chunk/"
        files_in_chunk = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_chunk:
            os.remove(directory+f)
        files_in_chunk = os.listdir(abs_data_path+"/train_model/data/is_waiting/chunk/")
        #print("chunk after remove:{}".format(files_in_chunk))

        directory = abs_data_path+"/train_model/data/is_waiting/full/"
        files_in_full = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_full:
            os.remove(directory+f)
        files_in_full = os.listdir(abs_data_path+"/train_model/data/is_waiting/full/")
        #print("chunk after remove:{}".format(files_in_chunk))
        
        if files_in_chunk:
            sys.exit("Directory `chunk` is NOT empty.")
        if files_in_full:
            sys.exit("Directory `full` is NOT empty.")

        tic = time.time()
        load_and_process_is_waiting_data(debuglog_name=debuglog_name, # debuglog_name="test_700000_lines[0_700000]"
                              num_lines=100000000, 
                              chunk_size=100000)
        #load_and_process_data(num_lines=20000, chunk_size=5000)
        toc = time.time()
        print("Finished load and process: {:.2f} seconds".format(toc-tic))
        
        tic = time.time()
        features = load_and_process_is_waiting_sparse_data(dir_path=abs_data_path+"/train_model/data/is_waiting/chunk/")
        save_sparse_csr(abs_data_path+"/train_model/data/is_waiting/full/is_waiting_sparse_features", features)
        toc = time.time()
        print("Finished load sparse: {:.2f} seconds".format(toc-tic))
        
        
        # make a new dir to store the aggregated feature and target data
        os.mkdir(abs_data_path+"/train_model/data/is_waiting/full_for_partial_fit/full_{}".format(n))
        # copy the data to the folder
        copytree(abs_data_path+"/train_model/data/is_waiting/full", abs_data_path+"/train_model/data/is_waiting/full_for_partial_fit/full_{}/".format(n))
        n += 1
        
        print("-------------------------\n")
        
def waiting_tiles_data_preprocessing():
    debuglogs = os.listdir(abs_data_path+"/debuglogs/")
    debuglogs_cp = debuglogs[:]
    for debuglog in debuglogs:
        if "testwaiting" not in debuglog:
            debuglogs_cp.remove(debuglog)
    debuglogs = debuglogs_cp      
    log_index = [int(d) for d in [log.split("_")[0][11:] for log in debuglogs]]
    
    for n in range(1,len(debuglogs)+1):
        log_idx = log_index.index(n)
        debuglog = debuglogs[log_idx]
        #print(debuglog)
    
        debuglog_name = debuglog.split(".")[0]
        print("{}. -----------------------------".format(n))
        print(debuglog_name)
        
        # remove files in chunk and full
        directory = abs_data_path+"/train_model/data/waiting_tiles/chunk/"
        files_in_chunk = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_chunk:
            os.remove(directory+f)
        files_in_chunk = os.listdir(abs_data_path+"/train_model/data/waiting_tiles/chunk/")
        #print("chunk after remove:{}".format(files_in_chunk))

        directory = abs_data_path+"/train_model/data/waiting_tiles/full/"
        files_in_full = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_full:
            os.remove(directory+f)
        files_in_full = os.listdir(abs_data_path+"/train_model/data/waiting_tiles/full/")
        #print("chunk after remove:{}".format(files_in_chunk))
        
        if files_in_chunk:
            sys.exit("Directory `chunk` is NOT empty.")
        if files_in_full:
            sys.exit("Directory `full` is NOT empty.")

        tic = time.time()
        load_and_process_waiting_tiles_data(debuglog_name=debuglog_name, # debuglog_name="test_700000_lines[0_700000]"
                              num_lines=100000000, 
                              chunk_size=100000)
        #load_and_process_data(num_lines=20000, chunk_size=5000)
        toc = time.time()
        print("Finished load and process: {:.2f} seconds".format(toc-tic))
        
        tic = time.time()
        targets, features = load_and_process_waiting_tiles_sparse_data(dir_path=abs_data_path+"/train_model/data/waiting_tiles/chunk/")
        save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/full/waiting_tiles_sparse_features", features)
        save_sparse_csr(abs_data_path+"/train_model/data/waiting_tiles/full/waiting_tiles_sparse_targets", targets)
        toc = time.time()
        print("Finished load sparse: {:.2f} seconds".format(toc-tic))
        
        
        # make a new dir to store the aggregated feature and target data
        os.mkdir(abs_data_path+"/train_model/data/waiting_tiles/full_for_partial_fit/full_{}".format(n))
        # copy the data to the folder
        copytree(abs_data_path+"/train_model/data/waiting_tiles/full", abs_data_path+"/train_model/data/waiting_tiles/full_for_partial_fit/full_{}/".format(n))
        n += 1
        
        print("-------------------------\n")

    
def scores_data_preprocessing():
    debuglogs = os.listdir(abs_data_path+"/debuglogs/")
    debuglogs_cp = debuglogs[:]
    for debuglog in debuglogs:
        if "testscores" not in debuglog:
            debuglogs_cp.remove(debuglog)
    debuglogs = debuglogs_cp          
    log_index = [int(d) for d in [log.split("_")[0][10:] for log in debuglogs]]
    
    for n in range(1,len(debuglogs)+1):
        log_idx = log_index.index(n)
        debuglog = debuglogs[log_idx]
        #print(debuglog)
    
        debuglog_name = debuglog.split(".")[0]
        print("{}. -----------------------------".format(n))
        print(debuglog_name)
        
        # remove files in chunk and full
        directory = abs_data_path+"/train_model/data/scores/chunk/"
        files_in_chunk = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_chunk:
            os.remove(directory+f)
        files_in_chunk = os.listdir(abs_data_path+"/train_model/data/scores/chunk/")
        #print("chunk after remove:{}".format(files_in_chunk))

        directory = abs_data_path+"/train_model/data/scores/full/"
        files_in_full = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_full:
            os.remove(directory+f)
        files_in_full = os.listdir(abs_data_path+"/train_model/data/scores/full/")
        #print("chunk after remove:{}".format(files_in_chunk))
        
        if files_in_chunk:
            sys.exit("Directory `chunk` is NOT empty.")
        if files_in_full:
            sys.exit("Directory `full` is NOT empty.")

        tic = time.time()
        load_and_process_scores_data(debuglog_name=debuglog_name, # debuglog_name="test_700000_lines[0_700000]"
                              num_lines=100000000, 
                              chunk_size=100000)
        #load_and_process_data(num_lines=20000, chunk_size=5000)
        toc = time.time()
        print("Finished load and process: {:.2f} seconds".format(toc-tic))
        
        tic = time.time()
        targets, features = load_and_process_scores_sparse_data(dir_path=abs_data_path+"/train_model/data/scores/chunk/")
        save_sparse_csr(abs_data_path+"/train_model/data/scores/full/scores_sparse_features", features)
        save_sparse_csr(abs_data_path+"/train_model/data/scores/full/scores_sparse_targets", targets)
        toc = time.time()
        print("Finished load sparse: {:.2f} seconds".format(toc-tic))
        
        
        # make a new dir to store the aggregated feature and target data
        os.mkdir(abs_data_path+"/train_model/data/scores/full_for_partial_fit/full_{}".format(n))
        # copy the data to the folder
        copytree(abs_data_path+"/train_model/data/scores/full", abs_data_path+"/train_model/data/scores/full_for_partial_fit/full_{}/".format(n))
        n += 1
        
        print("-------------------------\n")
        
def wfw_scores_data_preprocessing():
    debuglogs = os.listdir(abs_data_path+"/debuglogs/")
    debuglogs_cp = debuglogs[:]
    for debuglog in debuglogs:
        if "test_wfw_scores" not in debuglog:
            debuglogs_cp.remove(debuglog)
    debuglogs = debuglogs_cp          
    log_index = [int(d) for d in [log.split("_")[2][6:] for log in debuglogs]]
    
    for n in range(1,len(debuglogs)+1):
        log_idx = log_index.index(n)
        debuglog = debuglogs[log_idx]
        #print(debuglog)
    
        debuglog_name = debuglog.split(".")[0]
        print("{}. -----------------------------".format(n))
        print(debuglog_name)
        
        # remove files in chunk and full
        directory = abs_data_path+"/train_model/data/wfw_scores/chunk/"
        files_in_chunk = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_chunk:
            os.remove(directory+f)
        files_in_chunk = os.listdir(abs_data_path+"/train_model/data/wfw_scores/chunk/")
        #print("chunk after remove:{}".format(files_in_chunk))

        directory = abs_data_path+"/train_model/data/wfw_scores/full/"
        files_in_full = os.listdir(directory)
        #print("chunk before remove:{}".format(files_in_chunk))
        for f in files_in_full:
            os.remove(directory+f)
        files_in_full = os.listdir(abs_data_path+"/train_model/data/wfw_scores/full/")
        #print("chunk after remove:{}".format(files_in_chunk))
        
        if files_in_chunk:
            sys.exit("Directory `chunk` is NOT empty.")
        if files_in_full:
            sys.exit("Directory `full` is NOT empty.")

        tic = time.time()
        load_and_process_wfw_scores_data(debuglog_name=debuglog_name, # debuglog_name="test_700000_lines[0_700000]"
                                         num_lines=100000000, 
                                         chunk_size=100000)
        #load_and_process_data(num_lines=20000, chunk_size=5000)
        toc = time.time()
        print("Finished load and process: {:.2f} seconds".format(toc-tic))
        
        tic = time.time()
        targets, features = load_and_process_wfw_scores_sparse_data(dir_path=abs_data_path+"/train_model/data/wfw_scores/chunk/")
        save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/full/scores_sparse_features", features)
        save_sparse_csr(abs_data_path+"/train_model/data/wfw_scores/full/scores_sparse_targets", targets)
        toc = time.time()
        print("Finished load sparse: {:.2f} seconds".format(toc-tic))
        
        
        # make a new dir to store the aggregated feature and target data
        os.mkdir(abs_data_path+"/train_model/data/wfw_scores/full_for_partial_fit/full_{}".format(n))
        # copy the data to the folder
        copytree(abs_data_path+"/train_model/data/wfw_scores/full", abs_data_path+"/train_model/data/wfw_scores/full_for_partial_fit/full_{}/".format(n))
        n += 1
        
        print("-------------------------\n")        
        
        
class WaitingTilesEvaluation(object):
    
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
            
            
if __name__=="__main__":

    tic = time.time()
    print(abs_path)
    
#    is_waiting_data_preprocessing()
#    waiting_tiles_data_preprocessing() 
#    scores_data_preprocessing() 
    
#    clf, avg_accuracy_scores, avg_auc_scores = train_is_waiting_partial_fit(load_classifier=False, save_classifier=True)
#    plot_scores(avg_accuracy_scores,
#                avg_auc_scores,
#                save_path="trained_classifiers/plots/",
#                save_name=("Accuracy_is_waiting",
#                           "AUC_is_waiting")
#                )
    
#    for tile in range(34):
#        clf, avg_accuracy_scores, avg_auc_scores = train_waiting_tiles_partial_fit(tile=tile, load_classifier=False, save_classifier=True)     
#        plot_scores(avg_accuracy_scores, 
#                    avg_auc_scores,
#                    save_path="trained_classifiers/plots/",
#                    save_name=("Accuracy_waiting_tile_{}.png".format(tile),
#                               "AUC_waiting_tile_{}.png".format(tile))
#                    )
    
#    clf, avg_mse_scores = train_scores_partial_fit(load_classifier=False, save_classifier=True)
#    
#    waiting_tiles_evaluation = WaitingTilesEvaluation()
#    evaluation = waiting_tiles_evaluation.accuracy_of_prediction()
#    print("Evaluation value: {}".format(evaluation))

    toc = time.time()
    print("Elapsed time: {:.2} seconds.".format(toc-tic)) 


    

    
        


    