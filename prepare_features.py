# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:29:13 2017

@author: joseph.chen
"""
import os
import re
import sys

from mahjong.ai.check_waiting import CheckWaiting
from mahjong.tile import TilesConverter
from mahjong.ai.agari import Agari

def is_waiting(hand:list,
               open_sets:list,
               wall_tiles:list) ->dict:
    """
    :param hand: list of hand tiles in 136-tile format
    :param open_sets: "list of list" of open sets tiles in 136-tile format
    :param wall_tiles: list of wall tiles in 136-tile format
    
    :return {} if not waiting; {tile1: yaku_result1, tile2: yaku_result2, ...} if waiting                
    
    ------------------------------
    Tile in 136 format (34 format)
    1 man: 0, 1, 2, 3 (0)
    2 man: 4, 5, 6, 7 (1)
    3 man: 8, 9, 10, 11 (2)
    4 man: 12, 13, 14, 15 (3)
    5 man: 16, 17, 18, 19 (4)
    6 man: 20, 21, 22, 23 (5)
    7 man: 24, 25, 26, 27 (6)
    8 man: 28, 29, 30, 31 (7)
    9 man: 32, 33, 34, 35 (8)
    1 suo: 36, 37, 38, 39 (9)
    2 suo: 40, 41, 42, 43 (10)
    3 suo: 44, 45, 46, 47 (11)
    4 suo: 48, 49, 50, 51 (12)
    5 suo: 52, 53, 54, 55 (13)
    6 suo: 56, 57, 58, 59 (14)
    7 suo: 60, 61, 62, 63 (15)
    8 suo: 64, 65, 66, 67 (16)
    9 suo: 68, 69, 70, 71 (17)
    1 pin: 72, 73, 74, 75 (18)
    2 pin, 76, 77, 78, 79 (19)
    3 pin: 80, 81, 82, 83 (20)
    4 pin: 84, 85, 86, 87 (21)
    5 pin: 88, 89, 90, 91 (22)
    6 pin: 92, 93, 94, 95 (23)
    7 pin: 96, 97, 98, 99 (24)
    8 pin: 100, 101, 102, 103 (25)
    9 pin: 104, 105, 106, 107 (26)
    East: 108, 109, 110, 111 (27)
    South: 112, 113, 114, 115 (28)
    West: 116, 117, 118, 119 (29)
    North: 120, 121, 122, 123 (30)
    White: 124, 125, 126, 127 (31)
    Green: 128, 129, 130, 131 (32)
    Red: 132, 133, 134, 135 (33)
    """
    agari = Agari()
    check_waiting = CheckWaiting()   
    waiting_result = {}
    for tile in wall_tiles:
        open_sets_flattened = [v for open_set in open_sets for v in open_set]
        completed_hand = hand + open_sets_flattened + [tile]
        hand_34_array = TilesConverter.to_34_array(completed_hand)
        melds_34_tiles = [TilesConverter.to_34_tiles(meld) for meld in open_sets]
        
        # check whether this tile is a winning tile
        is_waiting = agari.is_agari(hand_34_array, melds_34_tiles)
        
        # if it is a winning tile, append the completed hand result
        # TODO: we need to consider other params as well, such as riichi and tenhou, etc
        if is_waiting:
            result = check_waiting.check(hand, tile, open_sets=melds_34_tiles)
            assert result['error']==None, "tile {} is not the win tile!".format(tile)
            waiting_result[tile] = result
    
    return waiting_result

if __name__=="__main__":
    print("test check waiting")
    data_path = os.path.join("\\".join(os.path.realpath(__file__).split("\\")[:-3]), "data\logs")
    log_names = os.listdir(data_path)
    for log_name in log_names[0:1]:
        # open each log file and parse the data
        with open(data_path+"\\"+log_name, encoding="utf-8") as f:
            log = f.read()
            data = log.split("Initial Game State:")
            num_game = len(data)-1
            header = data[0].split("\n")
            for line in header:
                if "Lobby" in line:
                    lobby = line.split(":")[1]
                if "Table" in line:
                    table = line.split(":")[1]
                if "red" in line:
                    red = line.split(":")[1]
                if "kui" in line:
                    kui = line.split(":")[1]
                if "ton-nan" in line:
                    ton_nan = line.split(":")[1]
                if "sanma" in line:
                    sanma = line.split(":")[1]
                if "soku" in line:
                    soku = line.split(":")[1]
                if "0:" in line:
                    player0 = line.split(":")[1].split(",") # dan, rate, sex, name
                if "1:" in line:
                    player1 = line.split(":")[1].split(",") # dan, rate, sex, name
                if "2:" in line:
                    player2 = line.split(":")[1].split(",") # dan, rate, sex, name
                if "3:" in line:
                    player3 = line.split(":")[1].split(",") # dan, rate, sex, name
                if "Dealer:" in line:
                    dealer = line.split(":")[1] # player index
            print(lobby,table,red,kui,ton_nan,sanma,soku)
            
            for dt in data[4:5]: #data[1:]:
                              
                pattern = r"Dora Indicator: \w+\s\w+"
                match = re.search(pattern, dt) #dt
                if match:
                    dora_indicator = match.group()
                else:
                    sys.exit("Dora indicator not found!")
                
                pattern = r"Initial Scores:\n\s+\d:\s+\d+\n\s+\d:\s+\d+\n\s+\d:\s+\d+\n\s+\d:\s+\d+\n"
                match = re.search(pattern, dt) #dt
                if match:
                    initial_scores = match.group()
                    #print(initial_scores)
                else:
                    sys.exit("Initial scores not found!")
                    
                pattern = r"Dealer:\s\d"
                match = re.search(pattern, dt) #dt
                if match:
                    dealer = match.group()
                    #print(dealer)
                else:
                    sys.exit("Dealer not found!")
                
                pattern = r"Initial Hands:\n\s+\d:.+\n\s+\d:.+\n\s+\d:.+\n\s+\d:.+\n"
                match = re.search(pattern, dt) #dt
                if match:
                    initial_hands = match.group()
                    #print(initial_hands)
                else:
                    sys.exit("Initial hands not found!")
                    
                pattern = r"Player \d wins."
                match = re.search(pattern, dt) #dt
                if match:
                    player_win = match.group()
                    print(player_win)
                else:
                    pattern = r"Ryukyoku:"
                    match = re.search(pattern, dt) #dt
                    if match:
                        ryukyoku = match.group()
                        print(ryukyoku)
                    
                    
                
             
            
            
    
    
###############################################################################    
#    #meld1 = [0,1,2,3]  # 1m,1m,1m,1m
#    meld1 = [0,1,2,3]  # 1m,1m,1m,1m
#    meld2 = [4,5,6]    # 2m,2m,2m
#    meld3 = [28,29,30]   # 3m,3m,3m
#    meld4 = [12,13,14] # 4m,4m,4m
#    pair = [16,17]     # 6m,6m
#    hand = meld1+meld2+meld3+meld4+pair
#    melds = [meld1, meld2]
#    melds_34 = [TilesConverter.to_34_array(mld) for mld in melds]
#    melds_136 = [TilesConverter.to_136_array(mld) for mld in melds_34]
#    check_waiting = CheckWaiting()
#    #print(check_waiting.check(hand, [[0,0,0,0], [1,1,1]]))
#    print(check_waiting.check(hand))

   
#    agari = Agari()
##    meld1 = [0,1,2,3]     # 1m,1m,1m
##    meld2 = [4,5,6]     # 2m,2m,2m
##    meld3 = [28,29,30]  # 8m,8m,8m
##    meld4 = [12,13,14]  # 4m,4m,4m
##    pair = [16,17]      # 5m,5m    
#
#    meld1 = [0,5,9]     # 1m,2m,3m
#    meld2 = [12,13,15]     # 4m,4m,4m
#    meld3 = [14,17,20]  # 4m,5m,6m
#    meld4 = [120,121,122]  # north, north, north
#    pair = [108,110]      # east,east
#    
#    hand = meld1+meld2+meld3+meld4+pair
#    hand_34_array = TilesConverter.to_34_array(hand)
#    melds = [meld1, meld2]
#    #melds_34 = [TilesConverter.to_34_array(mld) for mld in melds]
#    #melds_in_34 = [[0,0,0,0], [1,1,1]]
#    melds_34_tiles = [TilesConverter.to_34_tiles(meld) for meld in melds]
#    print(agari.is_agari(hand_34_array, melds_34_tiles))
#    
#    check_waiting = CheckWaiting()
#    win_tile = 110 # win tile must be in the hand
#    print(check_waiting.check(hand, win_tile, open_sets=melds_34_tiles))
    