# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:26:12 2017

@author: joseph.chen
"""
from functools import reduce
import sys

from mahjong.tile import TilesConverter
from mahjong.ai.agari import Agari

class ExtractFeatures(object):
    """
    This class extracts features to predict waiting status and waiting tiles.
    """
    def __init__(self):
        # ADD: to determine whether it is waiting or not
        self.agari = Agari()
        # initially meld sets is empty
        self.meld_sets = [[],[],[],[]] # four players
        self.meld_discarded_tiles = [[],[],[],[]] # four players
    
    def get_is_waiting_features(self, table):
        """
        Get features from current table state.
        
        : revealed tiles : table.revealed_tiles, list of tile occurrence (0-4), 
                           of fixed length 34.
        :tiles of player*: table.players[0].tiles, list of tile number (0-135)
        :     meld sets : table.players[0].melds, list of `Meld`
                           Meld has three attributes:
                               opened: True/False
                               type: chi/pon/kan
                               tiles: list of tile number (0-135)
        :discarded tiles : table.players[0].discards, list of `Tile`
                           Tile has attribute:
                               value: tile number (0-135)
        :dora indicators : table.dora_indicators, list of tile number
        :        is dora : table.is_dora(tile number), tile number (0-135)
        :    closed hand*: table.players[0].closed_hand, list of tile number (0-135)
        
        *Note: these info (tiles of player, closed hand) is only visible to self 
               player, we can not see hands of other players anyway.
        """
        
        # If we need one more tile to complete our hand, and this specific tile
        # we want is known to be within the wall tiles, then the hand is waiting.
        current_hand = TilesConverter.to_34_array(table.players[0].tiles)
        winning_tiles = []
        for n in range(34):
            if table.revealed_tiles[n]<4:
                completed_hand = current_hand[:]
                completed_hand[n] += 1
                can_be_waiting = self.agari.is_agari(completed_hand)
                if can_be_waiting:
                    winning_tiles.append(n) # n is the winning tile we want 
                    
        # If there is at least one winning tile available in the wall, the hand
        # is waiting.            
        is_waiting = 0
        if len(winning_tiles)>0:
            is_waiting = 1        
        
        # Discarded tiles can be seen by everybody
        discarded_tiles = [d.value for d in table.players[0].discards]
        discarded_tiles_136_code = TilesConverter.tiles_to_136_code(discarded_tiles)
        discarded_tiles_136_code_str = ",".join([str(d) for d in discarded_tiles_136_code])
        
        # Meld sets (both open and concealed) are also visible to everybody
        meld_sets = [mt.tiles for mt in table.players[0].melds]
        #open_melds_sets = [mt.tiles for mt in table.players[0].melds if mt.opened]    
        if len(meld_sets)>0:
            meld_tiles = reduce(lambda x,y:x+y, meld_sets) 
        else:
            meld_tiles = []
        meld_tiles_136_code = TilesConverter.tiles_to_136_code(meld_tiles)
        meld_tiles_136_code_str = ",".join([str(m) for m in meld_tiles_136_code])    
        
        number_of_revealed_melds = len(meld_sets)
        number_of_discarded_tiles = len(discarded_tiles)
        
#        discarded_tiles_34 = TilesConverter.to_34_array(discarded_tiles)
#        discarded_tiles_34_str = ",".join([str(d) for d in discarded_tiles_34])
#        discarded_tiles_37 = TilesConverter.to_37_array(discarded_tiles)
#        discarded_tiles_37_str = ",".join([str(d) for d in discarded_tiles_37])

        
        
        # We don't try to estimate the probability of waiting until we
        # have sufficient information. Therefore we need at least two
        # discarded tiles in order to do the estimation.
        if len(discarded_tiles)>1:
            dora_discarded = [d for d in discarded_tiles if table.is_dora(d) ]
            dora_discarded_34 = TilesConverter.to_34_array(dora_discarded)
            dora_discarded_34_str = ",".join([str(d) for d in dora_discarded_34])
            
            last1_discarded_tile = discarded_tiles[-1]  # the last discarded tile 
            last1_discarded_tile_37 = TilesConverter.to_37_tiles([last1_discarded_tile])[0]
            last2_discarded_tile = discarded_tiles[-2] # the second last discarded tile
            last2_discarded_tile_37 = TilesConverter.to_37_tiles([last2_discarded_tile])[0]
            string_to_save = "{},{},{},{},{},{},{},{},{},{}".format(
                    is_waiting,
                    number_of_revealed_melds,  # after one_hot_encode, will become 5
                    number_of_discarded_tiles, # after one_hot_encode, will become 19
                    number_of_revealed_melds,  # after one_hot_encode, will become 5 (repeated)
                    last1_discarded_tile_37,   # after one_hot_encode, will become 37
                    last2_discarded_tile_37,   # after one_hot_encode, will become 37
                    last1_discarded_tile_37,   # after one_hot_encode, will become 37 (repeated)
                    discarded_tiles_136_code_str,   # discarded tiles 
                    meld_tiles_136_code_str,        # melded tiles
                    dora_discarded_34_str           # discarded doras                    
                    )
            
            return string_to_save
        return None
    
    def tile_136_to_37(self, tile):
        """
        convert a tile in 136 format to 37 format
        """
        tile //= 4
        if tile==16:
            return 34
        elif tile==52:
            return 35
        elif tile==88:
            return 36
        else:
            return tile
    
    def tile_136_to_34(self, tile):
        """
        convert a tile in 136 format to 34 format
        """
        tile //= 4
        return tile
        
    # TODO: unfinished function
    def get_waiting_tiles_features(self, table):
        """
        Get features from current table state.
        
        : revealed tiles : table.revealed_tiles, list of tile occurrence (0-4), 
                           of fixed length 34.
        :tiles of player*: table.players[0].tiles, list of tile number (0-135)
        :      meld sets : table.players[0].melds, list of `Meld`
                           Meld has three attributes:
                               opened: True/False
                               type: chi/pon/kan
                               tiles: list of tile number (0-135)
        :discarded tiles : table.players[0].discards, list of `Tile`
                           Tile has attribute:
                               value: tile number (0-135)
        :dora indicators : table.dora_indicators, list of tile number
        :        is dora : table.is_dora(tile number), tile number (0-135)
        :    closed hand*: table.players[0].closed_hand, list of tile number (0-135)
        
        *Note: these info (tiles of player, closed hand) is only visible to self 
               player, we can not see hands of other players anyway.
        """
        
        dora_tiles = [d for d in range(136) if table.is_dora(d) ]
        table_turns = min([len(table.players[m].discards)+1 for m in range(4)])
        table_info = "{},{},{},{},{},{},{},{},{},{}".format(
                      table.count_of_honba_sticks,
                      table.count_of_remaining_tiles,
                      table.count_of_riichi_sticks,
                      table.round_number,
                      table.round_wind,
                      table_turns,
                      table.dealer_seat,
                      table.dora_indicators,
                      dora_tiles,
                      table.revealed_tiles                    
                      ) 
        
        player_info = ""     
        for m in range(4): # There are four players
            player = table.players[m]
            
            '''
            player.discards
            player.closed_hand
            player.dealer_seat
            player.in_riichi
            #player.in_defence_mode
            #player.in_tempai
            player.is_dealer
            player.is_open_hand
            player.last_draw
            player.melds
            player.name
            player.position
            player.rank
            player.scores
            player.seat
            player.tiles
            player.uma
            '''
            
            # Discarded tiles can be seen by everybody
            discarded_tiles = [(d.value,1) if d.is_tsumogiri else (d.value,0) for d in player.discards]
            discarded_kinds = [d[0]//4 for d in discarded_tiles]
             
            
            # If we need one more tile to complete our hand, and this specific tile
            # we want is known to be within the wall tiles, then the hand is waiting.
            current_hand = TilesConverter.to_34_array(player.tiles)
            
#            if m==1:
#                print(TilesConverter.to_one_line_string(player.tiles),"~~")
            
            winning_tiles = []
            for n in range(34):
                # if there is no tile available in the wall, or the tile we need
                # to complete a hand has been discarded previously, we do not wait
                # for this tile. But we can still wait for tsumo??
                if (table.revealed_tiles[n]<4) and (n not in set(discarded_kinds)):
                    completed_hand = current_hand[:]
                    completed_hand[n] += 1
                    can_be_waiting = self.agari.is_agari(completed_hand)
#                    if completed_hand==[0,2,0,0,1,1,1,0,0, 0,0,0,0,0,1,1,1,0, 0,0,0,3,0,1,1,1,0, 0,0,0,0,0,0,0]:
#                        sys.exit("bingo!")
#                    print(can_be_waiting, completed_hand, "~~!!")
                    if can_be_waiting:
                        winning_tiles.append(n) # n is the winning tile we want
            
            # Meld sets (both open and concealed) are also visible to everybody
            meld_sets = [mt.tiles for mt in player.melds]
            # meld_sets_str = [TilesConverter.to_one_line_string(ms) for ms in meld_sets]
            # meld_types = [mt.type for mt in player.melds]
            meld_open = [mt.opened for mt in player.melds]
            if meld_sets != self.meld_sets[m]:
                self.meld_discarded_tiles[m].append(discarded_tiles[-1])
                self.meld_sets[m] = meld_sets
            
            # We don't try to estimate the probability of waiting until we
            # have sufficient information. Therefore we need at least two
            # discarded tiles in order to do the estimation.
            
                                
#                dora_tiles = [d for d in range(136) if table.is_dora(d) ]
#                dora_kinds = list(set([dt//4 for dt in dora_tiles]))
#                dora_kind_discarded = [discarded_kinds.count(d) for d in dora_kinds]

            melds = [(meld_sets[k], 1 if meld_open[k] else 0, self.meld_discarded_tiles[m][k]) for k in range(len(meld_sets))]
                
            string_to_save = "{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                    winning_tiles,
                    
                    discarded_tiles, #player.discards
                    #player.closed_hand, # this info is invisible
                    player.dealer_seat,
                    1 if player.in_riichi else 0,
                    #player.in_defence_mode
                    #player.in_tempai
                    1 if player.is_dealer else 0,
                    1 if player.is_open_hand else 0,
                    #player.last_draw, # this info is invisible
                    
#                    meld_sets,
#                    meld_open, 
#                    self.meld_discarded_tiles[m], 
                    melds,
                    
                    
                    player.name if player.name else -1,
                    player.position if player.position else -1,
                    player.rank if player.rank else -1,
                    player.scores if player.scores else -1,
                    player.seat if player.seat else -1,
                    #player.tiles, # this info is invisible
                    player.uma if player.uma else -1
            )
        
            player_info += string_to_save + ";"
        
        if player_info=="":
            return None
        else:
            return table_info + ";" + player_info
        

    def get_scores_features(self, table):
        """
        Get features from current table state.
        
        : revealed tiles : table.revealed_tiles, list of tile occurrence (0-4), 
                           of fixed length 34.
        :tiles of player*: table.players[0].tiles, list of tile number (0-135)
        :      meld sets : table.players[0].melds, list of `Meld`
                           Meld has three attributes:
                               opened: True/False
                               type: chi/pon/kan
                               tiles: list of tile number (0-135)
        :discarded tiles : table.players[0].discards, list of `Tile`
                           Tile has attribute:
                               value: tile number (0-135)
        :dora indicators : table.dora_indicators, list of tile number
        :        is dora : table.is_dora(tile number), tile number (0-135)
        :    closed hand*: table.players[0].closed_hand, list of tile number (0-135)
        
        *Note: these info (tiles of player, closed hand) is only visible to self 
               player, we can not see hands of other players anyway.
        """
        dora_tiles = [d for d in range(136) if table.is_dora(d) ]
        table_turns = min([len(table.players[m].discards)+1 for m in range(4)])
        table_info = "{},{},{},{},{},{},{},{},{},{}".format(
                      table.count_of_honba_sticks,
                      table.count_of_remaining_tiles,
                      table.count_of_riichi_sticks,
                      table.round_number,
                      table.round_wind,
                      table_turns,
                      table.dealer_seat,
                      table.dora_indicators,
                      dora_tiles, # 0-136
                      table.revealed_tiles                    
                      ) 
       
        player_info = ""     
        for m in range(4): # There are four players
            player = table.players[m]
            
            '''
            player.discards
            player.closed_hand
            player.dealer_seat
            player.in_riichi
            #player.in_defence_mode
            #player.in_tempai
            player.is_dealer
            player.is_open_hand
            player.last_draw
            player.melds
            player.name
            player.position
            player.rank
            player.scores
            player.seat
            player.tiles
            player.uma
            '''
            
            # Discarded tiles can be seen by everybody
            discarded_tiles = [(d.value,1) if d.is_tsumogiri else (d.value,0) for d in player.discards]
            discarded_kinds = [d[0]//4 for d in discarded_tiles]
                      
            # If we need one more tile to complete our hand, and this specific tile
            # we want is known to be within the wall tiles, then the hand is waiting.
            current_hand = TilesConverter.to_34_array(player.tiles)
                       
            winning_tiles = []
            for n in range(34):
                # if there is no tile available in the wall, or the tile we need
                # to complete a hand has been discarded previously, we do not wait
                # for this tile. But we can still wait for tsumo??
                if (table.revealed_tiles[n]<4) and (n not in set(discarded_kinds)):
                    completed_hand = current_hand[:]
                    completed_hand[n] += 1
                    can_be_waiting = self.agari.is_agari(completed_hand)
                    if can_be_waiting:
                        winning_tiles.append(n) # n is the winning tile we want
                        
#            hand = [8, 11, 43, 44, 48, 51, 58, 79, 82, 87, 88, 92, 98] #+ [55]
#            hand34 = TilesConverter.to_34_array(hand)
#            if current_hand==hand34:
#                print("player {} tiles: {}".format(m, player.tiles))
#                print("player {} winning tiles: {}".format(m, winning_tiles))
            
            # Meld sets (both open and concealed) are also visible to everybody
            meld_sets = [mt.tiles for mt in player.melds]
            meld_open = [mt.opened for mt in player.melds]
            if meld_sets != self.meld_sets[m]:
                self.meld_discarded_tiles[m].append(discarded_tiles[-1])
                self.meld_sets[m] = meld_sets
            # Example: melds=[([12, 18, 20], 1, (108, 0)), ([40, 45, 49], 1, (7, 0))]
            melds = [(meld_sets[k], 1 if meld_open[k] else 0, self.meld_discarded_tiles[m][k]) for k in range(len(meld_sets))]
  
            string_to_save = "{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                    winning_tiles,                  
                    discarded_tiles, #player.discards
                    #player.closed_hand, # this info is invisible
                    player.dealer_seat,
                    1 if player.in_riichi else 0,
                    #player.in_defence_mode
                    #player.in_tempai
                    1 if player.is_dealer else 0,
                    1 if player.is_open_hand else 0,
                    #player.last_draw, # this info is invisible
                    melds,
                    player.name if player.name else -1,
                    player.position if player.position else -1,
                    player.rank if player.rank else -1,
                    player.scores if player.scores else -1,
                    player.seat if player.seat else -1,
                    #player.tiles, # this info is invisible
                    player.uma if player.uma else -1
            )
        
            player_info += string_to_save + ";"
        
        if player_info=="":
            return None
        else:
            return table_info + ";" + player_info