# -*- coding: utf-8 -*-
import random
import pickle
import ast
import numpy as np

from mahjong.ai.base import BaseAI
from mahjong.meld import Meld
from mahjong.tile import TilesConverter
from mahjong.utils import is_pair, is_pon
from mahjong.ai.strategies.main import BaseStrategy

from config.config import abs_data_path
from train_model.train_model import gen_is_waiting_features

# TODO: For testing purpose. You can delete this function if unused.
from train_model.train_model import load_sparse_csr

# TODO: For testing purpose. You can delete this function if unused.
#def load_sparse_csr(filename):
#    loader = np.load(filename)
#    return sp.sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
#                         shape = loader['shape'])
    
class DefenceHandler(object):
    # Original DefenceHandler written by the author can be found in mahjong.ai.defence.main.py
    def __init__(self, player):
        pass

    def should_go_to_defence_mode(self, discard_candidate=None):
        # We never switch to `defence` mode, since our algorithm has already 
        # considered defence through evaluating a `trade-off` value.
        return False

class MainAI(BaseAI):
    """
    AI that is based on Monte Carlo simulation and opponent model.
    """

    version = 'random'

    def __init__(self, player):
        super(MainAI, self).__init__(player)
        
        # we don't `defense` since our algorithm will defense by evaluating "trade-off" value.
        self.defence = DefenceHandler(player) 
        # strategy_type is set to None. We use this BaseStrategy to call meld.
        self.current_strategy = BaseStrategy(None, player)   
        
        # We load the classifiers and regressors here
        self.clf_is_waiting = pickle.load(open(abs_data_path+"/train_model/trained_models/is_waiting.sav", "rb"))
        self.clf_waiting_tile = []
        for n in range(34):
            clf = pickle.load(open(abs_data_path+"/train_model/trained_models/waiting_tile_{}.sav".format(n), "rb"))
            self.clf_waiting_tile.append(clf)
        self.rgrs_scores = pickle.load(open(abs_data_path+"/train_model/trained_models/scores.sav", "rb"))
        self.rgrs_wfw_scores = pickle.load(open(abs_data_path+"/train_model/trained_models/wfw_scores.sav", "rb"))
        
    def erase_state(self):
        self.current_strategy = None
        self.in_defence = False
        
        self.meld_sets = [[],[],[],[]] # four players
        self.meld_discarded_tiles = [[],[],[],[]] # four players

    def determine_strategy(self):
        return False
    
    def can_call_kan(self, tile, open_kan):
        """
        Method will decide should we call a kan,
        or upgrade pon to kan
        :param tile: 136 tile format
        :param open_kan: boolean
        :return: kan type
        """
        # we don't need to add dora for other players
        if self.player.ai.in_defence:
            return None

        if open_kan:
            # we don't want to start open our hand from called kan
            if not self.player.is_open_hand:
                return None

            # there is no sense to call open kan when we are not in tempai
            if not self.player.in_tempai:
                return None

            # we have a bad wait, rinshan chance is low
            if len(self.waiting) < 2:
                return None

        tile_34 = tile // 4
        tiles_34 = TilesConverter.to_34_array(self.player.tiles)
        closed_hand_34 = TilesConverter.to_34_array(self.player.closed_hand)
        pon_melds = [x for x in self.player.open_hand_34_tiles if is_pon(x)]

        # let's check can we upgrade opened pon to the kan
        if pon_melds:
            for meld in pon_melds:
                # tile is equal to our already opened pon,
                # so let's call chankan!
                if tile_34 in meld:
                    return Meld.CHANKAN

        count_of_needed_tiles = 4
        # for open kan 3 tiles is enough to call a kan
        if open_kan:
            count_of_needed_tiles = 3

        # we have 3 tiles in our hand,
        # so we can try to call closed meld
        if closed_hand_34[tile_34] == count_of_needed_tiles:
            if not open_kan:
                # to correctly count shanten in the hand
                # we had do subtract drown tile
                tiles_34[tile_34] -= 1

            melds = self.player.open_hand_34_tiles
            previous_shanten = self.shanten.calculate_shanten(tiles_34, melds)

            melds += [[tile_34, tile_34, tile_34]]
            new_shanten = self.shanten.calculate_shanten(tiles_34, melds)

            # called kan will not ruin our hand
            if new_shanten <= previous_shanten:
                return Meld.KAN

        return None
    
    def try_to_call_meld(self, tile, is_kamicha_discard):
        if not self.current_strategy:
            return None, None

        return self.current_strategy.try_to_call_meld(tile, is_kamicha_discard)
    
    def discard_tile(self):
        tile_to_discard = random.randrange(len(self.player.tiles) - 1)
        tile_to_discard = self.player.tiles[tile_to_discard]
        
        print("\n")
        for p in range(1,4):
            print("prob of losing to {}: {}".format(p, self.get_losing_probability(p, tile_to_discard)))
        print("opponnet model discards: {}\n".format(tile_to_discard))
        return tile_to_discard
    
    def prob_is_waiting(self, p):
        """The probability that an opponent p is waiting.
        param p: int (1-3), index of opponent player
        return: probability of opponent p being waiting
        """
        # Get table information
        table = self.player.table
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
        
        # Get player information
        player_info = ""
        player = table.players[p]
        # Discarded tiles can be seen by everybody
        discarded_tiles = [(d.value,1) if d.is_tsumogiri else (d.value,0) for d in player.discards]
        # winning tiles of opponents are invisible, we just keep an empty 34 element array here
        winning_tiles = [0 for _ in range(34)]       
        # Meld sets (both open and concealed) are also visible to everybody
        meld_sets = [mt.tiles for mt in player.melds]
        meld_open = [mt.opened for mt in player.melds]
        if meld_sets != self.meld_sets[p]:
            self.meld_discarded_tiles[p].append(discarded_tiles[-1])
            self.meld_sets[p] = meld_sets
        melds = [(meld_sets[k], 1 if meld_open[k] else 0, self.meld_discarded_tiles[p][k]) for k in range(len(meld_sets))]           
        string_to_save = "{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                winning_tiles,               
                discarded_tiles,                 
                player.dealer_seat,
                1 if player.in_riichi else 0,               
                1 if player.is_dealer else 0,
                1 if player.is_open_hand else 0,                
                melds,
                -1, # we don't need player's name; player.name if player.name else -1,
                player.position if player.position else -1,
                -1, # we don't need player's rank; player.rank if player.rank else -1,
                player.scores if player.scores else -1,
                player.seat if player.seat else -1,               
                player.uma if player.uma else -1
                #player.closed_hand,     # this info is invisible
                #player.in_defence_mode, # this info is invisible
                #player.in_tempai,       # this info is invisible
                #player.last_draw,       # this info is invisible
                #player.tiles,           # this info is invisible
        )   
        player_info += string_to_save 
            
        # Parse the table info
        (table_count_of_honba_sticks,
         table_count_of_remaining_tiles,
         table_count_of_riichi_sticks,
         table_round_number,
         table_round_wind,
         table_turns,
         table_dealer_seat,
         table_dora_indicators,
         table_dora_tiles,
         table_revealed_tiles) = ast.literal_eval(table_info)
        
        # Parse the player info
        (player_winning_tiles,                   
         player_discarded_tiles, 
         player_dealer_seat,
         player_in_riichi,
         player_is_dealer,
         player_is_open_hand,              
         player_melds,                
         player_name, # player_name has been replaced with -1
         player_position,
         player_rank, # player_rank has been replaced with -1
         player_scores,
         player_seat,
         player_uma) = ast.literal_eval(player_info)
        
        # replace back the player name, although we may not need it
        player_name = player.name
        # replace back the player rank, although we may not need it
        player_rank = player.rank
        
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
        opponent_info = [f1]+[f2]+[f3]+[f4]+[f5]+f6+f7+f8+f9
        opponent_info = np.array([opponent_info])
        
        # Probability of p is waiting
        clf = self.clf_is_waiting
        prob_is_waiting = clf.predict_proba(opponent_info)[0][1]     
        return prob_is_waiting
            
    def prob_winning_tile(self, p, tile):
        """The probability that an opponent `p` is waiting for `tile`.
        param p: int (1-3), index of opponent player
        param tile: int (0-33), index of the tile kind
        return: probability of tile being waiting tile for opponent p
        """
        
    
    def get_expected_loss(self):
        """Use our trained model to predict loss if discarding a winning tile to
        the opponent.
        """
        # Firstly, get table information
        
    
    
