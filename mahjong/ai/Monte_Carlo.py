# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:54:32 2017

@author: joseph.chen
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:20:24 2017

@author: joseph.chen
"""
#import os
#import errno
import pickle
import numpy as np
#import scipy as sp
import random
import logging


from train_model.train_model import gen_one_player_features
from train_model.utils.tile import TilesConverter
from mahjong.utils import find_isolated_tile_indices
from mahjong.ai.agari import Agari as OldVersionAgari
from mahjong.ai.simple_oneplayer_model import MainAI as OnePlayerAI
#from mahjong.player import VisiblePlayer as Player
from mahjong.table import Table
from mahjong.meld import Meld
from mahjong.hand import FinishedHand
from tenhou.decoder import TenhouDecoder
#from mahjong.constants import DISPLAY_WINDS
#from mahjong.table import Table
from mahjong.player import VisiblePlayer
from mahjong.tile import Tile

from config.config import abs_data_path


# [Ref] http://kenby.iteye.com/blog/1162698
logger = logging.getLogger('MClogger')


fh = logging.FileHandler('MClogger.log')
fh.setLevel(logging.DEBUG)

#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
#fh.setFormatter(formatter)
#ch.setFormatter(formatter)

logger.addHandler(fh)
#logger.addHandler(ch)

class Agari(OldVersionAgari):

    def is_agari(self, tiles, melds=None):
        """
        Determine was it win or not
        :param tiles: 34 tiles format array
        :param melds: array of array of 34 tiles format
        :return: boolean
        """
        # we will modify them later, so we need to use a copy
        #tiles = copy.deepcopy(tiles)
        
        # TODO: I am not sure whether it's a bug or not. Originally the author
        # only took care of a hand of 14 tiles. However, when the closed hand
        # is consist of kans, there will be 15 or even more tiles. For this 
        # reason, I intentionally remove one tile from any kan here. This might
        # be a temporally solution. (joseph)
        if sum(tiles)>14:
            tiles = [t-1 if t==4 else t for t in tiles] # t==4 means a kan.

        # With open hand we need to remove open sets from hand and replace them with isolated pon sets
        # it will allow to determine agari state correctly
        if melds:
            isolated_tiles = find_isolated_tile_indices(tiles)
            for meld in melds:
                if not isolated_tiles:
                    break

                isolated_tile = isolated_tiles.pop()

                tiles[meld[0]] -= 1
                tiles[meld[1]] -= 1
                tiles[meld[2]] -= 1
                tiles[isolated_tile] = 3

        j = (1 << tiles[27]) | (1 << tiles[28]) | (1 << tiles[29]) | (1 << tiles[30]) | \
            (1 << tiles[31]) | (1 << tiles[32]) | (1 << tiles[33])

        if j >= 0x10:
            return False

        # 13 orphans
        if ((j & 3) == 2) and (tiles[0] * tiles[8] * tiles[9] * tiles[17] * tiles[18] *
                               tiles[26] * tiles[27] * tiles[28] * tiles[29] * tiles[30] *
                               tiles[31] * tiles[32] * tiles[33] == 2):
            return True

        # seven pairs
        if not (j & 10) and sum([tiles[i] == 2 for i in range(0, 34)]) == 7:
            return True

        if j & 2:
            return False

        n00 = tiles[0] + tiles[3] + tiles[6]
        n01 = tiles[1] + tiles[4] + tiles[7]
        n02 = tiles[2] + tiles[5] + tiles[8]

        n10 = tiles[9] + tiles[12] + tiles[15]
        n11 = tiles[10] + tiles[13] + tiles[16]
        n12 = tiles[11] + tiles[14] + tiles[17]

        n20 = tiles[18] + tiles[21] + tiles[24]
        n21 = tiles[19] + tiles[22] + tiles[25]
        n22 = tiles[20] + tiles[23] + tiles[26]

        n0 = (n00 + n01 + n02) % 3
        #print("n0={}".format(n0))
        if n0 == 1:
            return False

        n1 = (n10 + n11 + n12) % 3
        #print("n1={}".format(n1))
        if n1 == 1:
            return False

        n2 = (n20 + n21 + n22) % 3
        #print("n2={}".format(n2))
        if n2 == 1:
            return False


        if ((n0 == 2) + (n1 == 2) + (n2 == 2) + (tiles[27] == 2) + (tiles[28] == 2) +
                (tiles[29] == 2) + (tiles[30] == 2) + (tiles[31] == 2) + (tiles[32] == 2) +
                (tiles[33] == 2) != 1):
            return False

        nn0 = (n00 * 1 + n01 * 2) % 3
        m0 = self._to_meld(tiles, 0)
        nn1 = (n10 * 1 + n11 * 2) % 3
        m1 = self._to_meld(tiles, 9)
        nn2 = (n20 * 1 + n21 * 2) % 3
        m2 = self._to_meld(tiles, 18)

        if j & 4:
            return not (n0 | nn0 | n1 | nn1 | n2 | nn2) and self._is_mentsu(m0) \
                   and self._is_mentsu(m1) and self._is_mentsu(m2)

        if n0 == 2:
            return not (n1 | nn1 | n2 | nn2) and self._is_mentsu(m1) and self._is_mentsu(m2) \
                   and self._is_atama_mentsu(nn0, m0)

        if n1 == 2:
            return not (n2 | nn2 | n0 | nn0) and self._is_mentsu(m2) and self._is_mentsu(m0) \
                   and self._is_atama_mentsu(nn1, m1)

        if n2 == 2:
            return not (n0 | nn0 | n1 | nn1) and self._is_mentsu(m0) and self._is_mentsu(m1) \
                   and self._is_atama_mentsu(nn2, m2)

        return False
        
class MCVisibleTable(Table):
    """Visible table makes all players visible.
    """
    remaining_tiles = None
    _dead_wall = None
    _turn_number = None
    
    def _init_players(self,):
        self.player = MCVisiblePlayer(self, 0, self.dealer_seat, self.previous_ai)
        self.players = [self.player]
        for seat in range(1, self.count_of_players):
            player = MCVisiblePlayer(self, seat, self.dealer_seat, self.previous_ai)
            self.players.append(player)
            
    def erase_state(self):
        for player in self.players:
            player.erase_state()
        self.remaining_tiles = []
        self.dead_wall = []
        self.turn_number = [0,0,0,0]
            
    def add_discarded_tile(self, player_seat, tile, is_tsumogiri, drawn_tile=None):
        """
        :param player_seat:
        :param tile: 136 format tile
        :param is_tsumogiri: was tile discarded from hand or not
        :param drawn_tile: tile drawn from table
        """
        logger.debug("  check 1: %s, %s"%(self.count_of_remaining_tiles, len(self.remaining_tiles)))
        self.count_of_remaining_tiles -= 1
        if drawn_tile is not None:
            self.remaining_tiles.remove(drawn_tile) # add this info
        
        logger.debug("  add_discarded_tile:%s,%s,%s,%s"%(player_seat, tile, is_tsumogiri, drawn_tile))
        logger.debug("  check 2: %s, %s"%(self.count_of_remaining_tiles, len(self.remaining_tiles)))
        
        tile = Tile(tile, is_tsumogiri)
        self.get_player(player_seat).add_discarded_tile(tile)

        # cache already revealed tiles
        self._add_revealed_tile(tile.value)
        
    def add_called_meld(self, player_seat, meld):
#        # when opponent called meld it is means
#        # that he discards tile from hand, not from wall
#        self.count_of_remaining_tiles += 1
#
#        # we will decrease count of remaining tiles after called kan
#        # because we had to complement dead wall
#        if meld.type == Meld.KAN or meld.type == meld.CHANKAN:
#            self.count_of_remaining_tiles -= 1

        self.get_player(player_seat).add_called_meld(meld)

        tiles = meld.tiles[:]
        # called tile was already added to revealed array
        # because it was called on the discard
        if meld.called_tile:
            tiles.remove(meld.called_tile)

        # for chankan we already added 3 tiles
        if meld.type == meld.CHANKAN:
            tiles = [meld.tiles[0]]

        for tile in tiles:
            self._add_revealed_tile(tile)
        
    @property    
    def dead_wall(self):
        return self._dead_wall
    
    @dead_wall.setter
    def dead_wall(self, value):
        if not isinstance(value, list):
            raise ValueError('dead_wall must be an list!')
        self._dead_wall = value
    
    @property    
    def turn_number(self):
        return self._turn_number
    
    @turn_number.setter
    def turn_number(self, value):
        if type(value) is not list:
            raise TypeError('turn_number must be list of int!') 
        elif len(value)!=4:
            raise ValueError('turn_number must be list of length 4!')
        elif any(value)>19 or any(value)<0:
            raise ValueError('turn_number must be within [0,19]')
        self._turn_number = value
    
            
 
class MCOnePlayerAI(OnePlayerAI):
    def discard_tile(self, tiles, closed_hand, open_sets_34, table_revealed_tiles_34):   
        """
        :param tiles: array of tiles of player (in 136 format)
        :param closed_hand: array of tiles in 136 format
        :param open_sets_34: array of array with tiles in 34 format
        :param table_revealed_tiles_34: array of tiles in 34 format
        :return:
        """
        
        scores = []
        for tile in tiles: # self.player.tiles:
            # Probability to discard this tile
            discard_prob, discard_or_not = self.prob_to_discard(tile//4, 
                                                                tiles, 
                                                                table_revealed_tiles_34)
            scores.append((tile, discard_prob, discard_or_not))
            
        # sort by the discarding probablities (from highest to lowest)
        scores.sort(key=lambda tup:tup[1], reverse=True)
        
        # choose tile to discard
        tile_to_discard = None
        for score in scores:
            tile = score[0]
            if tile in closed_hand:
                tile_to_discard = tile
                break
        return tile_to_discard
    
    def prob_to_discard(self, tile, player_tiles, table_revealed_tiles_34):
        """The probability of discarding a tile based on one player mahjong model.
        :param tile: int (0-33), index of the tile kind 
        :param player_tiles: list of int (0-135), index of player's tiles
        :param table_revealed_tiles_34: list of int (0-4), tile occurance in 34 format
        return: probability of discarding a tile;
                discard the tile or not (1 or 0).
        """
        features = gen_one_player_features(player_tiles,
                                           table_revealed_tiles_34)
        
        f1, f2 = features
        one_player_info = f1+f2
        one_player_info = np.array([one_player_info])
        
        # Probability of `p` is waiting
        clf = self.clf_one_player[tile] # load (tile-th) classifier
        prob_to_discard_tile = clf.predict_proba(one_player_info)[0][1]     
        discard_or_not = clf.predict(one_player_info)[0]
        return prob_to_discard_tile, discard_or_not
    
class MCVisiblePlayer(VisiblePlayer):
    def _load_ai(self):
        # TODO: I don't quite understand why we put a self inside the parentheses
        # here.
        self.ai = MCOnePlayerAI(self)
           
class MonteCarlo(object):
    """A simulator to run Monte Carlo simulation on Mahjong game
    """
    decoder = TenhouDecoder()
    agari = Agari()
    finished_hand = FinishedHand() 
    verbose = False
    
    def __init__(self):
        
        previous_ai = False
        table = MCVisibleTable(previous_ai)       
        self.table = table
        
        self.initialize()
        
        # We load the classifiers and regressors here
        self.clf_one_player = []
        for n in range(34):
            clf = pickle.load(open(abs_data_path+"/train_model/trained_models/one_player_{}.sav".format(n), "rb"))
            self.clf_one_player.append(clf)
    
    def initialize(self): 
        # Mahjong table is ready
#        previous_ai = False
#        table = MCVisibleTable(previous_ai)       
#        self.table = table
        self.table.erase_state()
        self.table.turn_number = [0,0,0,0]
        
        # Prepare a new deck of mahjong tiles
        tiles = list(range(136))
        # shuffle the tiles
        random.shuffle(tiles)
        # seperate dead wall and dora indicator
        dead_wall = tiles[-14:]
        dora_indicator = dead_wall[4] # random.choice(dead_wall)
        self.table.dead_wall = dead_wall
        # generate init hands for 4 players
        tiles = tiles[0:-14]
        hands = []
        for n in range(4):
            init_tiles = tiles[0:13]
            hands.append(init_tiles)
            tiles = tiles[13:]
            
        self.table.remaining_tiles = tiles
            
        # formulate the init message
        dice1 = random.choice([1,2,3,4,5,6])
        dice2 = random.choice([1,2,3,4,5,6])
        dealer = (dice1-1+dice2-1)%4
        message = "<INIT "
        # TODO: You may need round number, number of combo sticks, and number of
        # riichi sticks other than 0?
        message += 'seed="{},{},{},{},{},{}" '.format(0,0,0,dice1-1,dice2-1,dora_indicator)
        message += 'ten="250,250,250,250" '
        message += 'oya="{}" '.format(dealer)
        message += 'hai0="{}" '.format(",".join(str(t) for t in hands[0]))
        message += 'hai1="{}" '.format(",".join(str(t) for t in hands[1]))
        message += 'hai2="{}" '.format(",".join(str(t) for t in hands[2]))
        message += 'hai3="{}"/>'.format(",".join(str(t) for t in hands[3]))

        # once message is ready, we decoder the info and initialize the table                       
        values = self.decoder.parse_initial_values(message)
        self.table.init_round(
            values['round_number'],
            values['count_of_honba_sticks'],
            values['count_of_riichi_sticks'],
            values['dora_indicator'],
            values['dealer'],
            values['scores'],
        )       
        # TODO: this part is unnecessary, since we have already the hands list
        hands = [
            [int(x) for x in self.decoder.get_attribute_content(message, 'hai0').split(',')],
            [int(x) for x in self.decoder.get_attribute_content(message, 'hai1').split(',')],
            [int(x) for x in self.decoder.get_attribute_content(message, 'hai2').split(',')],
            [int(x) for x in self.decoder.get_attribute_content(message, 'hai3').split(',')],
        ]       
        # Initialize all players on the table
        # TODO: ok, we always assume we are sitting at seat 0
        self.player_position = 0
        self.table.players[0].init_hand(hands[self.player_position])
        self.table.players[1].init_hand(hands[(self.player_position+1)%4])
        self.table.players[2].init_hand(hands[(self.player_position+2)%4])
        self.table.players[3].init_hand(hands[(self.player_position+3)%4])
#        main_player = self.table.player
#        print(self.table.__str__())
#        print('Players: {}'.format(self.table.get_players_sorted_by_scores()))
#        print('Dealer: {}'.format(self.table.get_player(values['dealer'])))
#        print('Round  wind: {}'.format(DISPLAY_WINDS[self.table.round_wind]))
#        print('Player wind: {}'.format(DISPLAY_WINDS[main_player.player_wind]))

        # TODO: this part may not be necessary. Basically we need to erase the 
        # melds information since it is a new game.
        # If we are using opponent_model, we need to reset the melds
        # when a new game initiated. For other model, this function
        # may not exist. [Joseph]
        try:
            self.table.player.ai.reset_melds()
        except:
            pass
     
    def _check_win(self, player_hand, melds):
        """ check the hand to see whether it's a win hand or not
        :param player_hand: list of int (0-33), the player's hand tiles in 34 format
        :return: True if win, else False.
        """
        return self.agari.is_agari(player_hand, melds)  
    
    def check_win(self, p):
        """check win hand
        :param p: int (0-3), player index
        :return: True if win, False if not.
        """
        player_hand = self.table.players[p].tiles
        player_hand_34 = TilesConverter.to_34_array(player_hand)
        melds = [meld.tiles for meld in self.table.players[p].melds] # self.table.players[p].melds is [list of Meld()]
        melds_34 = list(map(lambda lst:[l//4 for l in lst], melds))
        return self._check_win(player_hand_34, melds_34)
    
    def _check_waiting(self, player_hand):
        """Check whether a player is waiting or not based on his hand
        :param player_hand: list of elements in [0,133], the hand tiles of player
        :return: True for waiting, False for not waiting
        """
        # If we need one more tile to complete our hand, and this specific tile
        # we want is known to be within the wall tiles, then the hand is waiting.
        current_hand = TilesConverter.to_34_array(player_hand)
        #winning_tiles = []
        for n in range(34):
            table_revealed_tiles_34 = self.table.revealed_tiles
            if table_revealed_tiles_34[n]<4:
                completed_hand = current_hand[:]
                completed_hand[n] += 1
                can_be_waiting = self.agari.is_agari(completed_hand)
                if can_be_waiting:
                    return True
                    #winning_tiles.append(n) # n is the winning tile we want
        
        # If there exists winnint tiles, the player is waitinng
        #if len(winning_tiles)>0:
        #    return True
        return False
    
    def check_waiting(self, p):
        """check whether the player `p` is waiting or not
        :param p: int (0-3), player index
        :return: True for waiting, False for not waiting
        """
        player_hand = self.table.players[p].tiles
        return self._check_waiting(player_hand)
    
    def discard_tile(self, p):
        """choose a tile to discard based on hand tiles of player `p`
        :param p: int(0-3), player index
        :return: int(0-135), tile to discard
        """
        tiles = self.table.players[p].tiles
        closed_hand = self.table.players[p].closed_hand
        open_sets_34 = self.table.players[p].open_hand_34_tiles
        table_revealed_tiles_34 = self.table.revealed_tiles
        discarded_tile = self.table.players[p].ai.discard_tile(tiles, 
                                                               closed_hand, 
                                                               open_sets_34,
                                                               table_revealed_tiles_34)
        return discarded_tile
    
    def call_meld(self, type, who, from_who, opened, tiles, called_tile):
        meld = Meld()
        meld.type = type
        meld.who = who
        meld.from_who = from_who
        meld.opened = opened
        meld.tiles = tiles
        meld.called_tile = called_tile
        return meld
    
    def sim_WPET(self):
        """Simulation of a riichi mahjong game, and record the bw and r parameters
        """
        # We don't need to fix the seed now.
        #seed = random.randint(1,100000)
        #seed = 73823
        #random.seed(seed)
        
        bw_riichi = 0
        bw_stealing = 0
        r = 0
        
        # start a new game
        self.initialize()
        # rinshan 
        rinshan = self.table.dead_wall[0:4]
        # start from dealer
        p = self.table.dealer_seat
        #logger.info("seed: %d"%seed)
        logger.info("dealer seat: %d"%p)
        # Fixed: self.table.count_of_remaining_tiles>0 may not be accurate here
        while self.table.count_of_remaining_tiles>0: #len(self.table.remaining_tiles)>0:
            
            # our perspective of view is from the program's turn
            if p==0:
                r += 1
                
            if self.check_waiting(0):
                # TODO: Why don't we use in_riichi here? 
                if self.table.players[0].is_open_hand:
                    bw_stealing = 1
                else:
                    bw_riichi = 1
                break
            
            logger.debug("[begin]:%s,%s"%(self.table.count_of_remaining_tiles, len(self.table.remaining_tiles)))
            # p draw a tile
            drawn_tile = self.table.remaining_tiles[0]
            self.table.players[p].draw_tile(drawn_tile) 
            # check p win
            if self.check_win(p):
                logger.info("player {} wins (tsumo)!".format(p))
                break
            
            # TODO: we only apply discard_tile strategy to our program, for the
            # opponent players, we assume they simply discard what they draw
            # if not win, we do the followings
            if True: #p==0:
                if not self.table.players[p].in_riichi:                  
                    # choose a discard tile for playe `p`
                    discarded_tile = self.discard_tile(p)                   
                    # see if we can call riichi                  
                    if self.table.players[p].can_call_riichi():  
                        logger.debug("player {} call riichi.".format(p))
                        self.table.players[p].in_riichi = True                                         
                else: # if riichi, we have to discard whatever we draw
                    discarded_tile = drawn_tile
            else:
                discarded_tile = drawn_tile
                
            # remove the tile from player's hand
            discarded_tile_tmp = discarded_tile
            drawn_tile_tmp = drawn_tile
            is_tsumogiri = (discarded_tile==drawn_tile)
            self.table.players[p].tiles.remove(discarded_tile)
            logger.debug("player {} discards {}".format(p, discarded_tile))
            logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
            logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
            logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
            # now we check call meld 
            # TODO: Ok, here we only allow our program to call meld. But maybe
            # we should allow the opponents to call meld too?
            if p!=0:
                previous_drawn_tile = drawn_tile
                tile = discarded_tile
                is_kamicha_discard = (p==3)
                meld, discard_option = self.table.players[0].try_to_call_meld(tile, is_kamicha_discard)
                kan_type = self.table.players[0].can_call_kan(tile, True)
                if kan_type: # kan or chankan
                    tiles = [(tile//4)*4,
                             (tile//4)*4+1,
                             (tile//4)*4+2,
                             (tile//4)*4+3]
                    meld = self.call_meld(kan_type, 0, p, True, tiles, tile)
                    logger.debug("player 0 call kan from %d: %s"%(p,meld))
                    player_seat = meld.who 
                    self.table.add_called_meld(player_seat, meld)
                    # we had to delete called tile from hand
                    # to have correct tiles count in the hand
                    # TODO[joseph]: I still don't know why we need this
                    # if meld type is: chi, pon, or nuki
                    # maybe no use here, b/c the meld type is always Meld.KAN here
                    if meld.type != Meld.KAN and meld.type != Meld.CHANKAN:
                        self.table.players[player_seat].draw_tile(meld.called_tile)
                    # draw a tile from dead wall
                    if len(rinshan)>0:
                        drawn_tile = rinshan.pop(0)
                        self.table.players[0].draw_tile(drawn_tile) 
                        # check p win
                        if self.check_win(0):
                            logger.info("player {} wins (rinshan tsumo)!".format(0))
                            break
                        # if not win, we do the followings
                        if not self.table.players[0].in_riichi:
                            # choose a discard tile for playe `p`
                            discarded_tile = self.discard_tile(0)                           
                            # see if we can call riichi                  
                            if self.table.players[0].can_call_riichi():  
                                logger.info("player {} call riichi.".format(0))
                                self.table.players[0].in_riichi = True                                                 
                        else: # if riichi, we have to discard whatever we draw
                            discarded_tile = drawn_tile   
                        # remove the tile from player's hand
                        self.table.players[0].tiles.remove(discarded_tile)
                        logger.debug("player {} discards {} after kan".format(0, discarded_tile))
                        logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
                        logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
                        logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
                        # we had to add it to discards, to calculate remaining tiles correctly
                        # drawn tile is not the one drawn from rinshan, but 
                        # the one previously discarded by player `p`
                        self.table.add_discarded_tile(0, discarded_tile, True, previous_drawn_tile)
                        # after program discarding a card, next player is 1
                        p = 1
                        continue
                    else: # ryuukyoku
                        logger.debug("Rinshan empty. Ryuukyoku!")
                        break
                elif meld: # pon, chi
                    logger.debug("player 0 %s from %d: %s"%(meld.type,p,meld))
                    player_seat = 0    
                    # DEBUG: we change the add_called_meld method, delete the 
                    # part that changes self.table.count_of_remaining_tiles
                    self.table.add_called_meld(player_seat, meld)                                               
                    # Equivalently, program draws the tile discarded by opponent
                    self.table.players[0].draw_tile(tile)
                    # check p win
                    if self.check_win(0):
                        logger.info("player {} wins (by {})!".format(0, meld.type))
                        break
                    # if not win, we do the followings
                    if not self.table.players[0].in_riichi:
                        # choose a discard tile for playe `p`
                        discarded_tile = self.discard_tile(0)                           
                        # see if we can call riichi                  
                        if self.table.players[0].can_call_riichi():  
                            logger.debug("player {} call riichi.".format(0))
                            self.table.players[0].in_riichi = True                                                 
                    else: # if riichi, we can not call meld
                        raise("Riichi player can not call meld!")
                    # remove the tile from player's hand
                    self.table.players[0].tiles.remove(discarded_tile)  
                    # discarded tile added to table
                    self.table.add_discarded_tile(0, discarded_tile, True, previous_drawn_tile)
                    logger.debug("player {} discards {} after {}".format(0, discarded_tile, meld.type))
                    logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
                    logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
                    logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
                    # after program discarding a card, next player is 1
                    p = 1
                    continue
            # we had to add it to discards, to calculate remaining tiles correctly
            self.table.add_discarded_tile(p, discarded_tile_tmp, is_tsumogiri, drawn_tile_tmp)
                           
            # next player
            p = (p+1)%4

            logger.debug("[after]:%s,%s"%(self.table.count_of_remaining_tiles, len(self.table.remaining_tiles)))
                           
        # output results
        logger.debug('\n')
        for p in range(4):
            logger.info("\tPlayer %d: %s (%s)"%(p, 
                TilesConverter.to_one_line_string(self.table.players[p].tiles),
                TilesConverter.to_one_line_string(self.table.players[p].closed_hand))
            )  
        
        return bw_riichi, bw_stealing, r

    def get_WPET(self, Nsim):
        """Waiting probability at Each Turn
        """
        R = [0]*19
        BW_riichi = [0]*19
        BW_stealing = [0]*19
        n = 0
        while n<Nsim:
            bw_riichi, bw_stealing, r = self.sim_WPET()
            print("%d. bw_riichi=%s, bw_stealing=%s, r=%s"%(n+1, bw_riichi, bw_stealing, r))
            for m in range(r+1): # from 0 to r
                R[m] += 1
            BW_riichi[r] += bw_riichi
            BW_stealing[r] += bw_stealing
            n += 1
        Wpet_riichi = [BW_riichi[n]*1.0/R[n] for n in range(len(BW_riichi))]
        Wpet_stealing = [BW_stealing[n]*1.0/R[n] for n in range(len(BW_stealing))]
        return BW_riichi, BW_stealing, R, Wpet_riichi, Wpet_stealing
    
    def sim_game(self):
        """Simulation of a riichi mahjong game, and record the bw and r parameters
        """
        # We don't need to fix the seed now.
        #seed = random.randint(1,100000)
        #seed = 73823
        #random.seed(seed)
       
        # start a new game
        self.initialize()
        # rinshan 
        rinshan = self.table.dead_wall[0:4]
        # start from dealer
        p = self.table.dealer_seat
        #logger.info("seed: %d"%seed)
        logger.info("dealer seat: %d"%p)
        # Fixed: self.table.count_of_remaining_tiles>0 may not be accurate here
        while self.table.count_of_remaining_tiles>0: #len(self.table.remaining_tiles)>0:
    
            logger.debug("[begin]:%s,%s"%(self.table.count_of_remaining_tiles, len(self.table.remaining_tiles)))
            
            # update turn number
            self.table.turn_number[p] += 1
            
            # p draw a tile
            drawn_tile = self.table.remaining_tiles[0]
            self.table.players[p].draw_tile(drawn_tile) 
            # check p win
            if self.check_win(p):
                logger.info("player {} wins (tsumo)!".format(p))
                tiles = self.table.players[p].tiles
                win_tile = drawn_tile
                (is_tsumo,
                is_riichi,
                is_dealer,
                open_sets,
                dora_indicators,
                player_wind,
                round_wind) = self.check_status(p)
                ## TODO: Wait to be finished!
                result = self.finished_hand.estimate_hand_value(
                            tiles,
                            win_tile,
                            is_tsumo=is_tsumo,
                            is_riichi=is_riichi,
                            is_dealer=is_dealer,
                            is_ippatsu=False,
                            is_rinshan=False,
                            is_chankan=False,
                            is_haitei=False,
                            is_houtei=False,
                            is_daburu_riichi=False,
                            is_nagashi_mangan=False,
                            is_tenhou=False,
                            is_renhou=False,
                            is_chiihou=False,
                            open_sets=open_sets,
                            dora_indicators=dora_indicators,
                            called_kan_indices=None,
                            player_wind=player_wind,
                            round_wind=round_wind)
                logger.info(result)
                break
            
            # TODO: we only apply discard_tile strategy to our program, for the
            # opponent players, we assume they simply discard what they draw
            # if not win, we do the followings
            if True: #p==0:
                if not self.table.players[p].in_riichi:                  
                    # choose a discard tile for playe `p`
                    discarded_tile = self.discard_tile(p)                   
                    # see if we can call riichi                  
                    if self.table.players[p].can_call_riichi():  
                        logger.debug("player {} call riichi.".format(p))
                        self.table.players[p].in_riichi = True                                         
                else: # if riichi, we have to discard whatever we draw
                    discarded_tile = drawn_tile
            else:
                discarded_tile = drawn_tile
                
            # remove the tile from player's hand
            discarded_tile_tmp = discarded_tile
            drawn_tile_tmp = drawn_tile
            is_tsumogiri = (discarded_tile==drawn_tile)
            self.table.players[p].tiles.remove(discarded_tile)
            logger.debug("player {} discards {}".format(p, discarded_tile))
            logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
            logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
            logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
            # now we check call meld 
            # TODO: Ok, here we only allow our program to call meld. But maybe
            # we should allow the opponents to call meld too?
            if p!=0:
                previous_drawn_tile = drawn_tile
                tile = discarded_tile
                is_kamicha_discard = (p==3)
                meld, discard_option = self.table.players[0].try_to_call_meld(tile, is_kamicha_discard)
                kan_type = self.table.players[0].can_call_kan(tile, True)
                if kan_type: # kan or chankan
                    tiles = [(tile//4)*4,
                             (tile//4)*4+1,
                             (tile//4)*4+2,
                             (tile//4)*4+3]
                    meld = self.call_meld(kan_type, 0, p, True, tiles, tile)
                    logger.debug("player 0 call kan from %d: %s"%(p,meld))
                    player_seat = meld.who 
                    self.table.add_called_meld(player_seat, meld)
                    # we had to delete called tile from hand
                    # to have correct tiles count in the hand
                    # TODO[joseph]: I still don't know why we need this
                    # if meld type is: chi, pon, or nuki
                    # maybe no use here, b/c the meld type is always Meld.KAN here
                    if meld.type != Meld.KAN and meld.type != Meld.CHANKAN:
                        self.table.players[player_seat].draw_tile(meld.called_tile)
                    # draw a tile from dead wall
                    if len(rinshan)>0:
                        drawn_tile = rinshan.pop(0)
                        self.table.players[0].draw_tile(drawn_tile) 
                        # check p win
                        if self.check_win(0):
                            logger.info("player {} wins (rinshan tsumo)!".format(0))
                            tiles = self.table.players[p].tiles
                            win_tile = drawn_tile
                            (is_tsumo,
                            is_riichi,
                            is_dealer,
                            open_sets,
                            dora_indicators,
                            player_wind,
                            round_wind) = self.check_status(p)
                            is_rinshan = True
                            ## TODO: Wait to be finished!
                            result = self.finished_hand.estimate_hand_value(
                                        tiles,
                                        win_tile,
                                        is_tsumo=is_tsumo,
                                        is_riichi=is_riichi,
                                        is_dealer=is_dealer,
                                        is_ippatsu=False,
                                        is_rinshan=is_rinshan,
                                        is_chankan=False,
                                        is_haitei=False,
                                        is_houtei=False,
                                        is_daburu_riichi=False,
                                        is_nagashi_mangan=False,
                                        is_tenhou=False,
                                        is_renhou=False,
                                        is_chiihou=False,
                                        open_sets=open_sets,
                                        dora_indicators=dora_indicators,
                                        called_kan_indices=None,
                                        player_wind=player_wind,
                                        round_wind=round_wind)
                            logger.info(result)
                            break
                        # if not win, we do the followings
                        if not self.table.players[0].in_riichi:
                            # choose a discard tile for playe `p`
                            discarded_tile = self.discard_tile(0)                           
                            # see if we can call riichi                  
                            if self.table.players[0].can_call_riichi():  
                                logger.info("player {} call riichi.".format(0))
                                self.table.players[0].in_riichi = True                                                 
                        else: # if riichi, we have to discard whatever we draw
                            discarded_tile = drawn_tile   
                        # remove the tile from player's hand
                        self.table.players[0].tiles.remove(discarded_tile)
                        logger.debug("player {} discards {} after kan".format(0, discarded_tile))
                        logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
                        logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
                        logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
                        # we had to add it to discards, to calculate remaining tiles correctly
                        # drawn tile is not the one drawn from rinshan, but 
                        # the one previously discarded by player `p`
                        self.table.add_discarded_tile(0, discarded_tile, True, previous_drawn_tile)
                        # after program discarding a card, next player is 1
                        p = 1
                        continue
                    else: # ryuukyoku
                        logger.debug("Rinshan empty. Ryuukyoku!")
                        break
                elif meld: # pon, chi
                    logger.debug("player 0 %s from %d: %s"%(meld.type,p,meld))
                    player_seat = 0    
                    # DEBUG: we change the add_called_meld method, delete the 
                    # part that changes self.table.count_of_remaining_tiles
                    self.table.add_called_meld(player_seat, meld)                                               
                    # Equivalently, program draws the tile discarded by opponent
                    self.table.players[0].draw_tile(tile)
                    # check p win
                    if self.check_win(0):
                        logger.info("player {} wins (by {})!".format(0, meld.type))
                        tiles = self.table.players[p].tiles
                        win_tile = tile
                        (is_tsumo,
                        is_riichi,
                        is_dealer,
                        open_sets,
                        dora_indicators,
                        player_wind,
                        round_wind) = self.check_status(p)
                        ## TODO: Wait to be finished!
                        result = self.finished_hand.estimate_hand_value(
                                    tiles,
                                    win_tile,
                                    is_tsumo=is_tsumo,
                                    is_riichi=is_riichi,
                                    is_dealer=is_dealer,
                                    is_ippatsu=False,
                                    is_rinshan=False,
                                    is_chankan=False,
                                    is_haitei=False,
                                    is_houtei=False,
                                    is_daburu_riichi=False,
                                    is_nagashi_mangan=False,
                                    is_tenhou=False,
                                    is_renhou=False,
                                    is_chiihou=False,
                                    open_sets=open_sets,
                                    dora_indicators=dora_indicators,
                                    called_kan_indices=None,
                                    player_wind=player_wind,
                                    round_wind=round_wind)
                        logger.info(result)
                        break
                    # if not win, we do the followings
                    if not self.table.players[0].in_riichi:
                        # choose a discard tile for playe `p`
                        discarded_tile = self.discard_tile(0)                           
                        # see if we can call riichi                  
                        if self.table.players[0].can_call_riichi():  
                            logger.debug("player {} call riichi.".format(0))
                            self.table.players[0].in_riichi = True                                                 
                    else: # if riichi, we can not call meld
                        raise("Riichi player can not call meld!")
                    # remove the tile from player's hand
                    self.table.players[0].tiles.remove(discarded_tile)  
                    # discarded tile added to table
                    self.table.add_discarded_tile(0, discarded_tile, True, previous_drawn_tile)
                    logger.debug("player {} discards {} after {}".format(0, discarded_tile, meld.type))
                    logger.debug("\tclosed hand: %s"%self.table.players[0].closed_hand)
                    logger.debug("\topen hand: %s"%self.table.players[0].open_hand_34_tiles)
                    logger.debug("\tmeld tiles: %s"%self.table.players[0].meld_tiles)
                    # after program discarding a card, next player is 1
                    p = 1
                    continue
            # we had to add it to discards, to calculate remaining tiles correctly
            self.table.add_discarded_tile(p, discarded_tile_tmp, is_tsumogiri, drawn_tile_tmp)
                           
            # next player
            p = (p+1)%4

            logger.debug("[after]:%s,%s"%(self.table.count_of_remaining_tiles, len(self.table.remaining_tiles)))
                           
        # output results
        logger.debug('\n')
        for p in range(4):
            logger.info("\tPlayer %d: %s (%s)"%(p, 
                TilesConverter.to_one_line_string(self.table.players[p].tiles),
                TilesConverter.to_one_line_string(self.table.players[p].closed_hand))
            )  
        
    ## TODO: finish this part! 
    def check_status(self, p, is_tsumo=False):
        """ We want to check the status of player `p`
        :param is_tsumo:
        :param is_riichi:
        :param is_dealer:
        :param is_ippatsu:
        :param is_rinshan:
        :param is_chankan:
        :param is_haitei:
        :param is_houtei:
        :param is_tenhou:
        :param is_renhou:
        :param is_chiihou:
        :param is_daburu_riichi:
        :param is_nagashi_mangan:
        :param open_sets: array of array with open sets in 34-tile format
        :param dora_indicators: array of tiles in 136-tile format
        :param called_kan_indices: array of tiles in 136-tile format
        :param player_wind: index of player wind
        :param round_wind: index of round wind
        """
        is_tsumo = is_tsumo
        is_riichi = self.table.players[p].in_riichi
        is_dealer = (self.table.dealer_seat==p)
        open_sets = [meld.tiles for meld in self.table.players[p].melds if meld.opened==True]
        open_sets = list(map(lambda lst:[l//4 for l in lst], open_sets)) # convert to 34 format
        dora_indicators = self.table.dora_indicators
        player_wind = self.table.players[p].player_wind
        round_wind = self.table.round_wind
        
        return (is_tsumo,
                is_riichi,
                is_dealer,
                open_sets,
                dora_indicators,
                player_wind,
                round_wind)
        
    def _check2(self):
        self.sim_game()
        
    def _check(self):
        BW_riichi, BW_stealing, R, Wpet_riichi, Wpet_stealing = self.get_WPET(500)
        print("BW riichi: %s"%BW_riichi)
        print("BW stealing: %s"%BW_stealing)
        print("R: %s"%R)   
        print("WPET riichi: %s"%Wpet_riichi)
        print("WPET stealing: %s"%Wpet_stealing)
        
        import matplotlib.pyplot as plt
        plt.plot(range(19),Wpet_riichi,'*-r',range(19),Wpet_stealing,'o-b')
        
if __name__=="__main__":
    print("OMG")
    MC = MonteCarlo()
    MC._check()
 