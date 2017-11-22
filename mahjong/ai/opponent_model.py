# -*- coding: utf-8 -*-
import random

from mahjong.ai.base import BaseAI
from mahjong.meld import Meld
from mahjong.tile import TilesConverter
from mahjong.utils import is_pair, is_pon
from mahjong.ai.strategies.main import BaseStrategy

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
        
    def erase_state(self):
        self.current_strategy = None
        self.in_defence = False

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
        print("\n opponnet model discards: {}\n".format(tile_to_discard))
        return tile_to_discard
    
    
