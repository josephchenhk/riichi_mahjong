# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:46:22 2017

@author: joseph.chen
"""

from mahjong.hand import FinishedHand

class CheckWaiting(object):
    
    def __init__(self):
        self.finished_hand = FinishedHand()
    
    def check(self, 
              hand, 
              win_tile,
              is_tsumo=False,
              is_riichi=False,
              is_dealer=False,
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
              open_sets=None,
              dora_indicators=None,
              called_kan_indices=None,
              player_wind=None,
              round_wind=None):
        result = self.finished_hand.estimate_hand_value(hand, 
                                                        win_tile,
                                                        is_tsumo=is_tsumo,
                                                        is_riichi=is_riichi,
                                                        is_dealer=is_dealer,
                                                        is_ippatsu=is_ippatsu,
                                                        is_rinshan=is_rinshan,
                                                        is_chankan=is_chankan,
                                                        is_haitei=is_haitei,
                                                        is_houtei=is_houtei,
                                                        is_daburu_riichi=is_daburu_riichi,
                                                        is_nagashi_mangan=is_nagashi_mangan,
                                                        is_tenhou=is_tenhou,
                                                        is_renhou=is_renhou,
                                                        is_chiihou=is_chiihou,
                                                        open_sets=open_sets,
                                                        dora_indicators=dora_indicators,
                                                        called_kan_indices=called_kan_indices,
                                                        player_wind=player_wind,
                                                        round_wind=round_wind)
        return result
