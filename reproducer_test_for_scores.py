import os
import re
import sys
from optparse import OptionParser

import logging
import requests
import copy
#from functools import reduce

from mahjong.ai.discard import DiscardOption
from mahjong.meld import Meld

# TODO: temporarily switch to transparent table (the enemy's hand is visible now)
#from mahjong.table import TransparentTable as Table

from mahjong.table import Table
from mahjong.tile import TilesConverter
from tenhou.client import TenhouClient
from tenhou.decoder import TenhouDecoder
from utils.logger import set_up_logging

#from mahjong.ai.agari import Agari
from extract_features import ExtractFeatures
from mahjong.table import VisibleTable as Table # we comment line 15, replace Table with VisibleTable

logger = logging.getLogger('tenhou')
logger2 = logging.getLogger('debug')


class TenhouLogReproducer(object):
    """
    The way to debug bot decisions that it made in real tenhou.net games
    """

    def __init__(self, mjlog_file=None, log_url=None, stop_tag=None):
        if log_url:
            log_id, player_position, needed_round = self._parse_url(log_url)
            log_content = self._download_log_content(log_id)
        elif mjlog_file:
            with open(mjlog_file, encoding="utf8") as f:
                log_id = mjlog_file.split("/")[-1].split(".")[0]
                player_position = 0 # tw: seat
                needed_round = 1 # ts: round
                log_content = f.read()
        rounds = self._parse_rounds(log_content)
        
        self.player_position = player_position
        self.round_content = rounds[needed_round]
        self.stop_tag = stop_tag
        self.decoder = TenhouDecoder()
        
        
        # ADD: to get results of all rounds
        self.rounds = rounds
        # ADD: to extract features to be saved
        self.extract_features = ExtractFeatures()

    def reproduce(self, dry_run=False):
        draw_tags = ['T', 'U', 'V', 'W']
        discard_tags = ['D', 'E', 'F', 'G']

        player_draw = draw_tags[self.player_position]

        player_draw_regex = re.compile('^<[{}]+\d*'.format(''.join(player_draw)))
        
        draw_regex = re.compile('^<[{}]+\d*'.format(''.join(draw_tags)))
        discard_regex = re.compile('^<[{}]+\d*'.format(''.join(discard_tags)))

        table = Table()
        previous_tag = ""
        score = 1
        is_valid_sample = False
        for n, tag in enumerate(self.round_content):
            if dry_run:
                print(tag)

            if not dry_run and tag == self.stop_tag:
                break

            if 'INIT' in tag:
                values = self.decoder.parse_initial_values(tag)

                shifted_scores = []
                for x in range(0, 4):
                    shifted_scores.append(values['scores'][self._normalize_position(x, self.player_position)])

                table.init_round(
                    values['round_number'],
                    values['count_of_honba_sticks'],
                    values['count_of_riichi_sticks'],
                    values['dora_indicator'],
                    self._normalize_position(self.player_position, values['dealer']),
                    shifted_scores,
                )

                hands = [
                    [int(x) for x in self.decoder.get_attribute_content(tag, 'hai0').split(',')],
                    [int(x) for x in self.decoder.get_attribute_content(tag, 'hai1').split(',')],
                    [int(x) for x in self.decoder.get_attribute_content(tag, 'hai2').split(',')],
                    [int(x) for x in self.decoder.get_attribute_content(tag, 'hai3').split(',')],
                ]
                 
                # DEL: we can't only initialize the main player, we must initialize
                # other players as well.
                #table.player.init_hand(hands[self.player_position])
                
                # ADD: initialize all players on the table
                table.players[0].init_hand(hands[self.player_position])
                table.players[1].init_hand(hands[(self.player_position+1)%4])
                table.players[2].init_hand(hands[(self.player_position+2)%4])
                table.players[3].init_hand(hands[(self.player_position+3)%4])
                
                # ADD: when restart a new game, we need to reinitialize the config
                self.extract_features.__init__()

            # We must deal with ALL players.
            #if player_draw_regex.match(tag) and 'UN' not in tag:
            if draw_regex.match(tag) and 'UN' not in tag:
                tile = self.decoder.parse_tile(tag)
                
                # CHG: we must deal with ALL players
                #table.player.draw_tile(tile)
                if "T" in tag:
                    table.players[0].draw_tile(tile)
                elif "U" in tag:
                    table.players[1].draw_tile(tile)
                elif "V" in tag:
                    table.players[2].draw_tile(tile)
                elif "W" in tag:
                    table.players[3].draw_tile(tile)
                    #print("After draw `W`:", table.players[3].tiles)
                
            if discard_regex.match(tag) and 'DORA' not in tag:
                tile = self.decoder.parse_tile(tag)
                player_sign = tag.upper()[1]
                
                # TODO: I don't know why the author wrote the code as below, the 
                # player_seat won't work if we use self._normalize_position. This 
                # might be a tricky part, and we need to review it later.
                #player_seat = self._normalize_position(self.player_position, discard_tags.index(player_sign))
                
                # Temporally solution to modify the player_seat
                player_seat = (discard_tags.index(player_sign) + self.player_position)%4
                #print("updated player seat:",player_seat)
                
                
                if player_seat == 0:
                    table.players[player_seat].discard_tile(DiscardOption(table.players[player_seat], tile // 4, 0, [], 0))
                else:
                    # ADD: we must take care of ALL players
                    tile_to_discard = tile
            
                    is_tsumogiri = tile_to_discard == table.players[player_seat].last_draw
                    # it is important to use table method,
                    # to recalculate revealed tiles and etc.
                    table.add_discarded_tile(player_seat, tile_to_discard, is_tsumogiri)
                    
                    #print("seat:",player_seat)
                    #print("tiles:", TilesConverter.to_one_line_string(table.players[player_seat].tiles), " discard?:", TilesConverter.to_one_line_string([tile_to_discard]))
                    table.players[player_seat].tiles.remove(tile_to_discard)
            
                    
                    # DEL
                    #table.add_discarded_tile(player_seat, tile, False)

            if '<N who=' in tag:
                meld = self.decoder.parse_meld(tag)
                #player_seat = self._normalize_position(self.player_position, meld.who)
                # Again, we change the player_seat here
                player_seat = (meld.who + self.player_position) % 4
                table.add_called_meld(player_seat, meld)

                #if player_seat == 0:
                # CHG: we need to handle ALL players here    
                if True:
                    # we had to delete called tile from hand
                    # to have correct tiles count in the hand
                    if meld.type != Meld.KAN and meld.type != Meld.CHANKAN:
                        table.players[player_seat].draw_tile(meld.called_tile)

            if '<REACH' in tag and 'step="1"' in tag:
                who_called_riichi = self._normalize_position(self.player_position,
                                                             self.decoder.parse_who_called_riichi(tag))
                table.add_called_riichi(who_called_riichi)
              
            # This part is to extract the features that will be used to train
            # our model.
            try:
                next_tag = self.round_content[n+1]
            except IndexError:
                next_tag = ""
            if '<AGARI' in next_tag:           
                who_regex = re.compile("who=\"\d+\"")
                fromWho_regex = re.compile("fromWho=\"\d+\"")           
                sc_regex = "sc=\"[+-]?\d+,[+-]?\d+,[+-]?\d+,[+-]?\d+,[+-]?\d+,[+-]?\d+,[+-]?\d+,[+-]?\d+\""
                score_regex = re.compile(sc_regex)
                machi_regex = re.compile("machi=\"\d+\"")
                
                who = int(who_regex.search(next_tag).group(0).replace('"','').split("=")[1])
                fromWho = int(fromWho_regex.search(next_tag).group(0).replace('"','').split("=")[1])
                scores = [float(s) for s in score_regex.search(next_tag).group(0).replace('"','').split("=")[1].split(",")]              
                machi = int(machi_regex.search(next_tag).group(0).replace('"','').split("=")[1])
                score = scores[fromWho*2+1] 
                player_seat, features = self.execute_extraction(tag, table)
                
                if (who!=fromWho): # tsumo is not a valid sample for our training.                    
                    if (features is not None) and (player_seat is not None) and (score<0):
                        # The first element before ";" is table_info, therefor player_info starts
                        # from index 1, and we put who+1 here.
                        self.feature_to_logger(features, who+1, machi//4, score)
                        score = 1
                    #print("\n{}\n{}\n".format(tag,table.players[who].tiles))
            else:
                player_seat, features = self.execute_extraction(tag, table)
                            
        if not dry_run:
            tile = self.decoder.parse_tile(self.stop_tag)
            print('Hand: {}'.format(table.player.format_hand_for_print(tile)))

            # to rebuild all caches
            table.player.draw_tile(tile)
            tile = table.player.discard_tile()

            # real run, you can stop debugger here
            table.player.draw_tile(tile)
            tile = table.player.discard_tile()

            print('Discard: {}'.format(TilesConverter.to_one_line_string([tile])))
            
    def feature_to_logger(self, features, player_seat, machi_34, score):
        """
        param features:
        param player_seat:
        param machi_34: int (0-33), the discarded tile that leads to a winning hand of opponent
        param score: 
        """
        features_list = features.split(";")
        assert len(features_list)==6, "<D> Features format incorrect!"
        table_info = features_list[0]
        player_info = features_list[player_seat]
        logger2.info(table_info + ";" + player_info + ";" + str(machi_34) + ";" + str(score))    
        
    def execute_extraction(self, tag, table):
        if '<D' in tag:
            features = self.extract_features.get_scores_features(table)
            return 1, features              
        if '<E' in tag:
            features = self.extract_features.get_scores_features(table)
            return 2, features
        if '<F' in tag:
            features = self.extract_features.get_scores_features(table)
            return 3, features               
        if '<G' in tag:
            features = self.extract_features.get_scores_features(table)
            return 4, features
        return None, None
    
#    def execute_extraction(self, tag, score, table, to_logger):
#        if '<D' in tag:
#            #features = self.extract_features.get_is_waiting_features(table)
#            #features = self.extract_features.get_waiting_tiles_features(table)
#            features = self.extract_features.get_scores_features(score, table)
#            if (features is not None) and to_logger:
#                features_list = features.split(";")
#                assert len(features_list)==6, "<D> Features format incorrect!"
#                score_info = features_list[0]
#                player_info = features_list[1]
#                logger2.info(score_info + ";" + player_info)
#                
#        if '<E' in tag:
#            #features = self.extract_features.get_is_waiting_features(table)
#            #features = self.extract_features.get_waiting_tiles_features(table)
#            features = self.extract_features.get_scores_features(score, table)
#            if (features is not None) and to_logger:
#                features_list = features.split(";")
#                assert len(features_list)==6, "<E> Features format incorrect!"
#                score_info = features_list[0]
#                player_info = features_list[2]
#                logger2.info(score_info + ";" + player_info)
#                
#        if '<F' in tag:
#            #features = self.extract_features.get_is_waiting_features(table)
#            #features = self.extract_features.get_waiting_tiles_features(table)
#            features = self.extract_features.get_scores_features(score, table)
#            if (features is not None) and to_logger:
#                features_list = features.split(";")
#                assert len(features_list)==6, "<F> Features format incorrect!"
#                score_info = features_list[0]
#                player_info = features_list[3]
#                logger2.info(score_info + ";" + player_info)
#               
#        if '<G' in tag:
#            #features = self.extract_features.get_is_waiting_features(table)
#            #features = self.extract_features.get_waiting_tiles_features(table)
#            features = self.extract_features.get_scores_features(score, table)
#            if (features is not None) and to_logger:
#                features_list = features.split(";")
#                assert len(features_list)==6, "<G> Features format incorrect!"
#                score_info = features_list[0]
#                player_info = features_list[4]
#                logger2.info(score_info + ";" + player_info)
                
       
            
    def reproduce_all(self, dry_run=False):
        for r in self.rounds:
            self.round_content = r
            self.reproduce(dry_run=dry_run)
            print("--------------------------------------\n")

    def _normalize_position(self, who, from_who):
        positions = [0, 1, 2, 3]
        return positions[who - from_who]

    def _parse_url(self, log_url):
        temp = log_url.split('?')[1].split('&')
        log_id, player, round_number = '', 0, 0
        for item in temp:
            item = item.split('=')
            if 'log' == item[0]:
                log_id = item[1]
            if 'tw' == item[0]:
                player = int(item[1])
            if 'ts' == item[0]:
                round_number = int(item[1])
        return log_id, player, round_number

    def _download_log_content(self, log_id):
        """
        Check the log file, and if it is not there download it from tenhou.net
        :param log_id:
        :return:
        """
        temp_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        log_file = os.path.join(temp_folder, log_id)
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                return f.read()
        else:
            url = 'http://e.mjv.jp/0/log/?{0}'.format(log_id)
            response = requests.get(url)

            with open(log_file, 'w') as f:
                f.write(response.text)

            return response.text

    def _parse_rounds(self, log_content):
        """
        Build list of round tags
        :param log_content:
        :return:
        """
        rounds = []

        game_round = []
        tag_start = 0
        tag = None
        for x in range(0, len(log_content)):
            if log_content[x] == '>':
                tag = log_content[tag_start:x + 1]
                tag_start = x + 1

            # not useful tags
            if tag and ('mjloggm' in tag or 'TAIKYOKU' in tag):
                tag = None

            # new round was started
            if tag and 'INIT' in tag:
                rounds.append(game_round)
                game_round = []

            # the end of the game
            if tag and 'owari' in tag:
                rounds.append(game_round)

            if tag:
                # to save some memory we can remove not needed information from logs
                if 'INIT' in tag:
                    # we dont need seed information
                    find = re.compile(r'shuffle="[^"]*"')
                    tag = find.sub('', tag)

                # add processed tag to the round
                game_round.append(tag)
                tag = None

        return rounds[1:]


class SocketMock(object):
    """
    Reproduce tenhou <-> bot communication
    """

    def __init__(self, log_path, log_content=''):
        self.log_path = log_path
        self.commands = []
        if not log_content:
            self.text = self._load_text()
        else:
            self.text = log_content
        self._parse_text()

    def connect(self, _):
        pass

    def shutdown(self, _):
        pass

    def close(self):
        pass

    def sendall(self, message):
        pass

    def recv(self, _):
        if not self.commands:
            raise KeyboardInterrupt('End of commands')

        return self.commands.pop(0).encode('utf-8')

    def _load_text(self):
        log_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.log_path)
        with open(log_file, 'r') as f:
            return f.read()

    def _parse_text(self):
        """
        Load list of get commands that tenhou.net sent to us
        """
        results = self.text.split('\n')
        for item in results:
            if 'Get: ' not in item:
                continue

            item = item.split('Get: ')[1]
            item = item.replace('> <', '>\x00<')
            item += '\x00'

            self.commands.append(item)


def parse_args_and_start_reproducer():
    parser = OptionParser()

    parser.add_option('-o', '--online_log',
                      type='string',
                      help='Tenhou log with specified player and round number. '
                           'Example: http://tenhou.net/0/?log=2017041516gm-0089-0000-23b4752d&tw=3&ts=2')

    parser.add_option('-l', '--local_log',
                      type='string',
                      help='Path to local log file')

    parser.add_option('-d', '--dry_run',
                      action='store_true',
                      default=False,
                      help='Special option for tenhou log reproducer. '
                           'If true, it will print all available tags in the round')

    parser.add_option('-t', '--tag',
                      type='string',
                      help='Special option for tenhou log reproducer. It indicates where to stop parse round tags')

    parser.add_option('-m', '--local_mjlog',
                      type='string',
                      help='Path to local mjlog file(original mjlog file)')

    opts, _ = parser.parse_args()

    if not opts.online_log and not opts.local_log and not opts.local_mjlog:
        print('Please, set -o or -l or -m option')
        return

    if opts.online_log and not opts.dry_run and not opts.tag:
        print('Please, set -t for real run of the online log')
        return
    
    if opts.local_mjlog:  
        set_up_logging()
        
        reproducer = TenhouLogReproducer(mjlog_file=opts.local_mjlog, stop_tag=opts.tag)
        #reproducer.reproduce(opts.dry_run)
        # we want to run records of all rounds
        reproducer.reproduce_all(opts.dry_run)

    elif opts.online_log:
        if '?' not in opts.online_log and '&' not in opts.online_log:
            print('Wrong tenhou log format, please provide log link with player position and round number')
            return

        reproducer = TenhouLogReproducer(log_url=opts.online_log, stop_tag=opts.tag)
        reproducer.reproduce(opts.dry_run)
    else:
        set_up_logging()

        client = TenhouClient(SocketMock(opts.local_log))
        try:
            client.connect()
            client.authenticate()
            client.start_game()
        except (Exception, KeyboardInterrupt) as e:
            logger.exception('', exc_info=e)
            client.end_game()


def main():
    parse_args_and_start_reproducer()


if __name__ == '__main__':
    main()
