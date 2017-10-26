# -*- coding: utf-8 -*-


class Tile(object):
    value = None
    is_tsumogiri = None

    def __init__(self, value, is_tsumogiri):
        self.value = value
        self.is_tsumogiri = is_tsumogiri


class TilesConverter(object):

    @staticmethod
    def to_one_line_string(tiles):
        """
        Convert 136 tiles array to the one line string
        Example of output 123s123p123m33z
        """
        tiles = sorted(tiles)

        man = [t for t in tiles if t < 36]

        pin = [t for t in tiles if 36 <= t < 72]
        pin = [t - 36 for t in pin]

        sou = [t for t in tiles if 72 <= t < 108]
        sou = [t - 72 for t in sou]

        honors = [t for t in tiles if t >= 108]
        honors = [t - 108 for t in honors]

        sou = sou and ''.join([str((i // 4) + 1) for i in sou]) + 's' or ''
        pin = pin and ''.join([str((i // 4) + 1) for i in pin]) + 'p' or ''
        man = man and ''.join([str((i // 4) + 1) for i in man]) + 'm' or ''
        honors = honors and ''.join([str((i // 4) + 1) for i in honors]) + 'z' or ''

        return man + pin + sou + honors

    @staticmethod
    def to_34_array(tiles):
        """
        Convert 136 array to the 34 tiles array
        """
        results = [0] * 34
        for tile in tiles:
            tile //= 4
            results[tile] += 1
        return results

    @staticmethod
    def to_136_array(tiles):
        """
        Convert 34 array to the 136 tiles array
        """
        temp = []
        results = []
        for x in range(0, 34):
            if tiles[x]:
                temp_value = [x * 4] * tiles[x]
                for tile in temp_value:
                    if tile in results:
                        count_of_tiles = len([x for x in temp if x == tile])
                        new_tile = tile + count_of_tiles
                        results.append(new_tile)

                        temp.append(tile)
                    else:
                        results.append(tile)
                        temp.append(tile)
        return results

    @staticmethod
    def string_to_136_array(sou=None, pin=None, man=None, honors=None):
        """
        Method to convert one line string tiles format to the 136 array
        We need it to increase readability of our tests
        """
        def _split_string(string, offset):
            data = []
            temp = []

            if not string:
                return []

            for i in string:
                tile = offset + (int(i) - 1) * 4
                if tile in data:
                    count_of_tiles = len([x for x in temp if x == tile])
                    new_tile = tile + count_of_tiles
                    data.append(new_tile)

                    temp.append(tile)
                else:
                    data.append(tile)
                    temp.append(tile)

            return data

        results = _split_string(man, 0)
        results += _split_string(pin, 36)
        results += _split_string(sou, 72)
        results += _split_string(honors, 108)

        return results

    @staticmethod
    def string_to_34_array(sou=None, pin=None, man=None, honors=None):
        """
        Method to convert one line string tiles format to the 34 array
        We need it to increase readability of our tests
        """
        results = TilesConverter.string_to_136_array(sou, pin, man, honors)
        results = TilesConverter.to_34_array(results)
        return results

    @staticmethod
    def find_34_tile_in_136_array(tile34, tiles):
        """
        Our shanten calculator will operate with 34 tiles format,
        after calculations we need to find calculated 34 tile
        in player's 136 tiles.

        For example we had 0 tile from 34 array
        in 136 array it can be present as 0, 1, 2, 3
        """
        if tile34 is None or tile34 > 33:
            return None

        tile = tile34 * 4

        possible_tiles = [tile] + [tile + i for i in range(1, 4)]

        found_tile = None
        for possible_tile in possible_tiles:
            if possible_tile in tiles:
                found_tile = possible_tile
                break

        return found_tile
    
    # ADD: convert tiles in 136 format to 34 format
    @staticmethod
    def to_34_tiles(tiles:list)->list:
        """
        Convert tiles in 136 format to 34 format. An example of tiles [0,1,2,3] 
        in 136 format (which means four mans) would be converted to [0,0,0,0] 
        in 34 format.
        
        `Note`: this function serves different purpose to function `to_34_array`
        as above.
        """
        return [t//4 for t in tiles]

    # ADD: convert tiles to 37 (including three red 5 tiles) array
    @staticmethod
    def to_37_array(tiles):
        """
        Convert 136 array to the 37 tiles array
        """
        results = [0] * 37
        for tile in tiles:
            if tile==16: 
                results[34] += 1 # red man 5
            elif tile==52: 
                results[35] += 1 # red pin 5
            elif tile==88:
                results[36] += 1 # red suo 5
            else:
                tile //= 4
                results[tile] += 1
        return results
    
    # ADD: convert tiles in 136 format to 37 format
    @staticmethod
    def to_37_tiles(tiles:list)->list:
        """
        Convert tiles in 136 format to 37 format. An example of tiles [0,1,52,3] 
        in 136 format (which means four mans) would be converted to [0,0,35,0] 
        in 34 format.
        
        `Note`: this function serves different purpose to function `to_34_array`
        as above.
        """
        tiles_37 = []
        for tile in tiles:
            if tile==16: 
                tiles_37.append(34) # red man 5
            elif tile==52: 
                tiles_37.append(35) # red pin 5
            elif tile==88:
                tiles_37.append(36) # red suo 5
            else:
                tile //= 4
                tiles_37.append(tile)
        return tiles_37
    
    # ADD: convert list of tiles to a fixed length list (of length 136). For
    # example, [0,4,8,9] would be converted to [1,0,0,0,1,0,0,0,1,1,0,...]
    @staticmethod
    def tiles_to_136_code(tiles:list)->list:
        tiles_136 = [0]*136
        for t in tiles:
            tiles_136[t] = 1
        return tiles_136