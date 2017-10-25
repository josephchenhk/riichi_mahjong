# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:36:35 2017

@author: joseph.chen
"""
import os
# abs_path is the directory to store data. We don't want the data mess with 
# our scripts, and therefore put them in a separate directory.

mirror_path = os.path.join(os.getcwd(), "../../../../Projects/riichi_mahjong")
abs_data_path = os.path.abspath(mirror_path)