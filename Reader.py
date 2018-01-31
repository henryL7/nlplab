# ================================
# 
#  Test for reading files in "./raw" and "./postagged"
#  
#  

"""
Usage:

    path = "./LDC2010T07/ctb7.0/data/utf-8" # locate the path at "utf-8" sub-folder
    reader = Reader(file_path)

    raw_text = reader.readRaw() # get the raw text, stored in list
    type(raw_text)
>>> list

    text_seg, text_tag = reader.readPos() # get the segments and tags, both lists
    
"""

import re
import os


class Reader:
    """Read related files, return structured samples"""
    def __init__(self, directory):
        self.dir = directory
        self.regex = "<S ID=\d+>\n(.*?)\n</S>"

    def readPos(self, folder='postagged/'):
        # input:  folder(optinal) - the subfolder name
        # output: Python tuple of (text_split, text_tag)
        #         text_split - list of list of string. The parsed text.
        #         text_tag - list of list of string. The tags.
        counter = 0
        text_seg = []
        text_tag = []
        for _, _, files in os.walk(self.dir + folder):
            for file in files:
                if not file[-4:] == ".pos":
                    continue
                counter += 1
                if counter % 100 == 0:
                    print('collecting ', counter)

                with open(self.dir + folder + file, 'r', encoding='utf-8') as f:
                    raw = f.read()
                text_list = re.findall(self.regex, raw)
                
                for text in text_list:
                    items = text.strip().split(' ')
                    words = []
                    pos = []
                    for item in items:
                        t = item.split('_')
                        words.append(t[0])
                        pos.append(t[1])
                    text_seg.append(words)
                    text_tag.append(pos)

        return text_seg, text_tag


    def readRaw(self, folder='raw/'):
        # input: folder(optional) - the subfolder name
        # output: list of string. The raw text.
        text = []
        counter = 0
        for _, _, files in os.walk(self.dir + folder):
            for file in files:
                if not file[-4:] == ".raw":
                    continue
                counter += 1
                if counter % 100 == 0:
                    print('collecting ', counter)

                with open(self.dir + folder + file, 'r', encoding='utf-8') as f:
                    raw = f.read()
                text_list = re.findall(self.regex, raw)
                text += [text.strip() for text in text_list]
        print('read ', len(text), 'raw')
        return text


