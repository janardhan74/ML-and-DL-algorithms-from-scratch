# data link : https://download.pytorch.org/tutorial/data.zip

import string
import unicodedata
import torch
import glob
import io
import os
import random

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s:str):
    return ''.join([c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c) != "Mn" and c in ALL_LETTERS])


def load_data():
    category_lines = {}
    all_categories = []
    
    def find_files(path):
        return glob.glob(path)
    
    def read_lines(filename):
        lines = io.open(filename,encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    # print(find_files(r'RNN\data\names\*.txt'))
    for filename in find_files(r'RNN\data\names\*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        # print(lines)
        
        category_lines[category] = lines
        
    return category_lines , all_categories

"""
To represent a single letter , we use a "one-hot vector" of size <1 x n_letterd> . A one-hot vector is filled with 0's except for a 1st at index of the current letter , e.g. "b" = <0 1 0 0 0 ....>

To make a word we join bunch of those into a 2D matrix <line_length x 1 x n_letters>

The extra 1 dimesions is because the pytorch assumes that everything is in batches - we are just batch dimension of 1 here
"""

def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1,N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line),1,N_LETTERS)
    for i , letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
        
    return tensor

def random_training_example(category_lines,all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0,len(a)-1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    
    category_tensor = torch.tensor([all_categories.index(category)],dtype=torch.long)
    line_tensor = line_to_tensor(line)
    
    return category , line , category_tensor , line_tensor
    

    
if __name__ == "__main__":
    print(ALL_LETTERS)
    print(unicode_to_ascii("$%^HI how are you ;"))
    
    category_lines,all_categories = load_data()
    # print(category_lines)
    print(category_lines["Arabic"][:5])
    
    print(letter_to_tensor("J"))
    print(line_to_tensor("Jones").size())