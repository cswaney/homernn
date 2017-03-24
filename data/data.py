import numpy as np
import h5py as h5
import collections
import time

def create_data(n):

    odyssey = open('/Users/colinswaney/Desktop/datasets/odyssey.txt')
    text_lines = odyssey.readlines()
    odyssey.close()
    iliad = open('/Users/colinswaney/Desktop/datasets/iliad.txt')
    text_lines.extend(iliad.readlines())
    iliad.close()

    text_lines = [line.rstrip('\n') for line in text_lines]
    for line in text_lines:
        if len(line) == 0:
            text_lines.remove(line)
        if line[0:4] == 'BOOK':
            text_lines.remove(line)
        if line[0:4] == '----':
            text_lines.remove(line)
        if line == '\n':
            text_lines.remove(line)
    text_chars = []
    for line in text_lines:
        text_chars.extend(line)
    chars = list(set(text_chars))
    chars.sort()
    chars_map = {}
    for i in range(0, len(chars)):
        chars_map[chars[i]] = i

    text_data = []
    step = 0
    while len(text_chars) > n:
        line = []
        while len(line) < n:
            ch = text_chars.pop(0)
            line.append(chars_map[ch])
        text_data.append(line)
        step += 1
        if step % 100 == 0:
            print('step={}'.format(step))
    text_data = np.array(text_data)
    with h5.File('/Users/colinswaney/Desktop/datasets/homer.hdf5', 'w') as hdf:
        hdf.create_dataset('homer', data=text_data)
    return text_data, chars, chars_map

print("Creating text data...", end='')
data, chars, chars_map = create_data(101)
print(" done.")
