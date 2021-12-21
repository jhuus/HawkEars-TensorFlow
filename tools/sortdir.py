# Output a list of files in the input dir, sorted the way they are in Windows File Explorer,
# which is quite different from the sorting in "dir /O /B".
# This is very kludgy and it does not actually handle all Windows File Explorer sorting, 
# but it's close enough for now.
# It's used after running "extract.py -m 1" to generate a list of spectrograms in specs.txt.
# Then review them, deleting or editing lines in specs.txt as needed before re-running extract.py,
# or running delfs.py. 

import argparse
import os
import re
import sys

def sort_key(file_name):
    file_name = file_name.lower()
    start = 0
    curr = 0
    tokens = []
    while curr < len(file_name):
        if curr == start:
            # new token
            if file_name[curr].isdigit():
                in_number = True
            else:
                in_number = False
        elif in_number:
            if not file_name[curr].isdigit():
                # end this numeric token
                val = int(file_name[start:curr])
                if file_name[start] == '0':
                    tokens.append(f'{val:012d}')
                else:
                    tokens.append(f'{val:011d}')
                    
                in_number = False
                start = curr
        elif file_name[curr].isdigit():
            # end this non-numeric token
            tokens.append(file_name[start:curr])
            in_number = True
            start = curr
                
        curr += 1
        
    if in_number:
        val = int(file_name[start:curr])
        if file_name[start] == '0':
            tokens.append(f'{val:012d}')
        else:
            tokens.append(f'{val:011d}')
    else:
        tokens.append(file_name[start:curr])
        
    tokens.append(' zzz') # this puts "1.png" after "1 a".png" and before "1a.png"
    return tuple(tokens)

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='Directory containing the files.')
parser.add_argument('-o', type=str, default='out.txt', help='Output file path.')
args = parser.parse_args()

temp = os.listdir(args.d)
files = []
for file_name in temp:
    if os.path.isfile(os.path.join(args.d, file_name)):
        base, ext = os.path.splitext(file_name)
        if ext == '.png':
            files.append(base)

files.sort(key=sort_key)
with open(args.o, 'w') as output:
    # output those with numeric prefixes
    for file in files:
        if file[0].isnumeric():
            output.write(f'{file}\n')

    # output those with prefixes that are not numeric or alpha
    for file in files:
        if not file[0].isnumeric() and not file[0].isalpha():
            output.write(f'{file}\n')

    # output those with alpha prefixes
    for file in files:
        if file[0].isalpha():
            output.write(f'{file}\n')

output.close()
