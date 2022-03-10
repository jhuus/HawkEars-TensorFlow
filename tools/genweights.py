# Generate data/weights.txt, with a weight for every class in data/classes.txt.
# Set each weight to average / count, given the count of spectrograms for the class.
# This compensates for data imbalances.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import constants
from core import database
from core import util

WEIGHT_FILE = '../data/weights.txt'

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name.')
parser.add_argument('-m', type=float, default=0.4, help='Minimum weight.')
parser.add_argument('-x', type=float, default=2.5, help='Maximum weight.')
args = parser.parse_args()

db_name = args.f
db = database.Database(f'../data/{db_name}.db')

class_names = util.get_class_list(f'../{constants.CLASSES_FILE}')
if len(class_names) == 0:
    print('empty class list')
    sys.exit()

num_specs = {}
count = 0
sum = 0
for class_name in class_names:
    print(f'processing {class_name}')
    num_specs[class_name] = db.get_num_spectrograms(class_name)
    if num_specs[class_name] != None and num_specs[class_name] > 0:
        result = db.get_subcategory_details_by_name(class_name)
        
        # factor in the weight field from the database, which scales the normalized
        # weight accordingly
        subcat_id, cat_id, code, synonym, ignore, weight = result
        if weight != None:
            num_specs[class_name] /= weight
        
        sum += num_specs[class_name]
        count += 1
    else:
        print(f'fatal error: no spectrograms found for {class_name}')
        quit()
    
# dividing each count into the average keeps the average weight at 1,
# which ensures that weighting doesn't change the overall learning rate
average = sum / count
with open(WEIGHT_FILE,'w') as output:
    for class_name in class_names:
        if class_name in num_specs.keys():
            weight = average / num_specs[class_name]
            weight = max(weight, args.m)
            weight = min(weight, args.x)
            output.write(f'{class_name},{weight :.3f}\n')
