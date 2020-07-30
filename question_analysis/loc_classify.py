import json
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
import copy
import math

import pandas as pd

from qclassify import qclass

classifier = qclass()
print(classifier.spatial_tokens)
print(classifier.number_tokens)

def slip_que_ans_2(game, human=False):
    new_game = copy.deepcopy(game)
    for key in game:
        new_game[key] = copy.deepcopy(game[key])
        qas = game[key]['qas']
        new_game[key]['que'] = []
        new_game[key]['ans'] = []
        if human:
            new_game[key]['mu'] = []
        for qa in qas:
            new_game[key]['que'].append(qa['question'])
            new_game[key]['ans'].append(qa['ans'])
            if human:
                new_game[key]['mu'].append(qa['ans'] != qa['model_ans'])
    print('Splitting is done')
    return(new_game)

if __name__ == '__main__':
    with open('./analysis_data/location_only.json') as fl:
        game = json.load(fl)

    absolute ={}
    counting ={}
    relative ={}

    # Data to store in the csv
    loc_questions = {'game_id':[],
                     'question': [],
                     'dial_pos': [],
                     'location_type':[]}

    # Preprocess
    for k in game:
        data=game[k]        
        for idx, q in enumerate(data['que']):
            loc_questions['game_id'].append(k)
            loc_questions['question'].append(q)
            loc_questions['dial_pos'].append(data['qpos'][idx]) 

    total_size = len(loc_questions['question'])
    print('Total questions:', total_size)

    # Batch of questions to pass through Stanza
    batch_size = 32

    total_batches = math.ceil(len(loc_questions['question'])/batch_size)
    for b in tqdm(range(total_batches)):
        # I actually forgot how to batch correctly. So... here it is.
        # Please forgive me.
        if b == total_batches -1:
            sample = loc_questions['question'][b*batch_size: total_size] 
        else:
            sample = loc_questions['question'][b*batch_size: (b+1)*batch_size] 

        # Form the batch for Stanza separated by \n\n
        allqs = ''
        for q in sample:
            allqs += q
            allqs += '\n\n'
        allqs = allqs.strip()

        # Get classifications for all questions in allqs
        types = classifier.cascade_classifier(allqs)
        assert(len(types) == len(sample))
        loc_questions['location_type'] += types

    data = pd.DataFrame(loc_questions)
    # Print some data
    print(len(data), ' questions processed')
    print(data.groupby('location_type')['question'].count())
    data.to_csv('locq_type_v3.csv', index=False)
