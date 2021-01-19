#!/usr/bin/env python
# coding: utf-8

import json
import jsonlines
import re
from flair.data import Sentence
from flair.models import SequenceTagger

# import BERT vocab.

BERT_vocab = set()
with open("./cased_L-12_H-768_A-12/vocab.txt", 'r', encoding='UTF-8') as reader:
    for l in reader:
        BERT_vocab.add(l.rstrip('\n'))

#########################
# (1) Process train set #
#########################

tagger = SequenceTagger.load('ner') # call the tagger

# 1.1 Data Cleaning #

train_list = list()

with open('lama/train.jsonl', 'r') as jsonl_file:
    file = list(jsonl_file)

for e,jsonl_str in enumerate(file): #go over all the strings in the list
    result = json.loads(jsonl_str) # load the string as dict
    
    if result['label'] != 'NOT ENOUGH INFO': # takes only datapoints with either 'SUPPORTS' or 'REFUTES' 
        train_list.append(result)

    if e%1000 == 0:
        print("Progress: {}".format(e), end="\r")

        
print(str(len(file)-len(train_list))+
      ' datapoints out of '+str(len(file))+' have been discarded'
     )
del file
del result

# 1.2 Name Entity Recognition with flair #

new_train_list = list()

for e,d in enumerate(train_list): # iterate over the list where we pick every dictionary
    sentence = Sentence(d["claim"]) # generate the sentence object by passing the string claim
    tagger.predict(sentence) # the predict() method of the tagger is used on a sentece to return the tag of the tokens
    entities = sentence.to_dict(tag_type='ner')["entities"]  # we get access on the found entities information 
    if len(entities) == 1: # only claims with 1 entity have to be considered
        
        text = re.split('[- ]', entities[0]["text"]) # split to check if entity is multiple token (we want single token) 
        if len(text) == 1:
            
            # create the dict with the token info 
            entity_dict = {
                "mention": text[0],
                "start_character": entities[0]["start_pos"],
                "end_character": entities[0]["end_pos"]
            }
            # enrich the datapoint with token info
            d["entity"] = entity_dict
            
            # add the datapoint to the "new_train_list"
            
            new_train_list.append(d)
    else:
        continue
            
    if e%100 == 0:
        print("Progress: {}".format(e), end="\r")


# 1.3 discard entities not in BERT vocab #

counter=0

for e,d in enumerate(new_train_list):
    if d['entity']['mention'] not in BERT_vocab: # if the entity not in bert, discard it
        del new_train_list[e]
        counter +=1
print('Total entites not in BERT vocab: '+str(counter))


with jsonlines.open('output_after_Flair_and_BERT_checking.jsonl', mode='w') as writer: # write intermediate data into file
    writer.write_all(new_train_list)

#########################
# (2) Process dev_set #
#########################

# 2.1 Data Cleaning #

dev_set = list()

with open('lama/paper_dev.jsonl', 'r') as jsonl_file:
    file = list(jsonl_file)

for e,jsonl_str in enumerate(file): #go over all the strings in the list
    result = json.loads(jsonl_str) # load the string as dict
    
    if result['label'] != 'NOT ENOUGH INFO': # takes only datapoints with either 'SUPPORTS' or 'REFUTES' 
        dev_set.append(result)

    if e%1000 == 0:
        print("Progress: {}".format(e), end="\r")

        
print(str(len(file)-len(dev_set))+
      ' datapoints out of '+str(len(file))+' have been discarded'
     )
del file
del result

# 2.2 Name Entity Recognition with flair #

new_dev_set = list()

for e,d in enumerate(dev_set): # iterate over the list where we pick every dictionary
    sentence = Sentence(d["claim"]) # generate the sentence object by passing the string claim
    tagger.predict(sentence) # the predict() method of the tagger is used on a sentece to return the tag of the tokens
    entities = sentence.to_dict(tag_type='ner')["entities"]  # we get access on the found entities information 
    if len(entities) == 1: # only claims with 1 entity have to be considered
        
        text = re.split('[- ]', entities[0]["text"]) # split to check if entity is multiple token (we want single token) 
        if len(text) == 1:
            
            # create the dict with the token info 
            entity_dict = {
                "mention": text[0],
                "start_character": entities[0]["start_pos"],
                "end_character": entities[0]["end_pos"]
            }
            # enrich the datapoint with token info
            d["entity"] = entity_dict
            
            # add the datapoint to the "new_train_list"
            
            new_dev_set.append(d)
    else:
        continue
            
    if e%100 == 0:
        print("Progress: {}".format(e), end="\r")

# 2.3 discard entities not in BERT vocab #
counter=0

for e,d in enumerate(new_dev_set):
    if d['entity']['mention'] not in BERT_vocab: # if the entity not in bert, discard it
        del new_dev_set[e]
        counter +=1
print('Total entites not in BERT vocab: '+str(counter))


with jsonlines.open('output_after_Flair_and_BERT_checking_on_dev_set.jsonl', mode='w') as writer: # write intermediate data into file
    writer.write_all(new_dev_set)


# 2.4 mask the entities

masked_claim_dict = list()
for d in new_dev_set:
    entity_mention = d["entity"]["mention"]
    old_claim = d['claim']
    masked_claim = old_claim.replace(entity_mention, '[MASK]')
    #print(masked_claim)
    
    masked_claim_dict.append({d['id']:masked_claim})
    
with jsonlines.open('masked_claim_with_ID_on_dev_set.jsonl', mode='w') as writer: # create a file with masked claims for train set
    writer.write_all(masked_claim_dict)
#########################################################
# (3) Mask singletoken_test_fever_homework_NLP entities #
#########################################################

# 3.1 import singletoken_test_fever_homework_NLP #

test_NLP_list = list()
with open("lama/singletoken_test_fever_homework_NLP.jsonl", "r", encoding="UTF-8") as json_file3:
    test_NLP = list(json_file3)

    for e, jsonl_str in enumerate(test_NLP):
        get_dict = json.loads(jsonl_str)
        test_NLP_list.append(get_dict)
del get_dict
del test_NLP

# 3.2 mask the entities #

masked_test_NLP = list()
for d in test_NLP_list:
    entity_mention = d["entity"]["mention"]
    old_claim = d['claim']
    masked_claim = old_claim.replace(entity_mention, '[MASK]')
    # print(masked_claim)

    masked_test_NLP.append({d['id']: masked_claim})

with jsonlines.open('masked_claim_singletoken_test_fever_homework_NLP.jsonl',
                    mode='w') as writer:  # create a file with masked claims for train set
    writer.write_all(masked_test_NLP)
