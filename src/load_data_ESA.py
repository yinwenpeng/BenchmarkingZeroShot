#conding=utf8

import json
import codecs
# import nltk
from collections import defaultdict
# import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
# import time
# from scipy import sparse
# from sklearn.metrics.pairwise import cosine_similarity
ESA_word2id={}

def transfer_wordlist_2_idlist_with_existing_word2id(token_list):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:
        id=ESA_word2id.get(word)
        if id is not None: # if word was not in the vocabulary
            idlist.append(id)
    return idlist

def load_ESA_word2id():
    global ESA_word2id
    route = '/home/wyin3/Datasets/Wikipedia20190320/parsed_output/statistics_from_json/'
    with open(route+'word2id.json', 'r') as fp2:
        ESA_word2id = json.load(fp2)
    print('load ESA word2id succeed')

def load_yahoo():
    yahoo_path = '/home/wyin3/Datasets/YahooClassification/yahoo_answers_csv/'
    files = ['test_tokenized.txt'] #'train_tokenized.txt',
    # word2id={}
    all_texts=[]
    # all_masks=[]
    all_labels=[]
    all_word2DF=defaultdict(int)
    max_sen_len=0
    for i in range(len(files)):
        print('loading file:', yahoo_path+files[i], '...')

        texts=[]
        # text_masks=[]
        labels=[]
        readfile=codecs.open(yahoo_path+files[i], 'r', 'utf-8')
        line_co=0
        for line in readfile:
            parts = line.strip().split('\t')
            if len(parts)==2:
                label_id = int(parts[0])
                text_wordlist = parts[1].strip().lower().split()
                text_len=len(text_wordlist)
                if text_len > max_sen_len:
                    max_sen_len=text_len
                text_idlist=transfer_wordlist_2_idlist_with_existing_word2id(text_wordlist)
                if len(text_idlist) >0:
                    texts.append(text_idlist)
                    labels.append(label_id)
                    idset = set(text_idlist)
                    for iddd in idset:
                        all_word2DF[iddd]+=1
                else:
                    continue

            line_co+=1
            if line_co%10000==0:
                print('line_co:', line_co)
            # if i==0 and line_co==train_size_limit:
            #     break


        all_texts.append(texts)
        all_labels.append(labels)
        print('\t\t\t size:', len(labels), 'samples')
    print('load yahoo text succeed, max sen len:',   max_sen_len)
    return all_texts, all_labels, all_word2DF

def load_labels():
    yahoo_path = '/home/wyin3/Datasets/YahooClassification/yahoo_answers_csv/'
    texts=[]
    # text_masks=[]

    readfile=codecs.open(yahoo_path+'classes.txt', 'r', 'utf-8')
    for line in readfile:
        wordlist = line.strip().replace('&', ' ').lower().split()

        text_idlist=transfer_wordlist_2_idlist_with_existing_word2id(wordlist)
        if len(text_idlist) >0:
            texts.append(text_idlist)

    print('load yahoo labelnames succeed, totally :', len(texts), 'label names')

    return texts

def load_yahoo_and_labelnames():
    load_ESA_word2id()
    all_texts, all_labels, all_word2DF = load_yahoo()
    labelnames = load_labels()
    return all_texts, all_labels, all_word2DF, labelnames
