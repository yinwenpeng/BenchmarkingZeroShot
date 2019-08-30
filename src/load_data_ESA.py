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
    route = '/export/home/Dataset/wikipedia/parsed_output/statistics_from_json/'
    with open(route+'word2id.json', 'r') as fp2:
        ESA_word2id = json.load(fp2)
    print('load ESA word2id succeed')

def load_yahoo():
    yahoo_path = '/export/home/Dataset/YahooClassification/yahoo_answers_csv/'
    files = ['zero-shot-split/test.txt'] #'train_tokenized.txt','zero-shot-split/test.txt'
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
                '''truncate can speed up'''
                text_wordlist = parts[1].strip().lower().split()[:100]#[:30]
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


def load_situation():
    yahoo_path = '/export/home/Dataset/LORELEI/'
    files = ['zero-shot-split/test.txt'] #'train_tokenized.txt','zero-shot-split/test.txt'
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
                label_id = parts[0].strip().split()
                '''truncate can speed up'''
                text_wordlist = parts[1].strip().lower().split()[:30]#[:30]
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
            if line_co%100==0:
                print('line_co:', line_co)
            # if i==0 and line_co==train_size_limit:
            #     break

        readfile.close()
        all_texts.append(texts)
        all_labels.append(labels)
        print('\t\t\t size:', len(labels), 'samples')
    print('load situation text succeed, max sen len:',   max_sen_len)
    return all_texts, all_labels, all_word2DF

def load_emotion():
    yahoo_path = '/export/home/Dataset/Stuttgart_Emotion/unify-emotion-datasets-master/'
    files = ['zero-shot-split/test.txt'] #'train_tokenized.txt','zero-shot-split/test.txt'
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
            if len(parts)==3:
                label_id = parts[0].strip()
                '''truncate can speed up'''
                text_wordlist = parts[2].strip().lower().split()[:30]#[:30]
                '''we found use the tokenzied text make performance always zero'''
                # text_wordlist =  [word for word in  nltk.word_tokenize(parts[2].strip()) if word.isalpha()]
                # text_wordlist = text_wordlist[:30]#[:30]
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
            if line_co%100==0:
                print('line_co:', line_co)
            # if i==0 and line_co==train_size_limit:
            #     break

        readfile.close()
        all_texts.append(texts)
        all_labels.append(labels)
        print('\t\t\t size:', len(labels), 'samples')
    print('load situation text succeed, max sen len:',   max_sen_len)
    return all_texts, all_labels, all_word2DF

def load_labels():
    yahoo_path = '/export/home/Dataset/YahooClassification/yahoo_answers_csv/'
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

def load_labels_situation():
    # yahoo_path = '/export/home/Dataset/YahooClassification/yahoo_answers_csv/'
    predefined_types_enriched = ['search','evacuation','infrastructure','utilities utility','water','shelter','medical assistance','food', 'crime violence', 'terrorism', 'regime change']
    origin_type_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange']
    texts=[]
    # text_masks=[]

    # readfile=codecs.open(yahoo_path+'classes.txt', 'r', 'utf-8')
    for type in predefined_types_enriched:
        wordlist = type.split()

        text_idlist=transfer_wordlist_2_idlist_with_existing_word2id(wordlist)
        if len(text_idlist) >0:
            texts.append(text_idlist)
    assert len(texts) ==  len(predefined_types_enriched)

    print('load yahoo labelnames succeed, totally :', len(texts), 'label names')

    return texts

def load_labels_emotion():
    # yahoo_path = '/export/home/Dataset/YahooClassification/yahoo_answers_csv/'
    type_list = ['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love']
    texts=[]
    # text_masks=[]

    # readfile=codecs.open(yahoo_path+'classes.txt', 'r', 'utf-8')
    for type in type_list:
        wordlist = type.split()

        text_idlist=transfer_wordlist_2_idlist_with_existing_word2id(wordlist)
        if len(text_idlist) >0:
            texts.append(text_idlist)
    assert len(texts) ==  len(type_list)

    print('load yahoo labelnames succeed, totally :', len(texts), 'label names')

    return texts

def load_yahoo_and_labelnames():
    load_ESA_word2id()
    all_texts, all_labels, all_word2DF = load_yahoo()
    labelnames = load_labels()
    return all_texts, all_labels, all_word2DF, labelnames

def load_situation_and_labelnames():
    load_ESA_word2id()
    all_texts, all_labels, all_word2DF = load_situation()
    # print('load all_labels:', all_labels[0][:10])
    labelnames = load_labels_situation()
    return all_texts, all_labels, all_word2DF, labelnames

def load_emotion_and_labelnames():
    load_ESA_word2id()
    all_texts, all_labels, all_word2DF = load_emotion()
    # print('load all_labels:', all_labels[0][:10])
    labelnames = load_labels_emotion()
    return all_texts, all_labels, all_word2DF, labelnames
