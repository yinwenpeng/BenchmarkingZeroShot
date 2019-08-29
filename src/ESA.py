#conding=utf8
import os
import json
import codecs
import nltk
from collections import defaultdict, Counter
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import time
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
'''seven global variables'''
title2id={} #5903490+1
title_size = 0
word2id={} #6161731+1
word_size = 0
# WordTitle2Count= lil_matrix((298099,40000))#(6113524, 5828563))
WordTitle2Count= lil_matrix((6161731, 5903486))#(6113524, 5828563))
Word2TileCount=defaultdict(int)
fileset=set()

def scan_all_json_files(rootDir):
    global fileset
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            scan_all_json_files(path)
        else: # is a file
            fileset.add(path)

def load_json():
    global title2id
    global word2id
    global title_size
    global word_size
    global WordTitle2Count
    global Word2TileCount
    global fileset

    json_file_size = 0
    wiki_file_size = 0
    for json_input in fileset:
        json_file_size+=1
        print('\t\t\t', json_input)
        with codecs.open(json_input, 'r', 'utf-8') as f:
            for line in f:
                try:
                    line_dic = json.loads(line)
                except ValueError:
                    continue
                title = line_dic.get('title')
                title_id = title2id.get(title)
                if title_id is None: # if word was not in the vocabulary
                    title_id=title_size  # id of true words starts from 1, leaving 0 to "pad id"
                    title2id[title]=title_id
                    title_size+=1

                content = line_dic.get('text')
                '''this tokenizer step should be time-consuming'''
                tokenized_text = nltk.word_tokenize(content)
                word_id_set = set()
                for word in tokenized_text:
                    if word.isalpha():
                        word_id = word2id.get(word)
                        if word_id is None:
                            word_id = word_size
                            word2id[word]=word_id
                            word_size+=1
                        WordTitle2Count[str(word_id)+':'+str(title_id)]+=1
                        word_id_set.add(word_id)
                for each_word_id in word_id_set:
                    Word2TileCount[str(each_word_id)]+=1 #this word meets a new title
                wiki_file_size+=1
                print(json_file_size, '&',wiki_file_size, '...over')
                # if wiki_file_size ==4:
                #     return

def load_tokenized_json():
    '''
    we first tokenzie tool output json files into a single json file "tokenized_wiki.txt"
    now we do statistics on it
    '''
    start_time = time.time()
    global title2id
    global word2id
    global title_size
    global word_size
    global WordTitle2Count
    global Word2TileCount
    # global fileset

    route = '/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/'
    wiki_file_size = 0
    with codecs.open(route+'tokenized_wiki.txt', 'r', 'utf-8') as f:
        for line in f:
            try:
                line_dic = json.loads(line)
            except ValueError:
                continue
            title = line_dic.get('title')
            title_id = title2id.get(title)
            if title_id is None: # if word was not in the vocabulary
                title_id=title_size  # id of true words starts from 1, leaving 0 to "pad id"
                title2id[title]=title_id
                title_size+=1

            # content = line_dic.get('text')
            tokenized_text = line_dic.get('text').split()
            word2tf=Counter(tokenized_text)
            for word, tf in word2tf.items():
                word_id = word2id.get(word)
                if word_id is None:
                    word_id = word_size
                    word2id[word]=word_id
                    word_size+=1
                WordTitle2Count[word_id, title_id]=tf
                Word2TileCount[word_id]+=1 #this word meets a new title
            wiki_file_size+=1
            if wiki_file_size%10000==0:
                print(wiki_file_size, '...over')
            # if wiki_file_size ==4000:
            #     break
    f.close()
    print('load_tokenized_json over.....words:', word_size, ' title size:', title_size)
    WordTitle2Count = divide_sparseMatrix_by_list_row_wise(WordTitle2Count, Word2TileCount.values())
    print('divide_sparseMatrix_by_list_row_wise....over')
    spend_time = (time.time()-start_time)/60.0
    print(spend_time, 'mins')

def store_ESA():
    start_time = time.time()
    global title2id
    global word2id
    global WordTitle2Count
    global Word2TileCount
    route = '/export/home/Dataset/wikipedia/parsed_output/statistics_from_json/'
    with open(route+'title2id.json', 'w') as fp1:
        json.dump(title2id, fp1)
    with open(route+'word2id.json', 'w') as fp2:
        json.dump(word2id, fp2)
    # with open(route+'WordTitle2Count.json', 'w') as f3:
    #     json.dump(WordTitle2Count, f3)
    '''note that WordTitle2Count is always a sparse matrix, not a dictionary'''
    sparse.save_npz(route+"ESA_Sparse_v1.npz", WordTitle2Count)
    print('ESA sparse matrix stored over, congrats!!!')
    with open(route+'Word2TileCount.json', 'w') as f4:
        json.dump(Word2TileCount, f4)
    print('store ESA over')
    spend_time = (time.time()-start_time)/60.0
    print(spend_time, 'mins')


def tokenize_filter_tokens():
    global fileset
    route = '/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/'
    writefile = codecs.open(route+'tokenized_wiki.txt' ,'a+', 'utf-8')
    json_file_size = 0
    wiki_file_size = 0
    for json_input in fileset:
        json_file_size+=1
        print('\t\t\t', json_input)
        with codecs.open(json_input, 'r', 'utf-8') as f:
            for line in f:
                try:
                    line_dic = json.loads(line)
                except ValueError:
                    continue
                # title = line_dic.get('title')
                content = line_dic.get('text')
                tokenized_text = nltk.word_tokenize(content)
                new_text = []
                for word in tokenized_text:
                    if word.isalpha():
                        new_text.append(word)
                line_dic['text']=' '.join(new_text)
                json.dump(line_dic, writefile)
                writefile.write('\n')
                wiki_file_size+=1
                print(json_file_size, '&',wiki_file_size, '...over')
    print('tokenize over')
    writefile.close()


def reformat_into_expected_ESA():
    '''
    super super slow. do not use it
    '''
    start_time = time.time()
    global Word2TileCount
    global WordTitle2Count
    route = '/home/wyin3/Datasets/Wikipedia20190320/parsed_output/statistics_from_json/'
    rows=[]
    cols=[]
    values =[]
    size = 0
    print('WordTitle2Count:', WordTitle2Count)
    for key, value in WordTitle2Count.items(): #"0:0": 8, "1:0": 24,
        key_parts = key.split(':')
        word_id_str = key_parts[0]
        concept_id_str = key_parts[1]
        word_df =Word2TileCount.get(word_id_str)
        rows.append(int(word_id_str))
        cols.append(int(concept_id_str))
        values.append(value/word_df)
        size+=1
        if size%10000000 ==0:
            print('reformat entry sizes:', size)
    WordTitle2Count=None # release the memory of big dictionary
    print('reformat entry over, building sparse matrix...')
    sparse_matrix = csr_matrix((values, (rows, cols)))
    non_zero=sparse_matrix.nonzero()
    row_array = list(non_zero[0])
    col_array = non_zero[1]
    print('sparse matrix build succeed, start store...')
    writefile = codecs.open(route+'ESA.v1.json', 'w', 'utf-8')
    prior_row = -1
    finish_size=0
    for id, row_id in enumerate(row_array):
        if row_id !=prior_row:
            if row_id>0:
                # print(prior_row.dtype)
                json.dump({str(prior_row):new_list}, writefile)
                writefile.write('\n')
                new_list=None
                finish_size+=1
                if finish_size %1000:
                    print('finish store rows ', finish_size)
            # else:
            new_list=[]
            new_list.append(str(col_array[id])+':'+str(sparse_matrix[row_id,col_array[id]]))
            prior_row=row_id
        else:
            new_list.append(str(col_array[id])+':'+str(sparse_matrix[row_id,col_array[id]]))

    json.dump({str(prior_row):new_list}, writefile) # the last row
    writefile.close()
    print('ESA format over')
    spend_time = (time.time()-start_time)/60.0
    print(spend_time, 'mins')

def reformat_into_sparse_matrix_store():
    start_time = time.time()
    global Word2TileCount
    global WordTitle2Count
    route = '/export/home/Dataset/wikipedia/parsed_output/statistics_from_json/'
    rows=[]
    cols=[]
    values =[]
    size = 0
    for key, value in WordTitle2Count.items(): #"0:0": 8, "1:0": 24,
        key_parts = key.split(':')
        word_id_str = key_parts[0]
        concept_id_str = key_parts[1]
        word_df =Word2TileCount.get(word_id_str)
        rows.append(int(word_id_str))
        cols.append(int(concept_id_str))
        values.append(value/word_df)
        size+=1
        if size%10000000 ==0:
            print('reformat entry sizes:', size)
    WordTitle2Count=None # release the memory of big dictionary
    print('reformat entry over, building sparse matrix...')
    sparse_matrix = csr_matrix((values, (rows, cols)))
    print('sparse matrix build succeed, start store...')
    sparse.save_npz(route+"ESA_Sparse_v1.npz", sparse_matrix)
    print('ESA sparse matrix stored over, congrats!!!')
    spend_time = (time.time()-start_time)/60.0
    print(spend_time, 'mins')

def divide_sparseMatrix_by_list_row_wise(mat, lis):
    # C=lil_matrix([[2,4,6], [5,10,15]])
    # print(C)
    D=np.asarray(list(lis))
    r,c = mat.nonzero()
    val = np.repeat(1.0/D, mat.getnnz(axis=1))
    rD_sp = csr_matrix((val, (r,c)), shape=(mat.shape))
    out = mat.multiply(rD_sp)
    return  out

def multiply_sparseMatrix_by_list_row_wise(mat, lis):
    # C=lil_matrix([[2,4,6], [5,10,15]])
    # print(C)
    D=np.asarray(list(lis))
    r,c = mat.nonzero()
    val = np.repeat(D, mat.getnnz(axis=1))
    rD_sp = csr_matrix((val, (r,c)), shape=(mat.shape))
    out = mat.multiply(rD_sp)
    return  out

def load_sparse_matrix_4_cos(row1, row2):
    print('loading sparse matrix for cosine computation...')
    sparse_matrix = sparse.load_npz('/home/wyin3/Datasets/Wikipedia20190320/parsed_output/statistics_from_json/ESA_Sparse_v1.npz')
    print('cos: ', cosine_similarity(sparse_matrix.getrow(row1), sparse_matrix.getrow(row2)))

def load_ESA_sparse_matrix():
    # print('loading sparse matrix for cosine computation...')
    sparse_matrix = sparse.load_npz('/export/home/Dataset/wikipedia/parsed_output/statistics_from_json/ESA_Sparse_v1.npz')
    print('load ESA sparse matrix succeed')
    return sparse_matrix

def crs_matrix_play():
    # mat = lil_matrix((3, 5))
    # mat[0,0]+=1
    # print(mat)
    # simi = cosine_similarity(mat.getrow(0), mat.getrow(0))
    # print(simi)
    # C=lil_matrix([[2,4,6], [5,10,15]])
    # print(C)
    # D=[2,5]
    # C=divide_sparseMatrix_by_list_row_wise(C,D)
    # print(C)

    C=lil_matrix([[2,4,6], [5,10,15], [1,10,9]])
    sub=C[[0,2],:]
    print(C)
    print('haha',sub)
    print(sub.sum(axis=0))


def get_wordsize_pagesize():
    '''
    we first tokenzie tool output json files into a single json file "tokenized_wiki.txt"
    now we do statistics on it
    '''
    start_time = time.time()
    global title2id
    global word2id
    global title_size
    global word_size
    # global WordTitle2Count
    # global Word2TileCount
    # global fileset

    route = '/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/'
    wiki_file_size = 0
    with codecs.open(route+'tokenized_wiki.txt', 'r', 'utf-8') as f:
        for line in f:
            try:
                line_dic = json.loads(line)
            except ValueError:
                continue
            title = line_dic.get('title')
            title_id = title2id.get(title)
            if title_id is None: # if word was not in the vocabulary
                title_id=title_size  # id of true words starts from 1, leaving 0 to "pad id"
                title2id[title]=title_id
                title_size+=1

            # content = line_dic.get('text')
            tokenized_text = line_dic.get('text').split()
            word2tf=Counter(tokenized_text)
            for word, tf in word2tf.items():
                word_id = word2id.get(word)
                if word_id is None:
                    word_id = word_size
                    word2id[word]=word_id
                    word_size+=1
                # WordTitle2Count[word_id, title_id]=tf
                # Word2TileCount[word_id]+=1 #this word meets a new title
            wiki_file_size+=1
            if wiki_file_size%1000==0:
                print(wiki_file_size, '...over')
            if wiki_file_size ==4000:
                break
    f.close()
    print('word size:', word_size, ' title size:', title_size)

if __name__ == '__main__':
    # scan_all_json_files('/export/home/Dataset/wikipedia/parsed_output/json/')
    '''note that file size 13354 does not mean wiki pages; each file contains multiple wiki pages'''
    # print('fileset size:', len(fileset)) #fileset size: 13354
    # load_json() #time-consuming, not useful
    # store_ESA()
    '''to save time, we tokenize wiki dump and save into files for future loading'''
    # tokenize_filter_tokens()
    '''word size 6161731; page size: 5903486'''
    # get_wordsize_pagesize()
    load_tokenized_json()
    '''store all the statistic dictionary into files for future loading'''
    store_ESA()
    # load_sparse_matrix_4_cos(1,2)

    # reformat_into_sparse_matrix_store()
