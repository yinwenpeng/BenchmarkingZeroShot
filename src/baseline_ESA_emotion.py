import time
from load_data_ESA import load_emotion_and_labelnames
from ESA import load_ESA_sparse_matrix, divide_sparseMatrix_by_list_row_wise, multiply_sparseMatrix_by_list_row_wise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import vstack
import numpy as np
from operator import itemgetter
from scipy.special import softmax
from preprocess_emotion import emotion_f1_given_goldlist_and_predlist



all_texts, all_labels, all_word2DF, labelnames = load_emotion_and_labelnames()
ESA_sparse_matrix = load_ESA_sparse_matrix().tocsr()
# ESA_sparse_matrix_2_dict = {}
#
# def ESA_sparse_matrix_into_dict():
#     global ESA_sparse_matrix_2_dict
#     for i in range(ESA_sparse_matrix.shape[0]):
#         ESA_sparse_matrix_2_dict[i] = ESA_sparse_matrix.getrow(i)
#     print('ESA_sparse_matrix_into_dict succeed')


def text_idlist_2_ESAVector(idlist, text_bool):
    # sub_matrix = ESA_sparse_matrix[idlist,:]
    # return  sub_matrix.mean(axis=0)
    # matrix_list = []
    # for id in idlist:
    #     matrix_list.append(ESA_sparse_matrix_2_dict.get(id))
    # stack_matrix = vstack(matrix_list)
    # return  stack_matrix.mean(axis=0)
    # print('idlist:', idlist)
    if text_bool:
        sub_matrix = ESA_sparse_matrix[idlist,:]
        # myvalues = list(itemgetter(*idlist)(all_word2DF))
        myvalues = [all_word2DF.get(id) for id in idlist]
        weighted_matrix = divide_sparseMatrix_by_list_row_wise(sub_matrix, myvalues)
        return  weighted_matrix.sum(axis=0)
    else: #label names
        sub_matrix = ESA_sparse_matrix[idlist,:]
        return  sub_matrix.sum(axis=0)

def text_idlist_2_ESAVector_attention(idlist, text_bool, label_veclist):


    sub_matrix = ESA_sparse_matrix[idlist,:]
    result_veclist = []
    for vec in label_veclist:
        cos = cosine_similarity(sub_matrix, vec)
        att_matrix = multiply_sparseMatrix_by_list_row_wise(sub_matrix, softmax(cos))
        result_veclist.append(att_matrix.sum(axis=0))

    return  result_veclist

    # if text_bool:
    #     sub_matrix = ESA_sparse_matrix[idlist,:]
    #     myvalues = [all_word2DF.get(id) for id in idlist]
    #     weighted_matrix = divide_sparseMatrix_by_list_row_wise(sub_matrix, myvalues)
    #     return  weighted_matrix.sum(axis=0)
    # else: #label names
    #     sub_matrix = ESA_sparse_matrix[idlist,:]
    #     return  sub_matrix.sum(axis=0)

def ESA_cosine():
    origin_type_list = ['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love']
    label_veclist = []
    for i in range(len(labelnames)):
        labelname_idlist = labelnames[i]
        '''label rep is sum up all word ESA vectors'''
        label_veclist.append(text_idlist_2_ESAVector(labelname_idlist, False))
    # print(label_veclist)
    print('all labelnames are in vec succeed')
    labels = all_labels[0]
    # print('all_labels:', labels[:10])
    sample_size = len(labels)
    print('total test size:', sample_size)
    hit_size = 0
    co=0
    start_time = time.time()
    pred_type_list = []
    gold_type_list = []
    for sample_index in range(sample_size):
        text_idlist = all_texts[0][sample_index]
        '''text rep is weighted sum up of ESA vectors'''
        text_vec = text_idlist_2_ESAVector(text_idlist, True)
        cos_array=cosine_similarity(text_vec, np.vstack(label_veclist))
        list_cosine = list(cos_array[0])
        # print('list_cosine:', list_cosine)
        max_prob = -100.0
        max_index = -1
        for i in range(len(list_cosine)):
            if list_cosine[i] > max_prob:
                max_prob = list_cosine[i]
                max_index = i
        if max_index ==-1 or max_prob < 0.01:
            pred_type = 'noemo'
        else:
            pred_type = origin_type_list[max_index]
        # print('pred_type:', pred_type)
        # print('pred_type_list_i:', pred_type_list_i)
        # print('all_labels:', labels[:10])
        # print('gold_type_list_i:',labels[i], i )
        pred_type_list.append(pred_type)
        gold_type_list.append(labels[sample_index])

        v0, v1, all_f1 = emotion_f1_given_goldlist_and_predlist(gold_type_list, pred_type_list, set(['sadness',  'anger', 'fear',  'shame',  'love']), set(['joy',  'disgust',  'surprise',  'guilt']))
        # seen_f1_v1, unseen_f1_v1, all_f1_v1 = situation_f1_given_goldlist_and_predlist(gold_type_list, pred_type_list, set(['evac','utils', 'shelter','food', 'terrorism']))

        co+=1
        print(co, '...', v0, v1, all_f1)
        if co%10==0:
            spend_time = (time.time()-start_time)/60.0
            print('\t\t\t\t\t',spend_time, 'mins')

    print('over.')



if __name__ == '__main__':
    ESA_cosine()
