
import codecs
from collections import defaultdict
from sklearn.metrics import f1_score
import random
import numpy as np

path = '/export/home/Dataset/LORELEI/'

def combine_all_available_labeled_datasets():

    files = [
    'full_BBN_multi.txt',
    'il9_sf_gold.txt', #repeat
    'il10_sf_gold.txt', # repeat
    'il5_translated_seg_level_as_training_all_fields.txt',
    'il3_sf_gold.txt',
    'Mandarin_sf_gold.txt' #repeat
    ]
    writefile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'w', 'utf-8')
    all_size = 0
    label2co = defaultdict(int)
    for fil in files:
        print('loading file:', path+fil, '...')
        size = 0
        readfile=codecs.open(path+fil, 'r', 'utf-8')
        stored_lines = set()
        for line in readfile:
            '''some labeled files have repeated lines'''
            if line.strip() not in stored_lines:
                parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
                label_list = parts[1].strip().split()
                for label in set(label_list):
                    label2co[label]+=1
                text=parts[2].strip()
                writefile.write(' '.join(label_list)+'\t'+text+'\n')
                size+=1
                all_size+=1
                stored_lines.add(line.strip())
        readfile.close()
        print('size:', size)
    writefile.close()
    print('all_size:', all_size, label2co)


# def split_all_labeleddata_into_subdata_per_label():
#     readfile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'r', 'utf-8')
#     label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
#     writefile_list = []
#     for label in label_list:
#         writefile = codecs.open(path+'data_per_label/'+label+'.txt', 'w', 'utf-8')
#         writefile_list.append(writefile)
#     for line in readfile:
#         parts=line.strip().split('\t')
#         label_list_instance = parts[0].strip().split()
#         for label in label_list_instance:
#             writefile_exit = writefile_list[label_list.index(label)]
#             writefile_exit.write(parts[1].strip()+'\n')
#
#     for writefile in writefile_list:
#         writefile.close()
#     readfile.close()



def build_zeroshot_test_dev_train_set():

    # test_label_size_max = {'search':80, 'evac':70, 'infra':120, 'utils':100,'water':120,'shelter':175,
    # 'med':250,'food':190,'regimechange':30,'terrorism':70,'crimeviolence':250,'out-of-domain':400}
    # dev_label_size_max = {'search':50, 'evac':30, 'infra':50, 'utils':50,'water':50,'shelter':75,
    # 'med':100,'food':80,'regimechange':15,'terrorism':40,'crimeviolence':100,'out-of-domain':200}

    label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']

    test_store_size = defaultdict(int)
    dev_store_size = defaultdict(int)
    write_test = codecs.open(path+'zero-shot-split/test.txt', 'w', 'utf-8')
    write_dev = codecs.open(path+'zero-shot-split/dev.txt', 'w', 'utf-8')
    write_train_v0 = codecs.open(path+'zero-shot-split/train_pu_half_v0.txt', 'w', 'utf-8')
    seen_types_v0 = ['search','infra','water','med','crimeviolence', 'regimechange']
    write_train_v1 = codecs.open(path+'zero-shot-split/train_pu_half_v1.txt', 'w', 'utf-8')
    seen_types_v1 = ['evac','utils', 'shelter','food', 'terrorism']
    readfile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'r', 'utf-8')
    for line in readfile:
        parts = line.strip().split('\t')
        type_set = set(parts[0].strip().split())
        '''test and dev set build'''
        rand_value = random.uniform(0, 1)
        if rand_value > 2.0/5.0:
            write_test.write(line.strip()+'\n')
        else:
            write_dev.write(line.strip()+'\n')

        '''train set build'''
        remain_type_v0 = type_set & set(seen_types_v0)
        if len(remain_type_v0) > 0:
            write_train_v0.write(' '.join(list(remain_type_v0))+'\t'+parts[1].strip()+'\n')
        remain_type_v1 = type_set & set(seen_types_v1)
        if len(remain_type_v1) > 0:
            write_train_v1.write(' '.join(list(remain_type_v1))+'\t'+parts[1].strip()+'\n')
    write_test.close()
    write_dev.close()
    write_train_v0.close()
    write_train_v1.close()
    print('zero-shot data split over')

def statistics():
    filename=[path+'zero-shot-split/test.txt', path+'zero-shot-split/dev.txt',
    path+'zero-shot-split/train_pu_half_v0.txt',path+'zero-shot-split/train_pu_half_v1.txt']
    for fil in filename:
        type2size= defaultdict(int)
        readfile=codecs.open(fil, 'r', 'utf-8')
        for line in readfile:
            type_list = line.strip().split('\t')[0].split()
            for type in type_list:
                type2size[type]+=1
        readfile.close()
        print('type2size:', type2size)


# def build_zeroshot_train_set():
#     readfile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
#     '''we do not put None type in train'''
#     label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange']
#     writefile_PU_half_0 = codecs.open(path+'zero-shot-split/train_pu_half_v0.txt', 'w', 'utf-8')
#     writefile_PU_half_1 = codecs.open(path+'zero-shot-split/train_pu_half_v1.txt', 'w', 'utf-8')
#
#     for line in readfile_remain:
#         parts = line.strip().split('\t')
#         type = parts[0]
#         if type in set(label_list):
#             if label_list.index(type) %2==0:
#                 writefile_PU_half_0.write(line.strip()+'\n')
#             else:
#                 writefile_PU_half_1.write(line.strip()+'\n')
#     writefile_PU_half_0.close()
#     writefile_PU_half_1.close()
#     print('PU half over')
#     '''PU_one'''
#     for i in range(len(label_list)):
#         readfile=codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
#         writefile_PU_one = codecs.open(path+'zero-shot-split/train_pu_one_'+'wo_'+str(i)+'.txt', 'w', 'utf-8')
#         line_co=0
#         for line in readfile:
#             parts = line.strip().split('\t')
#             type = parts[0]
#             if type in set(label_list):
#                 label_id = label_list.index(type)
#                 if label_id != i:
#                     writefile_PU_one.write(line.strip()+'\n')
#                     line_co+=1
#         writefile_PU_one.close()
#         readfile.close()
#         print('write size:', line_co)
#     print('build train over')



def evaluate_situation_zeroshot_TwpPhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types):
    '''
    pred_probs: a list, the prob for  "entail"
    pred_binary_labels: a lit, each  for 0 or 1
    eval_label_list: the gold type index; list length == lines in dev.txt
    eval_hypo_seen_str_indicator: totally hypo size, seen or unseen
    eval_hypo_2_type_index:: total hypo size, the type in [0,...n]
    seen_types: a set of type indices
    '''

    pred_probs = list(pred_probs)
    # pred_binary_labels = list(pred_binary_labels)
    total_hypo_size = len(eval_hypo_seen_str_indicator)
    total_premise_size = len(eval_label_list)
    assert len(pred_probs) == total_premise_size*total_hypo_size
    assert len(eval_hypo_seen_str_indicator) == len(eval_hypo_2_type_index)

    pred_label_list = []

    for i in range(total_premise_size):
        pred_probs_per_premise = pred_probs[i*total_hypo_size: (i+1)*total_hypo_size]
        pred_binary_labels_per_premise_harsh = pred_binary_labels_harsh[i*total_hypo_size: (i+1)*total_hypo_size]
        pred_binary_labels_per_premise_loose = pred_binary_labels_loose[i*total_hypo_size: (i+1)*total_hypo_size]

        pred_type = []
        for j in range(total_hypo_size):
            if (eval_hypo_seen_str_indicator[j] == 'seen' and pred_probs_per_premise[j]>0.6) or \
            (eval_hypo_seen_str_indicator[j] == 'unseen' and pred_probs_per_premise[j]>0.5):
                pred_type.append(eval_hypo_2_type_index[j])

        if len(pred_type) ==0:
            pred_type.append('out-of-domain')
        pred_label_list.append(pred_type)

    assert len(pred_label_list) ==  len(eval_label_list)
    type_in_test = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    type2col = { type:i for i, type in enumerate(type_in_test)}
    gold_array = np.zeros((total_premise_size,12), dtype=int)
    pred_array = np.zeros((total_premise_size,12), dtype=int)
    for i in range(total_premise_size):
        for type in pred_label_list[i]:
            pred_array[i,type2col.get(type)]=1
        for type in eval_label_list[i]:
            gold_array[i,type2col.get(type)]=1

    '''seen F1'''
    seen_f1_accu = 0.0
    seen_size = 0
    unseen_f1_accu = 0.0
    unseen_size = 0
    for i in range(len(type_in_test)):
        f1=f1_score(gold_array[:,i], pred_array[:,i], pos_label=1, average='binary')
        print(i, ':', f1)
        co = sum(gold_array[:,i])
        if type_in_test[i] in seen_types:
            seen_f1_accu+=f1*co
            seen_size+=co
        else:
            unseen_f1_accu+=f1*co
            unseen_size+=co

    seen_f1 = seen_f1_accu/(1e-6+seen_size)
    unseen_f1 = unseen_f1_accu/(1e-6+unseen_size)

    return seen_f1, unseen_f1


def situation_f1_given_goldlist_and_predlist(eval_label_list, pred_label_list, seen_types_v0, seen_types_v1):
    assert len(pred_label_list) ==  len(eval_label_list)
    total_premise_size = len(eval_label_list)
    type_in_test = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    type2col = { type:i for i, type in enumerate(type_in_test)}
    gold_array = np.zeros((total_premise_size,12), dtype=int)
    pred_array = np.zeros((total_premise_size,12), dtype=int)
    for i in range(total_premise_size):
        for type in pred_label_list[i]:
            pred_array[i,type2col.get(type)]=1
        for type in eval_label_list[i]:
            gold_array[i,type2col.get(type)]=1
    # print('gold_array:', gold_array)
    # print('pred_array:', pred_array)
    # print('seen_types:', seen_types)
    '''seen F1'''

    #


    f1_list = []
    co_list = []
    for i in range(len(type_in_test)):
        if sum(pred_array[:,i]) < 1:
            f1=0.0
        else:
            f1=f1_score(gold_array[:,i], pred_array[:,i], pos_label=1, average='binary')
        co = sum(gold_array[:,i])
        f1_list.append(f1)
        co_list.append(co)

    seen_f1_accu_v0 = 0.0
    seen_size_v0 = 0
    unseen_f1_accu_v0 = 0.0
    unseen_size_v0 = 0

    seen_f1_accu_v1 = 0.0
    seen_size_v1 = 0
    unseen_f1_accu_v1 = 0.0
    unseen_size_v1 = 0

    f1_accu = 0.0
    size_accu = 0
    for i in range(len(type_in_test)):
        f1 = f1_list[i]
        co =co_list[i]

        f1_accu+=f1*co
        size_accu+=co

        if type_in_test[i] in seen_types_v0:
            seen_f1_accu_v0+=f1*co
            seen_size_v0+=co
        else:
            unseen_f1_accu_v0+=f1*co
            unseen_size_v0+=co
        if type_in_test[i] in seen_types_v1:
            seen_f1_accu_v1+=f1*co
            seen_size_v1+=co
        else:
            unseen_f1_accu_v1+=f1*co
            unseen_size_v1+=co


    all_f1 = f1_accu/(1e-6+size_accu)

    v0 = (seen_f1_accu_v0/(1e-6+seen_size_v0), unseen_f1_accu_v0/(1e-6+unseen_size_v0))
    v1 = (seen_f1_accu_v1/(1e-6+seen_size_v1), unseen_f1_accu_v1/(1e-6+unseen_size_v1))

    return v0, v1, all_f1


def evaluate_situation_zeroshot_SinglePhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types):
    '''
    pred_probs: a list, the prob for  "entail"
    pred_binary_labels: a lit, each  for 0 or 1
    eval_label_list: the gold type index; list length == lines in dev.txt
    eval_hypo_seen_str_indicator: totally hypo size, seen or unseen
    eval_hypo_2_type_index:: total hypo size, the type in [0,...n]
    seen_types: a set of type indices
    '''

    pred_probs = list(pred_probs)
    # pred_binary_labels = list(pred_binary_labels)
    total_hypo_size = len(eval_hypo_seen_str_indicator)
    total_premise_size = len(eval_label_list)
    assert len(pred_probs) == total_premise_size*total_hypo_size
    assert len(eval_hypo_seen_str_indicator) == len(eval_hypo_2_type_index)

    # print('seen_types:', seen_types)
    # print('eval_hypo_seen_str_indicator:', eval_hypo_seen_str_indicator)
    # print('eval_hypo_2_type_index:', eval_hypo_2_type_index)


    pred_label_list = []

    for i in range(total_premise_size):
        pred_probs_per_premise = pred_probs[i*total_hypo_size: (i+1)*total_hypo_size]
        pred_binary_labels_per_premise_harsh = pred_binary_labels_harsh[i*total_hypo_size: (i+1)*total_hypo_size]
        pred_binary_labels_per_premise_loose = pred_binary_labels_loose[i*total_hypo_size: (i+1)*total_hypo_size]

        pred_type = []
        for j in range(total_hypo_size):
            if pred_binary_labels_per_premise_loose[j]==0: # is entailment
                pred_type.append(eval_hypo_2_type_index[j])

        if len(pred_type) ==0:
            pred_type.append('out-of-domain')
        pred_label_list.append(pred_type)

    assert len(pred_label_list) ==  len(eval_label_list)
    type_in_test = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    type2col = { type:i for i, type in enumerate(type_in_test)}
    gold_array = np.zeros((total_premise_size,12), dtype=int)
    pred_array = np.zeros((total_premise_size,12), dtype=int)
    for i in range(total_premise_size):
        for type in pred_label_list[i]:
            pred_array[i,type2col.get(type)]=1
        for type in eval_label_list[i]:
            gold_array[i,type2col.get(type)]=1

    '''seen F1'''
    seen_f1_accu = 0.0
    seen_size = 0
    unseen_f1_accu = 0.0
    unseen_size = 0
    for i in range(len(type_in_test)):
        f1=f1_score(gold_array[:,i], pred_array[:,i], pos_label=1, average='binary')
        co = sum(gold_array[:,i])
        if type_in_test[i] in seen_types:
            seen_f1_accu+=f1*co
            seen_size+=co
        else:
            unseen_f1_accu+=f1*co
            unseen_size+=co

    seen_f1 = seen_f1_accu/(1e-6+seen_size)
    unseen_f1 = unseen_f1_accu/(1e-6+unseen_size)

    return seen_f1, unseen_f1


def majority_baseline():
    readfile = codecs.open(path+'zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    for line in readfile:
        gold_label_list.append(line.strip().split('\t')[0].split())
    '''out-of-domain is the main type'''
    pred_label_list = [['out-of-domain']] *len(gold_label_list)
    # seen_labels = set(['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange'])
    # seen_types = set(['evac','utils','shelter','food', 'terrorism'])
    seen_types = set(['search','infra','water','med', 'crimeviolence', 'regimechange'])
    # f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = list(set(gold_label_list)), average='weighted')

    assert len(pred_label_list) ==  len(gold_label_list)
    total_premise_size = len(gold_label_list)
    type_in_test = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    type2col = { type:i for i, type in enumerate(type_in_test)}
    gold_array = np.zeros((total_premise_size,12), dtype=int)
    pred_array = np.zeros((total_premise_size,12), dtype=int)
    for i in range(total_premise_size):
        for type in pred_label_list[i]:
            pred_array[i,type2col.get(type)]=1
        for type in gold_label_list[i]:
            gold_array[i,type2col.get(type)]=1

    '''seen F1'''
    seen_f1_accu = 0.0
    seen_size = 0
    unseen_f1_accu = 0.0
    unseen_size = 0

    f1_accu = 0.0
    size_accu = 0
    for i in range(len(type_in_test)):
        f1=f1_score(gold_array[:,i], pred_array[:,i], pos_label=1, average='binary')
        co = sum(gold_array[:,i])

        f1_accu+=f1*co
        size_accu+=co
        if type_in_test[i] in seen_types:
            seen_f1_accu+=f1*co
            seen_size+=co
        else:
            unseen_f1_accu+=f1*co
            unseen_size+=co

    seen_f1 = seen_f1_accu/(1e-6+seen_size)
    unseen_f1 = unseen_f1_accu/(1e-6+unseen_size)

    all_f1 = f1_accu/(1e-6+size_accu)
    print('seen_f1:', seen_f1, 'unseen_f1:', unseen_f1, 'all:', all_f1)


if __name__ == '__main__':
    # combine_all_available_labeled_datasets()
    '''not useful'''
    # split_all_labeleddata_into_subdata_per_label()
    # build_zeroshot_test_dev_set()
    # build_zeroshot_train_set()

    # build_zeroshot_test_dev_train_set()
    # statistics()

    majority_baseline()
