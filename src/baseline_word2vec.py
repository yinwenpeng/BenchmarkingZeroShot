import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score
import codecs


def sent_2_emb(wordlist):
    emb_list = []
    for word in wordlist:
        emb = word2vec.get(word, None)
        if emb is not None:
            emb_list.append(emb)
    if len(emb_list) > 0:
        arr = np.array(emb_list)
        return np.sum(arr, axis=0)
    else:
        return np.array([0.0]*300)



def baseline_w2v():
    




    '''emotion'''
    type_list = ['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love']#, 'noemo']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/Stuttgart_Emotion/unify-emotion-datasets-master/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    co = 0
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0])
        text = parts[2].strip()
        max_cos = 0.0
        max_type = -1
        text_emb = sent_2_emb(text.split())
        for i, type in enumerate(type_list):

            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > max_cos:
                max_cos = cos
                max_type = type
        if max_cos > 0.0:
            pred_label_list.append(max_type)
        else:
            pred_label_list.append('noemo')
        co+=1
        if co % 1000 == 0:
            print('emotion co:', co)
    readfile.close()
    print('gold_label_list:', gold_label_list[:200])
    print('pred_label_list:', pred_label_list[:200])
    all_test_labels = list(set(gold_label_list))
    f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average=None)
    seen_types_group = [['sadness',  'anger',  'fear',  'shame',  'love'],['joy',  'disgust',  'surprise',  'guilt']]
    for i in range(len(seen_types_group)):
        seen_types = seen_types_group[i]

        seen_f1_accu = 0.0
        seen_size = 0
        unseen_f1_accu = 0.0
        unseen_size = 0
        for i in range(len(all_test_labels)):
            f1=f1_score_per_type[i]
            co = gold_label_list.count(all_test_labels[i])
            if all_test_labels[i] in seen_types:
                seen_f1_accu+=f1*co
                seen_size+=co
            else:
                unseen_f1_accu+=f1*co
                unseen_size+=co
        print('seen:', seen_f1_accu/seen_size, 'unseen:', unseen_f1_accu/unseen_size)
    print('overall:', f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average='weighted'))

    '''situation'''
    origin_type_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    type_list = ['search','evacuation','infrastructure','utilities utility','water','shelter','medical assistance','food', 'crime violence', 'terrorism', 'regime change']#, 'out-of-domain']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/LORELEI/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    co=0
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0].split())
        text = parts[1].strip()
        max_cos = 0.0
        pred_type_i = []
        text_emb = sent_2_emb(text.split())
        for i, type in enumerate(origin_type_list[:-1]):
            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > 0.5:
                pred_type_i.append(type)
        if len(pred_type_i) == 0:
            pred_type_i.append('out-of-domain')
        pred_label_list.append(pred_type_i)
        co+=1
        if co % 1000 == 0:
            print('situation co:', co)
    readfile.close()
    print('gold_label_list:', gold_label_list[:200])
    print('pred_label_list:', pred_label_list[:200])

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

    f1_list = []
    size_list = []
    for i in range(len(type_in_test)):
        f1=f1_score(gold_array[:,i], pred_array[:,i], pos_label=1, average='binary')
        co = sum(gold_array[:,i])
        f1_list.append(f1)
        size_list.append(co)

    print('f1_list:',f1_list)
    print('size_list:', size_list)
    seen_types_group = [['search','infra','water','med', 'crimeviolence', 'regimechange'],
    ['evac','utils','shelter','food', 'terrorism']]
    for i in range(len(seen_types_group)):
        seen_types = seen_types_group[i]

        seen_f1_accu = 0.0
        seen_size = 0
        unseen_f1_accu = 0.0
        unseen_size = 0
        for i in range(len(type_in_test)):
            if type_in_test[i] in seen_types:
                seen_f1_accu+=f1_list[i]*size_list[i]
                seen_size+=size_list[i]
            else:
                unseen_f1_accu+=f1_list[i]*size_list[i]
                unseen_size+=size_list[i]
        print('seen:', seen_f1_accu/seen_size, 'unseen:', unseen_f1_accu/unseen_size)

    overall = sum([f1_list[i]*size_list[i] for i in range(len(f1_list))])/sum(size_list)
    print('overall:', overall)



    '''yahoo'''
    type_list = ['society & culture', 'science & mathematics', 'health', 'education & reference','computer & internet','sports sport','business & finance','entertainment & music','Family & relationships relationship','politics & government']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/YahooClassification/yahoo_answers_csv/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    co = 0
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0])
        text = parts[1].strip()
        max_cos = 0.0
        max_type = ''
        text_emb = sent_2_emb(text.split())
        for i, type in enumerate(type_list):

            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > max_cos:
                max_cos = cos
                max_type = str(i)
        pred_label_list.append(max_type)
        co+=1
        if co % 1000 == 0:
            print('yahoo co:', co)
    readfile.close()
    print('gold_label_list:', gold_label_list[:200])
    print('pred_label_list:', pred_label_list[:200])

    # all_test_labels = list(set(gold_label_list))
    # f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average=None)
    seen_types_group = [['0','2','4','6','8'],['1','3','5','7','9']]

    for i in range(len(seen_types_group)):
        seen_types = set(seen_types_group[i])

        seen_hit = 0.0
        seen_size = 0
        unseen_hit = 0.0
        unseen_size = 0
        for i in range(len(gold_label_list)):
            if gold_label_list[i] in seen_types:

                seen_size+=1
                if gold_label_list[i] == pred_label_list[i]:
                    seen_hit+=1
            else:
                unseen_size+=1
                if gold_label_list[i] == pred_label_list[i]:
                    unseen_hit+=1
        print('seen:', seen_hit/seen_size, 'unseen:', unseen_hit/unseen_size)

    all_hit = 0
    for i in range(len(gold_label_list)):
        if gold_label_list[i] == pred_label_list[i]:
            all_hit+=1


    print('overall:', all_hit/len(gold_label_list))

if __name__ == '__main__':

    '''first load word2vec embeddings'''
    word2vec = {}

    print("==> loading 300d word2vec")

    f=open('/export/home/Dataset/word2vec_words_300d.txt', 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    co = 0
    for line in f:
        l = line.split()
        word2vec[l[0]] = list(map(float, l[1:]))
        co+=1
        if co % 50000 == 0:
            print('loading w2v size:', co)
        # if co % 10000 == 0:
        #     break
    print("==> word2vec is loaded")
    baseline_w2v()
