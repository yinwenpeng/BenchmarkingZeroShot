import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score



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



    '''yahoo'''
    type_list = ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference','Computers & Internet','Sports','Business & Finance','Entertainment & Music','Family & Relationships','Politics & Government']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/YahooClassification/yahoo_answers_csv/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0])
        text = parts[1].strip()
        max_cos = 0.0
        max_type = ''
        for i, type in enumerate(type_list):
            text_emb = sent_2_emb(text.split())
            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > max_cos:
                max_cos = cos
                max_type = str(i)
        pred_label_list.append(max_type)
    readfile.close()

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


    '''emotion'''
    type_list = ['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love']#, 'noemo']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/Stuttgart_Emotion/unify-emotion-datasets-master/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0])
        text = parts[2].strip()
        max_cos = 0.0
        max_type = ''
        for i, type in enumerate(type_list):
            text_emb = sent_2_emb(text.split())
            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > max_cos:
                max_cos = cos
                max_type = str(i)
        if max_cos > 0.5:
            pred_label_list.append(max_type)
        else:
            pred_label_list.append('noemo')
    readfile.close()

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
    type_list = ['search','evacuation','infrastructure','utilities utility','water','shelter','medical assistance','food', 'crime violence', 'terrorism', 'regime change']#, 'out-of-domain']
    type_2_emb = []
    for type in type_list:
        type_2_emb.append(sent_2_emb(type.split()))
    readfile = codecs.open('/export/home/Dataset/LORELEI/zero-shot-split/test.txt', 'r', 'utf-8')
    gold_label_list = []
    pred_label_list = []
    for line in readfile:
        parts = line.strip().split('\t')
        gold_label_list.append(parts[0])
        text = parts[1].strip()
        max_cos = 0.0
        max_type = ''
        for i, type in enumerate(type_list):
            text_emb = sent_2_emb(text.split())
            type_emb = type_2_emb[i]
            cos = 1.0-cosine(text_emb, type_emb)
            if cos > max_cos:
                max_cos = cos
                max_type = str(i)
        if max_cos > 0.5:
            pred_label_list.append(max_type)
        else:
            pred_label_list.append('out-of-domain')
    readfile.close()

    all_test_labels = list(set(gold_label_list))
    f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average=None)
    seen_types_group = [['search','infrastructure','water','medical assistance', 'crime violence',  'regime change'],
    ['evacuation','utilities utility','shelter','food', 'terrorism']]
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
        if co % 1000 == 0:
            break
    print("==> word2vec is loaded")
    baseline_w2v()
