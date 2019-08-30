
import jsonlines
from collections import defaultdict
import codecs
from sklearn.metrics import f1_score

path = '/export/home/Dataset/Stuttgart_Emotion/unify-emotion-datasets-master/'

def statistics():
    readfile = jsonlines.open(path+'unified-dataset.jsonl' ,'r')
    domain2size = defaultdict(int)
    source2size = defaultdict(int)
    emotion2size = defaultdict(int)
    '''single-label or multi-label'''
    single2size = defaultdict(int)
    emo_dom_size = defaultdict(int)
    line_co = 0
    valid_line_co = 0
    for line2dict in readfile:
        valid_line = False
        text = line2dict.get('text')
        domain = line2dict.get('domain') #tweets etc


        source_dataset = line2dict.get('source')

        single = line2dict.get('labeled')

        emotions =line2dict.get('emotions')
        if domain == 'headlines' or domain == 'facebook-messages':
            print(emotions)
        for emotion, label in emotions.items():
            if label == 1:
                emotion2size[emotion]+=1
                emo_dom_size[(emotion, domain)]+=1
                valid_line = True
        if valid_line:
            valid_line_co+=1
            domain2size[domain]+=1
            source2size[source_dataset]+=1
            single2size[single]+=1
        line_co+=1
        if line_co%100==0:
            print(line_co)
    readfile.close()
    print('domain2size:',domain2size)
    print('source2size:',source2size)
    print('emotion2size:',emotion2size)
    print('single2size:',single2size)
    print('emo_dom_size:',emo_dom_size)
    print('line_co:', line_co)
    print('valid_line_co:', valid_line_co)

'''
domain2size: defaultdict(<class 'int'>, {'tweets': 54203, 'emotional_events': 7666, 'fairytale_sentences': 14771, 'artificial_sentences': 2268})
source2size: defaultdict(<class 'int'>, {'grounded_emotions': 2585, 'ssec': 4776, 'isear': 7666, 'crowdflower': 39740, 'tales-emotion': 14771, 'emotion-cause': 2268, 'emoint': 7102})
emotion2size: defaultdict(<class 'int'>, {'sadness': 12947, 'joy': 17833, 'anger': 8335, 'disgust': 3931, 'trust': 2700, 'fear': 14752, 'surprise': 4304, 'shame': 1096, 'guilt': 1093, 'noemo': 18765, 'love': 3820})
single2size: defaultdict(<class 'int'>, {'single': 74132, 'multi': 4776})


emo_dom_size: defaultdict(<class 'int'>, {('sadness', 'tweets'): 10355, ('joy', 'tweets'): 14433, ('anger', 'tweets'): 6024, ('disgust', 'tweets'): 2362, ('trust', 'tweets'): 2700, ('fear', 'tweets'): 12522, ('surprise', 'tweets'): 3285, ('joy', 'emotional_events'): 1094, ('fear', 'emotional_events'): 1095, ('anger', 'emotional_events'): 1096, ('sadness', 'emotional_events'): 1096, ('disgust', 'emotional_events'): 1096, ('shame', 'emotional_events'): 1096, ('guilt', 'emotional_events'): 1093, ('noemo', 'tweets'): 9370, ('love', 'tweets'): 3820, ('noemo', 'fairytale_sentences'): 9395, ('disgust', 'fairytale_sentences'): 378, ('joy', 'fairytale_sentences'): 1827, ('surprise', 'fairytale_sentences'): 806, ('fear', 'fairytale_sentences'): 712, ('anger', 'fairytale_sentences'): 732, ('sadness', 'fairytale_sentences'): 921, ('joy', 'artificial_sentences'): 479, ('sadness', 'artificial_sentences'): 575, ('surprise', 'artificial_sentences'): 213, ('disgust', 'artificial_sentences'): 95, ('anger', 'artificial_sentences'): 483, ('fear', 'artificial_sentences'): 423})
'''


def build_zeroshot_test_dev_set():
    readfile = jsonlines.open(path+'unified-dataset.jsonl' ,'r')
    writefile_test = codecs.open(path+'zero-shot-split/test.txt', 'w', 'utf-8')
    writefile_dev = codecs.open(path+'zero-shot-split/dev.txt', 'w', 'utf-8')
    writefile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'w', 'utf-8')

    emotion_type_list = ['sadness', 'joy', 'anger', 'disgust', 'trust', 'fear', 'surprise', 'shame', 'guilt', 'love', 'noemo']
    domain_list = ['tweets', 'emotional_events', 'fairytale_sentences', 'artificial_sentences']
    test_size_matrix = [[1500,2150,1650,50,800,2150,880,0,0,1100,1000],
    [300,200,400,400,0,200,0,300,300,0,0],
    [300,500,250,120,0,250,220,0,0,0,1000],
    [200,150,200,30,0,100,100,0,0,0,0]]

    dev_size_matrix = [[900,1050,400,40,250,1200,370,0,0,400,500],
    [150,150,150,150,0,150,0,100,100,0,0],
    [150,300,150,90,0,150,80,0,0,0,500],
    [100,100,100,20,0,100,50,0,0,0,0]]

    test_write_size = defaultdict(int)
    dev_write_size = defaultdict(int)

    line_co = 0
    spec_co = 0
    for line2dict in readfile:
        valid_line = False
        text = line2dict.get('text').strip()
        domain = line2dict.get('domain') #tweets etc


        source_dataset = line2dict.get('source')

        single = line2dict.get('labeled')
        '''we only consider single-label instances'''
        if single == 'single':
            target_emotion = ''
            emotions =line2dict.get('emotions')
            for emotion, label in emotions.items():
                # print(emotion, label, label == 1)
                if label == 1:
                    target_emotion = emotion
                    break
            '''there is weird case that no positive label in the instances'''
            if len(target_emotion) > 0:
                if target_emotion == 'disgust' and domain =='tweets':
                    spec_co+=1

                emotion_index = emotion_type_list.index(target_emotion)
                domain_index = domain_list.index(domain)
                if test_write_size.get((domain, target_emotion),0) < test_size_matrix[domain_index][emotion_index]:
                    writefile_test.write(target_emotion+'\t'+domain+'\t'+text+'\n')
                    test_write_size[(domain, target_emotion)]+=1
                elif dev_write_size.get((domain, target_emotion),0) < dev_size_matrix[domain_index][emotion_index]:
                    writefile_dev.write(target_emotion+'\t'+domain+'\t'+text+'\n')
                    dev_write_size[(domain, target_emotion)]+=1
                else:
                    writefile_remain.write(target_emotion+'\t'+domain+'\t'+text+'\n')

                line_co+=1
                if line_co%100==0:
                    print(line_co)
    writefile_test.close()
    writefile_dev.close()
    writefile_remain.close()
    print('test, dev, train build over')
    print(spec_co)

    writefile_test = codecs.open(path+'zero-shot-split/test.txt', 'r', 'utf-8')
    co=defaultdict(int)
    for line in writefile_test:
        parts = line.strip().split('\t')
        co[(parts[0], parts[1])]+=1
    writefile_test.close()
    print(co, '\n')

    writefile_dev = codecs.open(path+'zero-shot-split/dev.txt', 'r', 'utf-8')
    co=defaultdict(int)
    for line in writefile_dev:
        parts = line.strip().split('\t')
        co[(parts[0], parts[1])]+=1
    writefile_dev.close()
    print(co, '\n')
    writefile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
    co=defaultdict(int)
    for line in writefile_remain:
        parts = line.strip().split('\t')
        co[(parts[0], parts[1])]+=1
    writefile_remain.close()
    print(co)

def build_zeroshot_train_set():
    readfile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
    emotion_type_list = ['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love']
    writefile_PU_half_0 = codecs.open(path+'zero-shot-split/train_pu_half_v0.txt', 'w', 'utf-8')
    writefile_PU_half_1 = codecs.open(path+'zero-shot-split/train_pu_half_v1.txt', 'w', 'utf-8')

    for line in readfile_remain:
        parts = line.strip().split('\t')
        emotion = parts[0]
        if emotion in set(emotion_type_list):
            if emotion_type_list.index(emotion) %2==0:
                writefile_PU_half_0.write(line.strip()+'\n')
            else:
                writefile_PU_half_1.write(line.strip()+'\n')
    writefile_PU_half_0.close()
    writefile_PU_half_1.close()
    print('PU half over')
    '''PU_one'''
    for i in range(len(emotion_type_list)):
        readfile=codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
        writefile_PU_one = codecs.open(path+'zero-shot-split/train_pu_one_'+'wo_'+str(i)+'.txt', 'w', 'utf-8')
        line_co=0
        for line in readfile:
            parts = line.strip().split('\t')
            if len(parts)==3:
                emotion = parts[0]
                if emotion in set(emotion_type_list):
                    label_id = emotion_type_list.index(emotion)
                    if label_id != i:
                        writefile_PU_one.write(line.strip()+'\n')
                        line_co+=1
        writefile_PU_one.close()
        readfile.close()
        print('write size:', line_co)
    print('build train over')




def evaluate_emotion_zeroshot_TwpPhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types):
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


        # print('pred_probs_per_premise:',pred_probs_per_premise)
        # print('pred_binary_labels_per_premise:', pred_binary_labels_per_premise)


        '''first check if seen types get 'entailment'''
        seen_get_entail_flag=False
        for j in range(total_hypo_size):
            if eval_hypo_seen_str_indicator[j] == 'seen' and pred_binary_labels_per_premise_loose[j]==0:
                seen_get_entail_flag=True
                break
        '''first check if unseen types get 'entailment'''
        unseen_get_entail_flag=False
        for j in range(total_hypo_size):
            if eval_hypo_seen_str_indicator[j] == 'unseen' and pred_binary_labels_per_premise_loose[j]==0:
                unseen_get_entail_flag=True
                break

        if seen_get_entail_flag and unseen_get_entail_flag:

            '''compare their max prob'''
            max_prob_seen = -1.0
            max_seen_index = -1
            max_prob_unseen = -1.0
            max_unseen_index = -1
            for j in range(total_hypo_size):
                its_prob = pred_probs_per_premise[j]
                if eval_hypo_seen_str_indicator[j] == 'unseen':
                    if its_prob > max_prob_unseen:
                        max_prob_unseen = its_prob
                        max_unseen_index = j
                else:
                    if its_prob > max_prob_seen:
                        max_prob_seen = its_prob
                        max_seen_index = j
            if  max_prob_seen - max_prob_unseen > 0.05:
                pred_type = eval_hypo_2_type_index[max_seen_index]
            else:
                pred_type = eval_hypo_2_type_index[max_unseen_index]

        elif unseen_get_entail_flag:
            '''find the unseen type with highest prob'''
            max_j = -1
            max_prob = -1.0
            for j in range(total_hypo_size):
                if eval_hypo_seen_str_indicator[j] == 'unseen':
                    its_prob = pred_probs_per_premise[j]
                    if its_prob > max_prob:
                        max_prob = its_prob
                        max_j = j
            pred_type = eval_hypo_2_type_index[max_j]

        elif seen_get_entail_flag:
            '''find the seen type with highest prob'''
            max_j = -1
            max_prob = -1.0
            for j in range(total_hypo_size):
                if eval_hypo_seen_str_indicator[j] == 'seen' and pred_binary_labels_per_premise_loose[j]==0:
                    its_prob = pred_probs_per_premise[j]
                    if its_prob > max_prob:
                        max_prob = its_prob
                        max_j = j
            assert max_prob > 0.5
            pred_type = eval_hypo_2_type_index[max_j]
        elif (not seen_get_entail_flag) and (not unseen_get_entail_flag):
            '''it means noemo'''
            pred_type = 'noemo'
        pred_label_list.append(pred_type)

    assert len(pred_label_list) ==  len(eval_label_list)

    all_test_labels = list(set(eval_label_list))
    f1_score_per_type = f1_score(eval_label_list, pred_label_list, labels = all_test_labels, average=None)
    print('all_test_labels:', all_test_labels)
    print('f1_score_per_type:', f1_score_per_type)
    print('type size:', [eval_label_list.count(type) for type in all_test_labels])

    '''seen F1'''
    seen_f1_accu = 0.0
    seen_size = 0
    unseen_f1_accu = 0.0
    unseen_size = 0
    for i in range(len(all_test_labels)):
        f1=f1_score_per_type[i]
        co = eval_label_list.count(all_test_labels[i])
        if all_test_labels[i] in seen_types:
            seen_f1_accu+=f1*co
            seen_size+=co
        else:
            unseen_f1_accu+=f1*co
            unseen_size+=co




    seen_f1 = seen_f1_accu/(1e-6+seen_size)
    unseen_f1 = unseen_f1_accu/(1e-6+unseen_size)

    return seen_f1, unseen_f1

def evaluate_emotion_zeroshot_SinglePhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types):
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


        max_prob = -100.0
        max_index = -1
        for j in range(total_hypo_size):
            if pred_binary_labels_per_premise_loose[j]==0: # is entailment
                if pred_probs_per_premise[j] > max_prob:
                    max_prob = pred_probs_per_premise[j]
                    max_index = j

        if max_index == -1:
            pred_label_list.append('out-of-domain')
        else:
            pred_label_list.append(eval_hypo_2_type_index[max_index])

    assert len(pred_label_list) ==  len(eval_label_list)

    all_test_labels = list(set(eval_label_list))
    f1_score_per_type = f1_score(eval_label_list, pred_label_list, labels = all_test_labels, average=None)
    print('all_test_labels:', all_test_labels)
    print('f1_score_per_type:', f1_score_per_type)
    print('type size:', [eval_label_list.count(type) for type in all_test_labels])

    '''seen F1'''
    seen_f1_accu = 0.0
    seen_size = 0
    unseen_f1_accu = 0.0
    unseen_size = 0
    for i in range(len(all_test_labels)):
        f1=f1_score_per_type[i]
        co = eval_label_list.count(all_test_labels[i])
        if all_test_labels[i] in seen_types:
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
        gold_label_list.append(line.strip().split('\t')[0])
    '''joy is the main emoion'''
    pred_label_list = ['joy'] *len(gold_label_list)
    # seen_labels = set(['sadness', 'joy', 'anger', 'disgust', 'fear', 'surprise', 'shame', 'guilt', 'love'])
    seen_types = set(['joy',  'disgust',  'surprise',  'guilt'])
    # f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = list(set(gold_label_list)), average='weighted')

    all_test_labels = list(set(gold_label_list))
    f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average=None)

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




    seen_f1 = seen_f1_accu/(1e-6+seen_size)
    unseen_f1 = unseen_f1_accu/(1e-6+unseen_size)

    print('seen_f1:', seen_f1, 'unseen_f1:', unseen_f1)

def emotion_f1_given_goldlist_and_predlist(gold_label_list, pred_label_list, seen_types_v0, seen_types_v1):

    # print('gold_label_list:', gold_label_list)
    # print('pred_label_list:', pred_label_list)
    all_test_labels = list(set(gold_label_list))
    f1_score_per_type = f1_score(gold_label_list, pred_label_list, labels = all_test_labels, average=None)
    # print('f1_score_per_type:', f1_score_per_type)
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
    for i in range(len(all_test_labels)):
        f1=f1_score_per_type[i]
        co = gold_label_list.count(all_test_labels[i])
        # print('f1:', f1)
        # print('co:', co)

        f1_accu+=f1*co
        size_accu+=co

        if all_test_labels[i] in seen_types_v0:
            seen_f1_accu_v0+=f1*co
            seen_size_v0+=co
        else:
            unseen_f1_accu_v0+=f1*co
            unseen_size_v0+=co

        if all_test_labels[i] in seen_types_v1:
            seen_f1_accu_v1+=f1*co
            seen_size_v1+=co
        else:
            unseen_f1_accu_v1+=f1*co
            unseen_size_v1+=co


    v0 = (seen_f1_accu_v0/(1e-6+seen_size_v0), unseen_f1_accu_v0/(1e-6+unseen_size_v0))
    v1 = (seen_f1_accu_v1/(1e-6+seen_size_v1), unseen_f1_accu_v1/(1e-6+unseen_size_v1))
    all_f1 = f1_accu/(1e-6+size_accu)


    return v0, v1, all_f1


def forfun():
    readfile = codecs.open('/export/home/Dataset/Stuttgart_Emotion/unify-emotion-datasets-master/zero-shot-split/test.txt', 'r', 'utf-8')
    co=0
    for line in readfile:
        if line.strip().split('\t')[0] != 'noemo':
            co+=1
        else:
            print(co) #4685
            break
    readfile.close()
if __name__ == '__main__':
    # statistics()
    # build_zeroshot_test_dev_set()
    # build_zeroshot_train_set()

    # majority_baseline()

    forfun()
