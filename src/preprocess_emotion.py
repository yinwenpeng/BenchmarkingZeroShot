
import jsonlines
from collections import defaultdict
import codecs

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





if __name__ == '__main__':
    # statistics()
    build_zeroshot_test_dev_set()
