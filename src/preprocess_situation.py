
import codecs
from collections import defaultdict

def combine_all_available_labeled_datasets():
    path = '/export/home/Dataset/LORELEI/'
    files = [
    'full_BBN_multi.txt',
    'il9_sf_gold.txt',
    'il10_sf_gold.txt',
    'il5_translated_seg_level_as_training_all_fields.txt',
    'il3_sf_gold.txt',
    'Mandarin_sf_gold.txt'
    ]
    label_id_re_map = {0:0,1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:11,10:9,11:10}
    writefile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'w', 'utf-8')
    all_size = 0
    label2co = defaultdict(int)
    for fil in files:
        print('loading file:', path+fil, '...')
        size = 0
        readfile=codecs.open(path+fil, 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label_list = parts[1].strip().split()
            for label in set(label_list):
                label2co[label]+=1
            text=parts[2].strip()
            writefile.write(' '.join(label_list)+'\t'+text+'\n')
            size+=1
            all_size+=1
        readfile.close()
        print('size:', size)
    writefile.close()
    print('all_size:', all_size, label2co)

if __name__ == '__main__':
    combine_all_available_labeled_datasets()
