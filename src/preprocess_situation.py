
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


def split_all_labeleddata_into_subdata_per_label():
    readfile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'r', 'utf-8')
    label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']
    writefile_list = []
    for label in label_list:
        writefile = codecs.open(path+'data_per_label/'+label+'.txt', 'w', 'utf-8')
        writefile_list.append(writefile)
    for line in readfile:
        parts=line.strip().split('\t')
        label_list_instance = parts[0].strip().split()
        for label in label_list_instance:
            writefile_exit = writefile_list[label_list.index(label)]
            writefile_exit.write(parts[1].strip()+'\n')

    for writefile in writefile_list:
        writefile.close()
    readfile.close()



# def build_zeroshot_test_dev_set():
#     readfile = codecs.open(path+'sf_all_labeled_data_multilabel.txt', 'r', 'utf-8')
#     writefile_test = codecs.open(path+'zero-shot-split/test.txt', 'w', 'utf-8')
#     writefile_dev = codecs.open(path+'zero-shot-split/dev.txt', 'w', 'utf-8')
#     writefile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'w', 'utf-8')
#     label_set = set(['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange'])
#
#     for line in readfile:
#         parts=line.strip().split('\t')
#         label_list = parts[0].strip().split()
#         if len(label_list) == 1 and label_list[0] == 'out-of-domain':










if __name__ == '__main__':
    # combine_all_available_labeled_datasets()
    split_all_labeleddata_into_subdata_per_label()
