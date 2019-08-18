
import codecs
from collections import defaultdict

path = '/export/home/Dataset/LORELEI/'

def combine_all_available_labeled_datasets():

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



def build_zeroshot_test_dev_set():

    test_label_size_max = {'search':80, 'evac':70, 'infra':120, 'utils':100,'water':120,'shelter':175,
    'med':250,'food':190,'regimechange':30,'terrorism':70,'crimeviolence':250,'out-of-domain':400}
    dev_label_size_max = {'search':50, 'evac':30, 'infra':50, 'utils':50,'water':50,'shelter':75,
    'med':100,'food':80,'regimechange':15,'terrorism':40,'crimeviolence':100,'out-of-domain':200}

    label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange', 'out-of-domain']

    test_store_size = defaultdict(int)
    dev_store_size = defaultdict(int)
    write_test = codecs.open(path+'zero-shot-split/test.txt', 'w', 'utf-8')
    write_dev = codecs.open(path+'zero-shot-split/dev.txt', 'w', 'utf-8')
    writefile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'w', 'utf-8')
    for label in label_list:
        readfile = codecs.open(path+'data_per_label/'+label+'.txt', 'r', 'utf-8')
        for line in readfile:
            if test_store_size.get(label, 0) < test_label_size_max.get(label):
                write_test.write(label+'\t'+line.strip()+'\n')
                test_store_size[label]+=1
            elif dev_store_size.get(label, 0) < dev_label_size_max.get(label):
                write_dev.write(label+'\t'+line.strip()+'\n')
                dev_store_size[label]+=1
            else:
                writefile_remain.write(label+'\t'+line.strip()+'\n')
        readfile.close()
    write_test.close()
    write_dev.close()
    writefile_remain.close()

    print('test and dev build over')


def build_zeroshot_train_set():
    readfile_remain = codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
    '''we do not put None type in train'''
    label_list = ['search','evac','infra','utils','water','shelter','med','food', 'crimeviolence', 'terrorism', 'regimechange']
    writefile_PU_half_0 = codecs.open(path+'zero-shot-split/train_pu_half_v0.txt', 'w', 'utf-8')
    writefile_PU_half_1 = codecs.open(path+'zero-shot-split/train_pu_half_v1.txt', 'w', 'utf-8')

    for line in readfile_remain:
        parts = line.strip().split('\t')
        type = parts[0]
        if type in set(label_list):
            if label_list.index(type) %2==0:
                writefile_PU_half_0.write(line.strip()+'\n')
            else:
                writefile_PU_half_1.write(line.strip()+'\n')
    writefile_PU_half_0.close()
    writefile_PU_half_1.close()
    print('PU half over')
    '''PU_one'''
    for i in range(len(label_list)):
        readfile=codecs.open(path+'unified-dataset-wo-devandtest.txt', 'r', 'utf-8')
        writefile_PU_one = codecs.open(path+'zero-shot-split/train_pu_one_'+'wo_'+str(i)+'.txt', 'w', 'utf-8')
        line_co=0
        for line in readfile:
            parts = line.strip().split('\t')
            type = parts[0]
            if type in set(label_list):
                label_id = label_list.index(type)
                if label_id != i:
                    writefile_PU_one.write(line.strip()+'\n')
                    line_co+=1
        writefile_PU_one.close()
        readfile.close()
        print('write size:', line_co)
    print('build train over')











if __name__ == '__main__':
    # combine_all_available_labeled_datasets()
    # split_all_labeleddata_into_subdata_per_label()
    # build_zeroshot_test_dev_set()
    build_zeroshot_train_set()
