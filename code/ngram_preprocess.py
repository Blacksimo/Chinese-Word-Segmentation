import io

def create_ngram(file_path, unigram=True, bigram=True):
    with io.open(file_path, 'r', encoding='utf8') as _file, io.open(file_path+'.unigram', 'w', encoding='utf8') as _unigram, io.open(file_path+'.bigram', 'w', encoding='utf8') as _bigram:
        for cnt, line in enumerate(_file):
            char_list = list()
            temp = list()
            for char in line.strip().replace('  ',''):
                char_list.append(char)
            if bigram:
                #print('Creating Bigram File, line ',cnt)
                for i, el in enumerate(char_list):
                    bigram_tmp = ''
                    if i == len(char_list)-1:
                        break
                    bigram_tmp = str(char_list[i])+str(char_list[i+1])
                    #print(bigram)
                    _bigram.write(bigram_tmp+' ')
                _bigram.write('\n')
            """ if cnt == 10:
                break """
            line = line.strip().split()
            if unigram:
                #print('Creating Unigram File, line ',cnt)
                for word in line:
                    for char in word:
                        _unigram.write(char+' ')
                _unigram.write('\n')
    _file.close()
    _unigram.close()
    _bigram.close()

create_ngram('../../Documents/NLP/resources/icwb2-data/training/msr_training.utf8')