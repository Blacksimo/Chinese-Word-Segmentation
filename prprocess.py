import io

with io.open('msr_training.utf8', 'r', encoding='utf8') as _file, io.open('msr_training.monogram', 'w', encoding='utf8') as _monogram, io.open('msr_training.bigram', 'w', encoding='utf8') as _bigram:
    for cnt, line in enumerate(_file):
        char_list = list()
        temp = list()
        for char in line.strip().replace('  ',''):
            char_list.append(char)
        for i, el in enumerate(char_list):
            bigram = ''
            if i == len(char_list)-2:
                break
            bigram = str(char_list[i])+str(char_list[i+1])
            #print(bigram)
            _bigram.write(bigram+' ')
        _bigram.write('\n')
        if cnt == 10:
            break
        line = line.strip().split()
        for word in line:
            for char in word:
                _monogram.write(char+' ')
        _monogram.write('\n')
_file.close()
_monogram.close()
_bigram.close()