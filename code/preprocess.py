from os import listdir, mkdir
from os.path import exists
import io
from sklearn import preprocessing
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Bidirectional, Dropout, LSTM, Input, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.layers import concatenate
from time import time


VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 50
MAX_NB_WORDS = 100
MAX_SEQUENCE_LENGTH = 32
folders = ['gold', 'testing', 'training']
path = '../../Documents/NLP/resources/icwb2-data/'
path2 = '../../Documents/NLP/resources/preprocessed-data/'
LABELS = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
SAVE_FILE_PATH = path2 + 'savez/first_try.npz'
EMBEDDING_FILE_PATH = path2 + 'vocab/embedding_file_small'


def preprocess_data():
    file_list = [fold+'/'+f for fold in folders for f in listdir(
        path+fold) if f.startswith('msr') and f.endswith('utf8')]
    print(file_list)
    mkdir(path2) if not exists(path2) else None
    for f in file_list:
        word_counter = 0
        file_name = f.split('/')[1]
        folder_name = f.split('/')[0]
        print('Preprocessing file: ', file_name)
        mkdir(path2+folder_name) if not exists(path2+folder_name) else None
        with io.open(path+f, 'r', encoding='utf8') as input_file, io.open(path2+folder_name+'/'+file_name+'.out', 'w', encoding='utf8') as output_file, io.open(path2+folder_name+'/'+file_name+'.labels', 'w', encoding='utf8') as labels_file:
            for cnt, line in enumerate(input_file):
                parse_label = ''
                parse_word = ''
                line = line.split('\x20\x20')
                for word in line:
                    word_counter += 1
                    individual_label = ''
                    word = word.strip()
                    if len(word) == 1:
                        individual_label += 'S'
                    else:
                        individual_label += 'B'+'I'*(len(word)-2)+'E'
                    parse_word += word
                    parse_label += individual_label
                # The last line will remain blank
                output_file.write(parse_word+'\n')
                labels_file.write(parse_label+'\n')
        input_file.close()
        output_file.close()
        labels_file.close()
        print('{} Words and {} lines, in file {}'.format(
            word_counter, cnt, file_name))
    print('Done')


# preprocess_data()

# Normalize the training data, and scale the testing data using the training data weights


def normalize_data(train_path, test_path):
    X_train, X_test, y_train, y_test = list(), list(), list(), list()
    for _file in listdir(train_path):
        if _file.endswith('.out'):
            with io.open(train_path+_file, 'r', encoding='utf8') as _file:
                for cnt, line in enumerate(_file):
                    if cnt == 3:
                        break
                    X_train.append([char for char in line.strip()])
                #X_train = [[line.strip()] for line in _file]
            _file.close()
        else:
            with io.open(train_path+_file, 'r', encoding='utf8') as _file:
                for cnt, line in enumerate(_file):
                    if cnt == 3:
                        break
                    label_data = []
                    for label in line.strip():
                        label_data.append(LABELS[label])
                    y_train.append(label_data)
            _file.close()
    for _file in listdir(test_path):
        if _file.endswith('.out'):
            with io.open(test_path+_file, 'r', encoding='utf8') as _file:
                for cnt, line in enumerate(_file):
                    if cnt == 3:
                        break
                    X_test.append([char for char in line.strip()])
                #X_test = [[line.strip()] for line in _file]
            _file.close()
        else:
            with io.open(test_path+_file, 'r', encoding='utf8') as _file:
                for cnt, line in enumerate(_file):
                    if cnt == 3:
                        break
                    label_data = []
                    for label in line.strip():
                        label_data.append(LABELS[label])
                    y_test.append(label_data)
            _file.close()
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

#normalize_data(path2+'training/', path2+'testing/')


def get_embedding_matrices(embedding_file):
    embedding_matrix = list()
    embedding_vocab = dict()
    with io.open(embedding_file, 'r', encoding='utf8') as _file:
        for cnt, line in enumerate(_file):
            if cnt == 0 or cnt == 1:
                continue
            word = line.split()[0]
            values = line.split()[1:]
            if word not in embedding_vocab:
                embedding_vocab[word] = len(embedding_vocab)
            embedding_matrix.append(values)
    _file.close()
    # print(embedding_matrix)

# get_embedding_matrices(path2+'vocab/embedding_file')


def devo_provarla(file_path):
    if not exists(SAVE_FILE_PATH):
        labels = list()
        labels_onehot = list()
        texts = list()
        embedding_index = dict()
        with io.open(file_path + '.bigram', 'r', encoding='utf8') as _file:
            for line in _file:
                line = line.strip()
                texts.append(line)
        _file.close()
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', data.shape)
        with io.open(file_path + '.labels', 'r', encoding='utf8') as label_file:
            for line in label_file:
                label_code = list()
                for char in line.strip():
                    label_code.append(LABELS[char])
                labels.append(label_code)
        #labels = pad_sequences(labels, maxlen=MAX_SEQUENCE_LENGTH, value=-1)
        label_file.close()
        #labels = np.asarray(labels)
        #print('Shape of label tensor:', labels.shape)
        labels_onehot = np.zeros((data.shape[0], data.shape[1], 4))
        #print(data[72193], labels[72193])
        print('diobono')
        for i, line in enumerate(data):
            zero_count = 0
            for j, word in enumerate(line):
                if data[i][j] != 0:
                    try:
                        labels_onehot[i][j][labels[i][j-zero_count]] = 1
                    except Exception as e:
                        print('Error: ',e)
                else:
                    zero_count += 1
        #labels_onehot = np.asarray(labels_onehot)
        with io.open(EMBEDDING_FILE_PATH, 'r', encoding='utf8') as _vocab:
            for line in _vocab:
                line = line.split()
                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')
                embedding_index[word] = coefs
        _vocab.close()
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i == 0:
                print(word, i)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        # print('ciaone')

        # split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels_onehot = labels_onehot[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        word_index_len = len(word_index)

        x_train = data[:-nb_validation_samples]
        y_train = labels_onehot[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels_onehot[-nb_validation_samples:]

        print('Saving npz file')
        np.savez(SAVE_FILE_PATH, x_train=x_train,
                 y_train=y_train, x_val=x_val, y_val=y_val,
                 word_index_len=word_index_len, embedding_matrix=embedding_matrix)

    else:
        print('Loading npz file')
        res = np.load(SAVE_FILE_PATH)
        x_train = res['x_train']
        y_train = res['y_train']
        x_val = res['x_val']
        y_val = res['y_val']
        word_index_len = res['word_index_len']
        embedding_matrix = res['embedding_matrix']
        del res
    """ y_train = np.zeros((69540, 4))
    y_val = np.zeros((17384, 4)) """

    embedding_unigram = Embedding(word_index_len + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    unigram_model = Sequential(name='Unigram')
    unigram_model.add(embedding_unigram)
    
    unigram_input = Input(shape=(32,))

    #unigram_input = Input(shape=(x_train.shape[1]))
    #unigram_input = unigram_model(unigram_input)

    embedding_bigram = Embedding(word_index_len + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    bigram_model = Sequential(name='Bigram')
    bigram_model.add(embedding_bigram)
    
    bigram_input = Input(shape=(32,))

    #bigram_input = Input(shape=(x_train.shape[1]))
    #bigram_input = bigram_model(bigram_input)

    #model = Sequential()
    #model.add(Dense(1, input_shape=(32,50,)))
    model = concatenate([embedding_unigram(unigram_input), embedding_bigram(bigram_input)])
    model = (Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1), input_shape=(MAX_SEQUENCE_LENGTH,)))(model)
    model = (Dense(units=4, activation='softmax'))(model)
    sgd = SGD(lr=0.04, momentum=0.95)
    model = Model([unigram_input, bigram_input], model)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit([x_train, x_train], y_train, validation_data=([x_val, x_val], y_val),
              epochs=3, batch_size=32, callbacks=[tensorboard])
    model.save(path2+'savez/first.h5')

    ##############################################

    """ banana = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 75, 33, 270, 25, 208, 3, 364, 75, 24, 36, 29, 78, 219, 14, 340, 30, 363, 371, 4]]
    banana = np.array(banana)
    model.load_weights(path2+'savez/first.h5')
    print(model.predict(banana)) """

devo_provarla('../../Documents/NLP/resources/preprocessed-data/training/msr_training.utf8')
