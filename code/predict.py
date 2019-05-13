from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    labels = list()
    file_path = resources_path + 'training/msr_training.utf8'
    INPUTS = ['unigram', 'bigram']
    LABELS = ['B','I','E','S']
    word_index = dict()
    for input_name in INPUTS:
      texts = list()
      with io.open(file_path + '.' + input_name, 'r', encoding='utf8') as input_file:
          for line in input_file:
              line = line.strip()
              texts.append(line)
      input_file.close()
      tokenizer = Tokenizer(oov_token='UNK')
      tokenizer.fit_on_texts(texts)
      sequences = tokenizer.texts_to_sequences(texts)
      word_index[input_name] = tokenizer.word_index
      print('Found %s unique tokens.' % len(word_index[input_name]))
      data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
      print('Shape of data tensor:', data.shape)

    """ resources_path = path2 
    input_path= path2 + 'gold/msr_test_gold.utf8.out'
    output_path = path2 + 'savez/msr_gold_test_predicted' """  
    input_line_count = list()

    model = load_model(resources_path + 'savez/weights.32-0.95.hdf5')
    #print(model.summary())
    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
      text = list()
      bigram_text = list()
      for cnt, line in enumerate(input_file):
        line = line.strip()
        input_line_count.append(len(line))
        bigram_line = list()
        text.append(line)
        for i, char in enumerate(line):
          if i == len(line)-1:
            break
          bigram_line.append(str(line[i])+str(line[i+1]))
        bigram_text.append(bigram_line)
      for input_name in INPUTS:
        if input_name == 'unigram':
          tokenizer = Tokenizer(oov_token='UNK', char_level=True)
          tokenizer.word_index = word_index[input_name]
          text = tokenizer.texts_to_sequences(text)
          text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)
        else:
          tokenizer = Tokenizer(oov_token='UNK')
          tokenizer.word_index = word_index[input_name]
          bigram_text = tokenizer.texts_to_sequences(bigram_text)
          bigram_text = pad_sequences(bigram_text, maxlen=MAX_SEQUENCE_LENGTH)
      text = np.asarray(text)
      bigram_text = np.asarray(bigram_text)
      prediction = model.predict([text, bigram_text])
      for cnt, pred_line in enumerate(prediction):
        if input_line_count[cnt] < MAX_SEQUENCE_LENGTH:
          match_index = range(MAX_SEQUENCE_LENGTH - input_line_count[cnt], MAX_SEQUENCE_LENGTH)
          flag = False
        else:
          match_index = range(MAX_SEQUENCE_LENGTH)
          flag = True
        label_line = list()
        for cnt2, pred_label in enumerate(pred_line):
          if cnt2 in match_index:
            dic_value = np.where( pred_label == max(pred_label))[0][0]
            label_line.append(LABELS[dic_value])
        if flag:
          for i in range(input_line_count[cnt] - MAX_SEQUENCE_LENGTH):
            label_line.append('S')
        print(label_line)
        final_string = ''
        final_string.join(label_line)
        output_file.write(final_string+'\n')
    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
