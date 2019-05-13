# Chinese Word Segmentation

Implementation of the state-of-the-art chinese word segmenter model in Keras.
[Link to the refernce paper](https://arxiv.org/pdf/1808.06511.pdf)

## The Model
A Stacked bidirectional LSTM which takes as input a set of strings of chinese characters (splitted in unigram and bigram) and embedding matrices for both unigram and bigram. The two n-gram are concatenated before feeding the model.

## The Dataset
The dataset used as corpus for the training and the embedding matrices is the Microsoft Reaserch (MSR) training set.
[Link to the dataset](http://sighan.cs.uchicago.edu/bakeoff2005/)

## Results
Results values and graph are shown in the [Report Paper](blob/master/paper.pdf)