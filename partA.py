import pandas as pd
import ast
from collections import Counter
import math
import re
import csv
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
from gensim.models import Word2Vec
from torchsummary import summary
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk import ngrams


EMBED_DIM = 32
HIDDEN_DIM = 16
w2v = 'word2vec' + str(EMBED_DIM) + '.model'
WORD2VEC = Word2Vec.load(w2v)
NAME = 'NewShoes'

# Full: GRU, No Dropout, EMBED_DIM = 32, HIDDEN_DIM = 16, LAYERS = 1, RMSProp, LR=3e-4, +All Features
# Baseline: GRU, No Dropout, EMBED_DIM = 32, HIDDEN_DIM = 16, LAYERS = 1, RMSProp, LR=3e-4, +Lengths

LAYERS = 1
EPOCHS = 1000
LR = 3e-4
DEVICE = torch.device('cpu')

CURRENT = os.curdir
MODELS = os.path.join(CURRENT, 'experimental_models')
PATH = os.path.join(MODELS, NAME)


def embed(train_data, dev_data, test_data):
    train_data = train_data.to_numpy()
    dev_data = dev_data.to_numpy()
    test_data = test_data.to_numpy()

    training_data = []
    training_data_prime = [[],[]]
    train_omega = []
    for row1 in train_data:
        sample1 = [[],[]]
        pos_lst = []
        neg_lst = []
        pvec_lst = []
        nvec_lst = []
        for sentence1 in range(1, 5):
            lst = [WORD2VEC.wv[word1] for word1 in row1[sentence1].split()]
            if sentence1 < 4:
                lst1 = lst
                lst2 = [WORD2VEC.wv[word1] for word1 in row1[sentence1+1].split()]
                lst1 = np.stack(lst1, axis=0)
                lst2 = np.stack(lst2, axis=0)
                lst1 = np.expand_dims(lst1, axis=0)
                lst2 = np.expand_dims(lst2, axis=0)
                train_omega.append(((torch.from_numpy(lst1).to(DEVICE), torch.from_numpy(lst2).to(DEVICE)), 1))
            sample1[0] += lst
            sample1[1] += lst
            pos_lst.append(row1[sentence1].split())
            neg_lst.append(row1[sentence1].split())
            pvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[sentence1].split()])
            nvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[sentence1].split()])
        if row1[7] == 1:
            sample1[0] += [WORD2VEC.wv[word1] for word1 in row1[5].split()]
            sample1[1] += [WORD2VEC.wv[word1] for word1 in row1[6].split()]
            pos_lst.append(row1[5].split())
            neg_lst.append(row1[6].split())
            pos_last = [WORD2VEC.wv[word1] for word1 in row1[5].split()]
            neg_last = [WORD2VEC.wv[word1] for word1 in row1[6].split()]
            pvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[5].split()])
            nvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[6].split()])
            pos1 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[4].split()], axis=0), axis=0)).to(DEVICE)
            pos2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[5].split()], axis=0), axis=0)).to(DEVICE)
            neg1 = pos1
            neg2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[6].split()], axis=0), axis=0)).to(DEVICE)
            train_omega.append(((pos1, pos2), 1))
            train_omega.append(((neg1, neg2), 0))
        elif row1[7] == 2:
            sample1[0] += [WORD2VEC.wv[word1] for word1 in row1[6].split()]
            sample1[1] += [WORD2VEC.wv[word1] for word1 in row1[5].split()]
            pos_lst.append(row1[6].split())
            neg_lst.append(row1[5].split())
            pos_last = [WORD2VEC.wv[word1] for word1 in row1[6].split()]
            neg_last = [WORD2VEC.wv[word1] for word1 in row1[5].split()]
            pvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[6].split()])
            nvec_lst.append([WORD2VEC.wv[word1] for word1 in row1[5].split()])
            pos1 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[4].split()], axis=0), axis=0)).to(DEVICE)
            pos2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[6].split()], axis=0), axis=0)).to(DEVICE)
            neg1 = pos1
            neg2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word1] for word1 in row1[5].split()], axis=0), axis=0)).to(DEVICE)
            train_omega.append(((pos1, pos2), 1))
            train_omega.append(((neg1, neg2), 0))
        pos = np.stack(sample1[0], axis=0)
        neg = np.stack(sample1[1], axis=0)
        pos = np.expand_dims(pos, axis=0)
        neg = np.expand_dims(neg, axis=0)
        pos_last = np.stack(pos_last, axis=0)
        pos_last = np.expand_dims(pos_last, axis=0)
        neg_last = np.stack(neg_last, axis=0)
        neg_last = np.expand_dims(neg_last, axis=0)
        train1 = (torch.from_numpy(pos).to(DEVICE), pos_lst, torch.from_numpy(pos_last).to(DEVICE), pvec_lst)
        train2 = (torch.from_numpy(neg).to(DEVICE), neg_lst, torch.from_numpy(neg_last).to(DEVICE), nvec_lst)
        # training_data.append((train1, train2))
        training_data.append((train1, 1))
        training_data.append((train2, 0))
        training_data_prime[0].append((torch.from_numpy(pos).to(DEVICE), pos_lst))
        training_data_prime[1].append((torch.from_numpy(neg).to(DEVICE), neg_lst))

    development_data = []
    development_data_prime = [[],[]]
    dev_omega = []
    for row2 in dev_data:
        sample2 = [[],[]]
        v1_lst = []
        v2_lst = []
        vec_lst1 = []
        vec_lst2 = []
        for sentence2 in range(1, 5):
            lst = [WORD2VEC.wv[word2] for word2 in row2[sentence2].split()]
            sample2[0] += lst
            sample2[1] += lst
            v1_lst.append(row2[sentence2].split())
            v2_lst.append(row2[sentence2].split())
            vec_lst1.append([WORD2VEC.wv[word2] for word2 in row2[sentence2].split()])
            vec_lst2.append([WORD2VEC.wv[word2] for word2 in row2[sentence2].split()])
        sample2[0] += [WORD2VEC.wv[word2] for word2 in row2[5].split()]
        sample2[1] += [WORD2VEC.wv[word2] for word2 in row2[6].split()]
        v1_lst.append(row2[5].split())
        v2_lst.append(row2[6].split())
        v1_last = [WORD2VEC.wv[word2] for word2 in row2[5].split()]
        v2_last = [WORD2VEC.wv[word2] for word2 in row2[6].split()]
        vec_lst1.append([WORD2VEC.wv[word2] for word2 in row2[5].split()])
        vec_lst2.append([WORD2VEC.wv[word2] for word2 in row2[6].split()])
        v1 = np.stack(sample2[0], axis=0)
        v2 = np.stack(sample2[1], axis=0)
        v1 = np.expand_dims(v1, axis=0)
        v2 = np.expand_dims(v2, axis=0)
        if row2[7] == 1:
            pos_lst = v1_lst
            neg_lst = v2_lst
            pos1 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[4].split()], axis=0), axis=0)).to(DEVICE)
            pos2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[5].split()], axis=0), axis=0)).to(DEVICE)
            neg1 = pos1
            neg2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[6].split()], axis=0), axis=0)).to(DEVICE)
            pos_pair = (pos1, pos2)
            neg_pair = (neg1, neg2)
            dev_omega.append((pos_pair, neg_pair, 0))
        else:
            pos_lst = v2_lst
            neg_lst = v1_lst
            pos1 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[4].split()], axis=0), axis=0)).to(DEVICE)
            pos2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[6].split()], axis=0), axis=0)).to(DEVICE)
            neg1 = pos1
            neg2 = torch.from_numpy(np.expand_dims(np.stack([WORD2VEC.wv[word2] for word2 in row2[5].split()], axis=0), axis=0)).to(DEVICE)
            pos_pair = (pos1, pos2)
            neg_pair = (neg1, neg2)
            dev_omega.append((pos_pair, neg_pair, 1))
        v1_last = np.stack(v1_last, axis=0)
        v1_last = np.expand_dims(v1_last, axis=0)
        v2_last = np.stack(v2_last, axis=0)
        v2_last = np.expand_dims(v2_last, axis=0)
        val1 = (torch.from_numpy(v1).to(DEVICE), v1_lst, torch.from_numpy(v1_last).to(DEVICE), vec_lst1)
        val2 = (torch.from_numpy(v2).to(DEVICE), v2_lst, torch.from_numpy(v2_last).to(DEVICE), vec_lst2)
        development_data.append((val1, val2, row2[7]-1))
        development_data_prime[0].append(pos_lst)
        development_data_prime[1].append(neg_lst)

    testing_data = []
    for row3 in test_data:
        sample3 = [[],[]]
        t1_lst = []
        t2_lst = []
        for sentence3 in range(1, 5):
            lst = [WORD2VEC.wv[word3] for word3 in row3[sentence3].split()]
            sample3[0] += lst
            sample3[1] += lst
            t1_lst.append(row3[sentence3].split())
            t2_lst.append(row3[sentence3].split())
        sample3[0] += [WORD2VEC.wv[word3] for word3 in row3[5].split()]
        sample3[1] += [WORD2VEC.wv[word3] for word3 in row3[6].split()]
        t1_lst.append(row3[5].split())
        t2_lst.append(row3[6].split())
        t1 = np.stack(sample3[0], axis=0)
        t2 = np.stack(sample3[1], axis=0)
        t1 = np.expand_dims(t1, axis=0)
        t2 = np.expand_dims(t2, axis=0)
        testing_data.append(((torch.from_numpy(t1).to(DEVICE), t1_lst), (torch.from_numpy(t2).to(DEVICE), t2_lst), row3[0]))

    return training_data, development_data, testing_data, training_data_prime, development_data_prime, train_omega, dev_omega

    # model = Word2Vec(total_sentences, size=EMBED_DIM, min_count=1)
    # name = 'word2vec' + str(EMBED_DIM) + '.model'
    # model.save(name)

def replace_sparse_words(train_data, dev_data, test_data):

    train_data = train_data.to_numpy()
    dev_data = dev_data.to_numpy()
    test_data = test_data.to_numpy()
    # Count words

    seen_vocab = {}
    for row in train_data:
        for i in range(1, 7):
            for word in row[i].split():
                if seen_vocab.get(word) is None:
                    seen_vocab[word] = 1
                else:
                    seen_vocab[word] += 1
    for row in dev_data:
        for i in range(1, 7):
             for word in row[i].split():
                if seen_vocab.get(word) is None:
                    seen_vocab[word] = 1
                else:
                    seen_vocab[word] += 1
    for row in test_data:
        for i in range(1, 7):
             for word in row[i].split():
                if seen_vocab.get(word) is None:
                    seen_vocab[word] = 1
                else:
                    seen_vocab[word] += 1

    # Replace words
    new_train = []
    for row in train_data:
        new_pos = [row[0]]
        new_neg = [row[0]]
        for i in range(1, 5):
            new_sentence = []
            for word in row[i].split():
                new_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
            new_pos.append(new_sentence)
            new_neg.append(new_sentence)
        pos_sentence = []
        neg_sentence = []
        if row[7] == 1:
            for word in row[5].split():
                pos_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
            for word in row[6].split():
                neg_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
        else:
            for word in row[6].split():
                pos_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
            for word in row[5].split():
                neg_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
        new_pos.append(pos_sentence)
        new_neg.append(neg_sentence)
        new_pos.append(row[7]-1)
        new_neg.append(row[7]-1)
        new_train.append(new_pos)
        new_train.append(new_neg)

    new_dev = []
    for row in dev_data:
        new_sample = [row[0]]
        for i in range(1, 7):
            new_sentence = []
            for word in row[i].split():
                new_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
            new_sample.append(new_sentence)
        new_sample.append(row[7]-1)
        new_dev.append(new_sample)

    new_test = []
    for row in test_data:
        new_sample = [row[0]]
        for i in range(1, 7):
            new_sentence = []
            for word in row[i].split():
                new_sentence.append(word if seen_vocab[word] > 3 else '<UNK>')
            new_sample.append(new_sentence)
        new_test.append(new_sample)

    return new_train, new_dev, new_test


def get_training_word_ngrams(data):
    sentence_ngrams = {}
    for sample in data:
        for sentence_index in range(1, 6):
            for n in range(1, 6):
                ngram_lst = list(ngrams(sample[sentence_index], n))
                for i in range(len(ngram_lst)):
                    ngram = ngram_lst[i]
                    if sentence_ngrams.get(ngram) is None:
                        sentence_ngrams[ngram] = 1
                    else:
                        sentence_ngrams[ngram] += 1
    copy = list(sentence_ngrams.keys())
    for ngram_key in copy:
        if sentence_ngrams[ngram_key] < 5:
            del sentence_ngrams[ngram_key]

    return sentence_ngrams.keys()


train_data = pd.read_csv('./train.csv', encoding='latin-1')
dev_data = pd.read_csv('./dev.csv', encoding='latin-1')
test_data = pd.read_csv('./test.csv', encoding='latin-1')

training_data, development_data, testing_data, training_data_prime, development_data_prime, train_omega, dev_omega = embed(train_data, dev_data, test_data)

new_train, new_dev, new_test = replace_sparse_words(train_data, dev_data, test_data)


def check_for_unk_words(wordlist, tokenlist):
    for i, token in enumerate(wordlist):
        if token not in tokenlist:
              wordlist[i] = '<UNK>'
    return wordlist


def get_unigram_corpus(wordlist):
    return dict(Counter(wordlist))


def get_bigram_corpus(wordlist):
    corpus = {}
    for i, word in enumerate(wordlist[1:], start=1):
        if word != '<s>':
            if (wordlist[i-1], word) not in corpus:
                corpus[(wordlist[i-1], word)] = 1
            else:
                corpus[(wordlist[i-1], word)] += 1
    return corpus


def get_smooth_bigram_corpus(tokenlist, bigram_corpus):
    tokenlist.append('<UNK>')
    df = pd.DataFrame(1, index = tokenlist, columns = tokenlist)
    for bigram in bigram_corpus:
        df.loc[bigram[0], bigram[1]] += bigram_corpus[bigram]
    return df


def get_smooth_bigram_prob(bigram, smooth_bigram_corpus):
    return smooth_bigram_corpus.loc[bigram[0], bigram[1]] / smooth_bigram_corpus.loc[bigram[0]].sum()


class NGramModel():
    def __init__(self, *args):
        super(NGramModel, self).__init__()

    def get_perp(self, *args):
        return


class SmoothBigramModel(NGramModel):
    def __init__(self, data):
        super(SmoothBigramModel, self).__init__()
        data = self.preprocess(data)
        self.tokens = list(get_unigram_corpus(data).keys())
        corpus = get_bigram_corpus(data)
        self.corpus = get_smooth_bigram_corpus(self.tokens, corpus)

    def preprocess(self, data):
        total = []
        for sample in data:
            sentences = sample[1]
            for sentence in sentences:
                total += sentence
        return total

    def get_perp(self, sentences):
        sentences = check_for_unk_words(sentences, self.tokens)
        N = len(sentences)
        acc = 0
        for i, word in enumerate(sentences):
            if i == 0:
                continue
            bigram = (sentences[i-1], word)
            acc -= math.log(get_smooth_bigram_prob(bigram, self.corpus))
        return math.exp(1/(N-1) * acc)


class bigram_nb_classifier():
    def __init__(self, data):
        truthful_train = data[0]
        false_train = data[1]
        truthful_pairs = self.preprocess(truthful_train)
        false_pairs = self.preprocess(false_train)
        truthful_data = [(self.create_bigram_features(pair), 1) for pair in truthful_pairs]
        false_data = [(self.create_bigram_features(pair), 0) for pair in false_pairs]
        self.classifier = NaiveBayesClassifier.train(false_data + truthful_data)
        self.latest_accuracy = -1

    def compute_accuracy(self, val_data):
        truthful_val = val_data[0]
        false_val = val_data[1]
        truthful_pairs = [sentences[3] + sentences[4] for sentences in truthful_val]
        false_pairs = [sentences[3] + sentences[4] for sentences in false_val]
        truthful_data_v = [(self.create_bigram_features(pair), 1) for pair in truthful_pairs]
        false_data_v = [(self.create_bigram_features(pair), 1) for pair in false_pairs]
        self.latest_accuracy = nltk.classify.util.accuracy(self.classifier, truthful_data_v + false_data_v )
        return self.latest_accuracy

    def create_bigram_features(self, words):
        ngram_vocab = ngrams(words, 2)
        my_dict = dict([(ng, True) for ng in ngram_vocab])
        return my_dict

    def preprocess(self, tuple_lst):
        story_lst = []
        for sample in tuple_lst:
            story_lst.append(sample[1])
        pair_lst = [sentences[3] + sentences[4] for sentences in story_lst]
        return pair_lst

    def classify(self, pair):
        return self.classifier.classify(self.create_bigram_features(pair))

# TMODEL = SmoothBigramModel(training_data_prime[0])
# FMODEL = SmoothBigramModel(training_data_prime[1])
# BNB = bigram_nb_classifier(training_data_prime)

# print(BNB.compute_accuracy(development_data_prime))

# data = development_data
# random.shuffle(data)
# correct = 0
# total = 0
# N = len(data)
# for index in tqdm(range(N)):
#     input_1, input_2, gold_label = data[index]
#     pair_1 = input_1[1][3] + input_1[1][4]
#     pair_2 = input_1[1][3] + input_1[1][4]
#     prob_truthful_1 = 1 / TMODEL.get_perp(pair_1)
#     prob_false_1 = 1 / FMODEL.get_perp(pair_1)
#     prob_truthful_2 = 1 / TMODEL.get_perp(pair_2)
#     prob_false_2 = 1 / FMODEL.get_perp(pair_2)
#     softmax = nn.Softmax(dim=0)
#     probs = torch.tensor([prob_truthful_1, prob_false_1, prob_truthful_2, prob_false_2], device=DEVICE)
#     probs = softmax(probs)
#     max_index = torch.argmax(probs)
#     if max_index == 0 or max_index == 3:
#         predicted_label = 0
#     if max_index == 1 or max_index == 2:
#         predicted_label = 1
#     correct += int(predicted_label == gold_label)
#     total += 1
# print(correct / total)



class MatchDotCOMP(nn.Module):
    def __init__(self):
        super(MatchDotCOMP, self).__init__()
        # self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(HIDDEN_DIM*2, 2)
        # self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        # print('MatchDotCOMP: predicted = {}, gold = {}'.format(predicted_vector, gold_label))
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        h_0 = torch.zeros((LAYERS*2, 1, HIDDEN_DIM), device=DEVICE)
        output, __ = self.gru(inputs, h_0)
        # c_0 = h_0.clone()
        # output, __ = self.lstm(inputs, (h_0, c_0))
        x = output[0][-1]
        x = self.linear(x)
        # x = self.softmax(x)
        return x


class Groot(nn.Module):
    def __init__(self):
        super(Groot, self).__init__()
        # self.lstm1 = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        # self.lstm2 = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        self.gru1 = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(HIDDEN_DIM*2, 2)
        # self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        # print('Groot: predicted = {}, gold = {}'.format(predicted_vector, gold_label))
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        input1 = []
        for i in range(4):
            input1 += inputs[i]
        input1 = np.stack(input1, axis=0)
        input1 = np.expand_dims(input1, axis=0)
        input1 = torch.from_numpy(input1).to(DEVICE)
        h_0 = torch.zeros((LAYERS*2, 1, HIDDEN_DIM), device=DEVICE)
        # c_0 = h_0.clone()
        # __, hcn = self.lstm1(input1, (h_0, c_0))
        __, h_n = self.gru1(input1, h_0)
        input2 = np.expand_dims(inputs[4], axis=0)
        input2 = torch.from_numpy(input2).to(DEVICE)
        # h_n = hcn[0]
        # c_n = hcn[1]
        # output, __ = self.lstm2(input2, (h_n, c_n))
        output, __ = self.gru2(input2, h_n)
        x = output[0][-1]
        x = self.linear(x)
        # x = self.softmax(x)
        return x


class Blind(nn.Module):
    def __init__(self):
        super(Blind, self).__init__()
        # self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(HIDDEN_DIM*2, 2)
        # self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        # print('Blind: predicted = {}, gold = {}'.format(predicted_vector, gold_label))
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        h_0 = torch.zeros((LAYERS*2, 1, HIDDEN_DIM), device=DEVICE)
        # c_0 = h_0.clone()
        # output, __ = self.lstm(inputs, (h_0, c_0))
        output, __ = self.gru(inputs, h_0)
        x = output[0][-1]
        x = self.linear(x)
        # x = self.softmax(x)
        return x


FEATURES = 5
# 1: length of last sentence
# 2: truthful bigram perplexity of last two sentences
# 3: false bigram perplexity of last two sentences
# 4: bigram naive bayes truthful/false classification of last two sentences
# 5: bigram overlap between last sentence and previous four


class LMFeatureExtractor(nn.Module):
    def __init__(self):
        super(LMFeatureExtractor, self).__init__()
        # self.linear = nn.Linear(FEATURES, 2)
        # self.softmax = nn.LogSoftmax(dim=0)
        # self.cuda(device=DEVICE)
        self.tmodel = TMODEL
        self.fmodel = FMODEL
        self.bnb = BNB

    def get_bigram_overlap(self, inputs):
        first_four = get_bigram_corpus(inputs[0]+inputs[1]+inputs[2]+inputs[3])
        last_one = get_bigram_corpus(inputs[4])
        total = 0
        intersect = 0
        for bigram in last_one:
            if bigram in first_four:
                intersect += 1
            total += 1
        overlap = intersect / total
        return overlap

    def extract_features(self, inputs):
        last_two = inputs[3] + inputs[4]
        features = []
        features.append(len(inputs[4]))
        features.append(self.tmodel.get_perp(last_two))
        features.append(self.fmodel.get_perp(last_two))
        features.append(self.bnb.classify(last_two))
        features.append(self.get_bigram_overlap(inputs))
        return torch.tensor(features, device=DEVICE, dtype=torch.float)

    def forward(self, inputs):
        x = self.extract_features(inputs)
        # x = self.linear(x)
        # return self.softmax(x)
        return x


class RNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(RNNFeatureExtractor, self).__init__()
        # self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True)
        self.linear = nn.Linear(HIDDEN_DIM, 2)
        self.softmax = nn.LogSoftmax(dim=0)
        # self.cuda(device=DEVICE)

    def forward(self, inputs):
        h_0 = torch.zeros(LAYERS, 1, HIDDEN_DIM, device=DEVICE)
        # output, __ = self.gru(inputs, h_0)
        c_0 = h_0.clone()
        output, __ = self.lstm(inputs, (h_0, c_0))
        x = output[0][-1]
        x = self.linear(x)
        x = self.softmax(x)
        return x


class RNNLM(nn.Module):
    def __init__(self):
        super(RNNLM, self).__init__()
        # self.rnn_fe = RNNFeatureExtractor()     # P(ending ^ story) ?
        self.lm_fe = LMFeatureExtractor()
        # self.blind = Blind()                    # P(ending) ?
        # self.groot = Groot()                    # P(ending | story) ?
        self.linear = nn.Linear(FEATURES, 2)
        self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=LR, momentum=0.9)
        # self.optimizer = optim.SGD(self.parameters(), lr=LR, momentum=0.9)
        # self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)
        # self.optimizer = adabound.AdaBound(self.parameters(), lr=LR, final_lr=0.01)
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        # rnn_features = self.rnn_fe(inputs[0])
        lm_features = self.lm_fe(inputs[1])
        # blind_features = self.blind(inputs[2])
        # groot_features = self.groot(inputs[3])
        # features = torch.cat((rnn_features, blind_features, groot_features), dim=0)
        # features = torch.cat((rnn_features, lm_features, blind_features, groot_features), dim=0)
        # features = torch.tensor([math.exp(blind_features[1]), math.exp(groot_features[1]), math.exp(blind_features[1]) / math.exp(groot_features[1])], device=DEVICE)
        # features = torch.cat((blind_features, groot_features), dim=0)
        x = self.linear(lm_features)
        x = self.softmax(x)
        return x

# previously: BigBand
# this was dumb
class BADBand(nn.Module):
    def __init__(self):
        super(BigBand, self).__init__()
        self.p_ending = Blind()
        self.p_ending_given_story = Groot()
        self.P_ending_and_story = MatchDotCOMP()
        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)
        self.pv1 = None
        self.pv2 = None
        self.pv3 = None
        # self.cuda(device=DEVICE)

    def compute_Loss(self, gold_label):
        loss1 = self.p_ending.compute_Loss(self.pv1.view(1, -1), gold_label)
        loss2 = self.p_ending_given_story.compute_Loss(self.pv2.view(1, -1), gold_label)
        loss3 = self.P_ending_and_story.compute_Loss(self.pv3.view(1, -1), gold_label)
        return loss1, loss2, loss3

    def forward(self, inputs):
        pv1 = self.p_ending(inputs[2])
        pv2 = self.p_ending_given_story(inputs[3])
        pv3 = self.P_ending_and_story(inputs[0])
        self.pv1 = pv1
        self.pv2 = pv2
        self.pv3 = pv3
        label1 = torch.argmax(pv1)
        label2 = torch.argmax(pv2)
        label3 = torch.argmax(pv3)
        lst = [label1, label2, label3]
        x = max(set(lst), key=lst.count)
        return x


class JAZZBand(nn.Module):
    def __init__(self):
        super(JAZZBand, self).__init__()
        self.saxophone = Blind()
        self.trumpet = Groot()
        self.drums = MatchDotCOMP()
        # self.JAZZ = nn.Linear(HIDDEN_DIM*2, 2)
        self.encore = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        sax = self.saxophone(inputs[2])
        trump = self.trumpet(inputs[3])
        dr = self.drums(inputs[0])
        x = sax + trump + dr
        # x = self.JAZZ(x)
        x = self.encore(x)
        return x


class NSP(nn.Module):
    def __init__(self):
        super(NSP, self).__init__()
        self.gru1 = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidrectional=True)
        self.gru2 = nn.GRU(EMBED_DIM, HIDDEN_DIM, LAYERS, batch_first=True, bidrectional=True)
        self.linear = nn.Linear(HIDDEN_DIM*2, 2)
        self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        h_0 = torch.zeros((LAYERS, 1, HIDDEN_DIM), device=DEVICE)
        __, h_n = self.gru1(inputs[0], h_0)
        output, __ = self.gru2(inputs[1], h_n)
        x = output[0][-1]
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.word_ngrams = get_training_word_ngrams(new_train)
        self.linear = nn.Linear(len(self.word_ngrams) + 1, 2)
        self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        # self.cuda(device=DEVICE)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.criterion(predicted_vector, gold_label)

    def extract_features(self, sentence):
        features = [len(sentence)]
        sentence_ngrams = {}
        for n in range(1, 6):
            ngram_lst = ngrams(sentence, n)
            for ngram in ngram_lst:
                if sentence_ngrams.get(ngram) is None:
                    sentence_ngrams[ngram] = 1
                else:
                    sentence_ngrams[ngram] += 1
        for key in list(sentence_ngrams.keys()):
            if key not in self.word_ngrams:
                del sentence_ngrams[key]
        for key in list(self.word_ngrams):
            features.append(sentence_ngrams.get(key, 0))
        return torch.tensor(features, device=DEVICE, dtype=torch.float)

    def forward(self, inputs):
        features = self.extract_features(inputs[4])
        x = self.linear(features)
        x = self.softmax(x)
        return x



def main():
    print('Initializing Model')
    model = Style()
    prev_dev_acc = 0.0
    for epoch in range(EPOCHS):
        checkpoint = PATH + '-e' + str((epoch + 1))
        model.train()
        model.optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print('Training started for epoch {}'.format(epoch + 1))
        random.shuffle(new_train)
        # random.shuffle(training_data)
        # random.shuffle(train_omega)
        N = len(new_train)
        # N = len(training_data)
        # N = len(train_omega)
        for index  in tqdm(range(N)):
            model.optimizer.zero_grad()
            sample = new_train[index]
            input_vector = sample[1:6]
            gold_label = sample[6]
            # input_vector, gold_label = training_data[index]
            # input_vector, gold_label = train_omega[index]
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
            loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label], device=DEVICE))
            loss.backward()
            model.optimizer.step()
        print('Training accuracy for epoch {}: {}'.format(epoch + 1, correct / total))
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(new_dev)
        # random.shuffle(development_data)
        # random.shuffle(dev_omega)
        N = len(new_dev)
        # N = len(development_data)
        # N = len(dev_omega)
        model.eval()
        model.optimizer.zero_grad()
        for index in tqdm(range(N)):
            sample = new_dev[index]
            input_1 = sample[1:6]
            input_2 = sample[1:5] + [sample[6]]
            gold_label = sample[7]
            # input_1, input_2, gold_label = development_data[index]
            # input_1, input_2, gold_label = dev_omega[index]
            prediction_1 = model(input_1)
            prediction_2 = model(input_2)
            prob_truthful_1 = prediction_1[1]
            prob_false_1 = prediction_1[0]
            prob_truthful_2 = prediction_2[1]
            prob_false_2 = prediction_2[0]
            probs = [prob_truthful_1, prob_false_1, prob_truthful_2, prob_false_2]
            max_index = probs.index(max(probs))
            if max_index == 0 or max_index == 3:
                predicted_label = 0
            if max_index == 1 or max_index == 2:
                predicted_label = 1
            correct += int(predicted_label == gold_label)
            total += 1
        dev_acc = correct / total
        if dev_acc > prev_dev_acc and dev_acc > 0.67:
            prev_dev_acc = dev_acc
            print('New Best Accuracy: {}'.format(dev_acc))
            acc = int(100 * dev_acc)
            torch.save(model.state_dict(), checkpoint + '-a' + str(acc) + '.pt')
        print('Development accuracy for epoch {}: {}'.format(epoch + 1, correct / total))

    torch.save(model.state_dict(), PATH + '-final.pt')

# Test : Kaggle A
# model = MatchDotCOMP()
# model.load_state_dict(torch.load(os.path.join(MODELS, 'small-gru-rms-ndo-e6-a64.pt')))

# N = len(testing_data)
# ids = []
# predictions = []
# for index in tqdm(range(N)):
#     input_1, input_2, id_tag = testing_data[index]
#     prediction_1 = model(input_1[0])
#     prediction_2 = model(input_2[0])

#     prob_truthful_1 = prediction_1[1]
#     prob_false_1 = prediction_1[0]
#     prob_truthful_2 = prediction_2[1]
#     prob_false_2 = prediction_2[0]

#     probs = [prob_truthful_1, prob_false_1, prob_truthful_2, prob_false_2]

#     max_index = probs.index(max(probs))
#     if max_index == 0 or max_index == 3:
#         predicted_label = 0
#     if max_index == 1 or max_index == 2:
#         predicted_label = 1
#     ids.append(id_tag)
#     predictions.append(predicted_label + 1)

# df = pd.DataFrame({'Id': ids, 'Prediction': predictions}, columns = ['Id', 'Prediction'])
# df.to_csv('Part-A_small-gru-rms-ndo-e6-a64.csv', index=False)


if __name__ == '__main__':
    main()
