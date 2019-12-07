import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import random
import os
import time
from tqdm import tqdm

def train_and_classify(df_X_train, Y_train, df_X_test):
    """
    Extracts features from df_X_train. Train a model
    on training data and training labels (Y_train).
    Predict the labels of df_X_test.

    df_X_train : pandas data frame of training data
    Y_train    : numpy array of labels for training data
    df_X_test  : pandas data frame of test data

    Output:
    Y_test : numpy array of labels for test data
    """

    EPOCHS = 1
    LR = 0.001

    bert = torch.hub.load('huggingface/transformers', 'model', None, 'bert-base-uncased')
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', None, 'bert-base-uncased')

    class BertnaryClassification(nn.Module):
        def __init__(self):
            super(BertnaryClassification, self).__init__()
            self.linear = nn.Linear(768, 2)
            self.softmax = nn.LogSoftmax(dim=0)
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.Adam(self.parameters(), lr=LR)

        def compute_Loss(self, predicted_vector, gold_label):
            return self.criterion(predicted_vector, gold_label)

        def forward(self, input_vector):
            features = torch.mean(bert(input_vector)[0].squeeze(), dim=0)
            prediction = self.linear(features)
            return self.softmax(prediction)

    def extract_feature_vec(df_X):
        df_X = [torch.tensor([tokenizer.encode(text, add_special_tokens=True)]) for i, text in enumerate(df_X['text'])]
        return df_X

    def classify(X_test, model):
        labels = [int(torch.argmax(model(test))) for test in X_test]
        adjusted_labels = [labels[i] if labels[i] == 1 else -1 for i in range(len(labels))]
        return torch.tensor([adjusted_labels]).squeeze().numpy()

    X_train = extract_feature_vec(df_X_train)
    X_test  = extract_feature_vec(df_X_test)


    # create and train model
    train_data = [(X_train[i], Y_train[i] if Y_train[i] == 1 else 0) for i in range(len(X_train))]

    model = BertnaryClassification()
    for epoch in range(EPOCHS):
        model.train()
        model.optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        random.shuffle(train_data)
        N = len(train_data)
        for index in range(N):
            model.optimizer.zero_grad()
            input_vector, gold_label = train_data[index]
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
            loss.backward()
            model.optimizer.step()

    # evaulate model
    model.optimizer.zero_grad()
    model.eval()
    Y_test = classify(X_test, model)

    return Y_test
