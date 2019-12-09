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
from transformers import *

train_data = pd.read_csv('./gpu_train.csv', encoding='latin-1')
dev_data = pd.read_csv('./gpu_dev.csv', encoding='latin-1')
test_data = pd.read_csv('./gpu_test.csv', encoding='latin-1')
data = [train_data, dev_data, test_data]

config = BertConfig.from_pretrained('./mini/config.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mini = BertModel.from_pretrained('./mini/pytorch_model.bin', config=config)

# bert = torch.hub.load('huggingface/transformers', 'model', 'bert-base-uncased')
# tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'bert-base-uncased')

EPOCHS = 4
LR = 0.001
NAME = 'mini1'
CURRENT = os.curdir
MODELS = os.path.join(CURRENT, 'experimental_models')
PATH = os.path.join(MODELS, NAME)

def train_and_classify(training_data, development_data, testing_data):
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
            # features = torch.mean(bert(input_vector)[0].squeeze(), dim=0)
            features = torch.mean(mini(input_vector)[0].squeeze(), dim=0)
            prediction = self.linear(features)

            return self.softmax(prediction)

    def setup(training_data, development_data, testing_data):
        print('\nInitializing Setup')
        train_data = []
        for row in training_data.iterrows():
            ID = row[1][0]
            if row[1][7] == 1:
                pos_story = ' '.join(word for word in row[1][1:6].values)
                pos_story = torch.tensor([tokenizer.encode(pos_story, add_special_tokens=True)])
                pos = (ID, pos_story, 1)
                neg_story = ' '.join(word for word in list(row[1][1:5].values) + [row[1][6]])
                neg_story = torch.tensor([tokenizer.encode(neg_story, add_special_tokens=True)])
                neg = (ID, neg_story, 0)
            else:
                neg_story = ' '.join(word for word in row[1][1:6].values)
                neg_story = torch.tensor([tokenizer.encode(neg_story, add_special_tokens=True)])
                neg = (ID, neg_story, 0)
                pos_story = ' '.join(word for word in list(row[1][1:5].values) + [row[1][6]])
                pos_story = torch.tensor([tokenizer.encode(pos_story, add_special_tokens=True)])
                pos = (ID, pos_story, 1)
            train_data.append(pos)
            train_data.append(neg)

        dev_data = []
        for row in development_data.iterrows():
            ID = row[1][0]
            LABEL = row[1][7] - 1
            story_1 = ' '.join(word for word in row[1][1:6].values)
            story_1 = torch.tensor([tokenizer.encode(story_1, add_special_tokens=True)])
            story_2 = ' '.join(word for word in list(row[1][1:5].values) + [row[1][6]])
            story_2 = torch.tensor([tokenizer.encode(story_2, add_special_tokens=True)])
            sample = (ID, story_1, story_2, LABEL)
            dev_data.append(sample)

        test_data = []
        for row in testing_data.iterrows():
            ID = row[1][0]
            story_1 = ' '.join(word for word in row[1][1:6].values)
            story_1 = torch.tensor([tokenizer.encode(story_1, add_special_tokens=True)])
            story_2 = ' '.join(word for word in list(row[1][1:5].values) + [row[1][6]])
            story_2 = torch.tensor([tokenizer.encode(story_2, add_special_tokens=True)])
            sample = (ID, story_1, story_2)
            test_data.append(sample)

        return train_data, dev_data, test_data

    def setup_mini(training_data, development_data, testing_data):
        print('\nInitializing Setup')
        train_data = []
        for row in training_data.iterrows():
            ID = row[1][0]
            if row[1][7] == 1:
                pos_story = ' '.join(word for word in row[1][4:6].values)
                pos_story = torch.tensor([tokenizer.encode(pos_story, add_special_tokens=True)])
                pos = (ID, pos_story, 1)
                neg_story = ' '.join(word for word in list(row[1][4:5].values) + [row[1][6]])
                neg_story = torch.tensor([tokenizer.encode(neg_story, add_special_tokens=True)])
                neg = (ID, neg_story, 0)
            else:
                neg_story = ' '.join(word for word in row[1][4:6].values)
                neg_story = torch.tensor([tokenizer.encode(neg_story, add_special_tokens=True)])
                neg = (ID, neg_story, 0)
                pos_story = ' '.join(word for word in list(row[1][4:5].values) + [row[1][6]])
                pos_story = torch.tensor([tokenizer.encode(pos_story, add_special_tokens=True)])
                pos = (ID, pos_story, 1)
            train_data.append(pos)
            train_data.append(neg)

        dev_data = []
        for row in development_data.iterrows():
            ID = row[1][0]
            LABEL = row[1][7] - 1
            story_1 = ' '.join(word for word in row[1][4:6].values)
            story_1 = torch.tensor([tokenizer.encode(story_1, add_special_tokens=True)])
            story_2 = ' '.join(word for word in list(row[1][4:5].values) + [row[1][6]])
            story_2 = torch.tensor([tokenizer.encode(story_2, add_special_tokens=True)])
            sample = (ID, story_1, story_2, LABEL)
            dev_data.append(sample)

        test_data = []
        for row in testing_data.iterrows():
            ID = row[1][0]
            story_1 = ' '.join(word for word in row[1][1:6].values)
            story_1 = torch.tensor([tokenizer.encode(story_1, add_special_tokens=True)])
            story_2 = ' '.join(word for word in list(row[1][4:5].values) + [row[1][6]])
            story_2 = torch.tensor([tokenizer.encode(story_2, add_special_tokens=True)])
            sample = (ID, story_1, story_2)
            test_data.append(sample)

        return train_data, dev_data, test_data

    def train(train_data, dev_data):
        model = BertnaryClassification()
        prev_acc = 0
        best_epoch = 0
        best_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            model.optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            epoch_loss = 0
            start_time = time.time()
            random.shuffle(train_data)
            N = len(train_data)
            print('Training started for epoch {}'.format(epoch + 1))
            for index in tqdm(range(N)):
                model.optimizer.zero_grad()
                __, input_vector, gold_label = train_data[index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                epoch_loss += loss
                loss.backward()
                model.optimizer.step()
            print('Average loss for epoch {}: {}'.format(epoch_loss / N))
            print('Training accuracy for epoch {}: {}'.format(epoch + 1, str(round((correct / total) * 100, 2)) + '%'))
            acc = validate(dev_data, model)
            print('Development accuracy for epoch {}: {}'.format(epoch + 1, str(round(acc * 100, 2)) + '%'))
            acc = int(acc * 100)
            if acc > prev_acc and acc > 70:
                prev_acc = acc
                print('New Best! Saving model')
                torch.save(model.state_dict(), PATH + '-epoch' + str(epoch + 1) + '-acc' + str(acc) + '.pt')
                best_epoch = epoch + 1
                best_acc = acc
        return best_epoch, best_acc

    def validate(dev_data, model):
        model.eval()
        model.optimizer.zero_grad()
        N = len(dev_data)
        predictions = []
        for index in tqdm(range(N)):
            __, input_1, input_2, __ = dev_data[index]
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
            predictions.append(predicted_label)
        correct = 0
        total = 0
        for i in range(len(predictions)):
            correct += int(predictions[i] == dev_data[i][3])
            total += 1
        return correct / total

    def classify(test_data, model):
        model.eval()
        model.optimizer.zero_grad()
        N = len(test_data)
        ids = []
        predictions = []
        print('\nClassifying test data')
        for index in tqdm(range(N)):
            ID, input_1, input_2 = test_data[index]
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
            ids.append(ID)
            predictions.append(predicted_label + 1)
        df = pd.DataFrame({'Id': ids, 'Prediction': predictions}, columns = ['Id', 'Prediction'])
        df.to_csv(NAME, index=False)
        return predictions

    # train_data, dev_data, test_data = setup(training_data, development_data, testing_data)
    train_data, dev_data, test_data = setup_mini(training_data, development_data, testing_data)

    # create and train model
    epoch, acc = train(train_data, dev_data)
    path = PATH + '-epoch' + str(epoch) + '-acc' + str(acc) + '.pt'
    model = BertnaryClassification()
    model.load_state_dict(torch.load(path))

    # evaulate model
    return classify(test_data, model)

train_and_classify(train_data, dev_data, test_data)
