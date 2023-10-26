# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE:
# SUNNY HUANG
# SHASHIDHAR GOLLAMUDI

"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk.classify import NaiveBayesClassifier
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
nltk.download('punkt')

def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    id\ttext\tlabel
    id\ttext\tlabel
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 1:
            if dev_y[i] == preds[i]:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if dev_y[i] == preds[i]:
                true_neg += 1
            else:
                false_neg += 1
    accuracy = (true_neg + true_pos) / (true_pos + true_neg + false_neg + false_pos)
    precision = (true_pos) / (true_pos + false_pos)
    recall = (true_pos) / (true_pos + false_neg)
    f1 = (true_pos) / (true_pos + (false_neg + false_pos) / 2)
    if verbose:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Accuracy: {accuracy}')
    return precision, recall, f1, accuracy

def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data, dev data, and percentage and returns preds and labels for the dev set
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    accuracies, precisions, f1s, recalls = [], [], [], []
    plot_x = []
    for i in range(10, 110, 10):
        plot_x.append(i)

        percentage = i / 100
        y_dev, preds = metrics_fun(train_feats, dev_feats, percentage)
        precision, recall, f1, accuracy = get_prfa(y_dev, preds)

        accuracies.append(accuracy)
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)

    if verbose:
        print(f'Metrics for {kind} when trained on 100% of data')
        print(f'Precision: {precisions[-1]}')
        print(f'Recall: {recalls[-1]}')
        print(f'F1 score: {f1s[-1]}')
        print(f'Accuracy: {accuracies[-1]}')

    plt.grid(True)

    plt.plot(plot_x, precisions, label="Precision")
    plt.plot(plot_x, recalls, label="Recall")
    plt.plot(plot_x, f1s, label="F1")
    plt.plot(plot_x, accuracies, label="Accuracy")

    plt.xlabel("Percentage of Training Data")
    plt.ylabel("Metric Scores")
    plt.title(f'Performance of {kind} Model')

    plt.legend()
    if savepath:
        plt.savefig(savepath)
    plt.show()


def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    unraveled = []
    for data in all_train_data_X:
        unraveled = unraveled + data
    return list(set(unraveled))


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    sparse_vector = []
    for data in data_to_be_featurized_X:
        counts = Counter(data)
        current_vector = []
        for word in vocab:
            if word not in counts:
                current_vector.append(0)
            else:
                if binary:
                    current_vector.append(1)
                else:
                    current_vector.append(counts[word])
        sparse_vector.append(current_vector)
    return sparse_vector