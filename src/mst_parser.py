from typing import Dict, List

import nltk
import numpy as np
from nltk.corpus import dependency_treebank

from src.Chu_Liu_Edmonds_algorithm import max_spanning_arborescence_nx
from src.utils import Printer

COLS = 1

1 = 1

TRAIN_SET_RATIO = 0.9
KEY_ADDRESS = 'address'
KEY_TAG = 'tag'
KEY_WORD = 'word'
LR = 1
N_ITER = 2


def prime_nltk():
    print(f'\nin prime_nltk()...')
    nltk.download('dependency_treebank')


def get_train_and_test_sets():
    print(f'\nin get_train_and_test_sets()...')
    parsed_sents = dependency_treebank.parsed_sents()
    sents_size = len(parsed_sents)
    print(f'size of parsed_sents = {sents_size}')
    train_size = int(TRAIN_SET_RATIO * sents_size)
    test_size = sents_size - train_size
    train_set = parsed_sents[:train_size]
    print(f'size of train_set = {len(train_set)}')
    test_set = parsed_sents[-test_size:]
    print(f'size of test_set = {len(test_set)}')
    return train_set, test_set


def compute_feature_vec():
    Printer.greet_function('compute_feature_vec')
    word_bigram_vec = compute_word_bigrams_feature_vec()
    pos_bigram_vec = compute_pos_bigrams_feature_vec()
    feature_vec = np.concatenate((word_bigram_vec, pos_bigram_vec))
    return feature_vec


def compute_word_bigrams_feature_vec(u: Dict, v: Dict, input_sentence: Dict[int, Dict], vocab: List):
    Printer.greet_function('compute_word_bigrams_feature_vec')
    size_vocab = len(vocab)
    feature_vec = np.zeros(size_vocab * size_vocab)
    pair_index = get_word_bigram_feature_index(u, v, vocab)
    feature_vec[pair_index] = 1
    return feature_vec


def get_word_bigram_feature_index(u: Dict, v: Dict, word_vocab: List):
    size_vocab = len(word_vocab)
    u_index = word_vocab.index(u[KEY_WORD])
    v_index = word_vocab.index(v[KEY_WORD])
    index = u_index * size_vocab + v_index
    return index


def get_pos_bigram_feature_index():
    pass


def compute_pos_bigrams_feature_vec():
    pass


def avg_perceptron(n_features, train_set, lr):
    Printer.greet_function('avg_perceptron')
    n_iter = N_ITER
    n_examples = len(train_set)
    n_steps = n_iter * n_examples
    weights = np.zeros((n_features, n_steps))
    for r in range(n_iter):
        for i in range(n_examples):
            best_tree = find_best_tree()
            new_weights = find_new_weights(lr)
            curr_step = r * n_examples + i
            weights[:, curr_step] = new_weights
    avg_weights = np.mean(weights, axis=COLS)
    return avg_weights


def find_best_tree():
    Printer.greet_function('find_best_tree')
    complete_graph = None
    arcs = None
    sink = None
    best_tree = max_spanning_arborescence_nx(arcs, sink)

    best_score = 0
    best_tree = None
    for


def find_new_weights(prev_weights, lr):
    pass


def explore_data(data):
    # todo: sample and show random sentences from the data, and its label
    print(f'\nin explore_data()...')
    print(f'type of treebank is: {type(data)}')
    print(f'len of treebank is: {len(data)}')
    print(f'first element is of type: {type(data[0])}')
    print(f'first tree: {data[0].tree()}')
    print(f'first element is: {data[0]}')


def evaluate_tree(gold_tree, predicted_tree):
    Printer.greet_function('evaluate_tree')
    n_edges = len(predicted_tree)
    count_edge_shared = 0
    for edge_i in range(n_edges):
        gold_edge = gold_tree[edge_i]
        pred_edge = predicted_tree[edge_i]
        # add 1 to the count iff the gold edge agrees (head-wise) with the predicted edge
        count_edge_shared += (gold_edge == pred_edge)
    # todo: check if there's an easier oneliner for the above
    attachment_score = count_edge_shared / n_edges
    print(f'attachment score is: {attachment_score}')
    return attachment_score


def evaluate_model(weights, test_set):
    Printer.greet_function('evaluate_learned_weights')
    sum_score = 0
    for sentence in test_set:
        gold_tree = get_gold_tree_from_sentence()
        pred_tree = predict_tree_from_sentence()
        attachment_score = evaluate_tree(gold_tree, pred_tree)
        sum_score += attachment_score
    n_examples = len(test_set)
    avg_score = sum_score / n_examples
    return avg_score


def train_model(train_set):
    Printer.greet_function('train_model')
    n_features = 0
    lr = LR
    weights = avg_perceptron(n_features, train_set, lr)


def main():
    print(f'in main()...')
    # prime_nltk()
    train_set, test_set = get_train_and_test_sets()
    explore_data(train_set)
    weights = train_model(train_set)
    score = evaluate_model(weights)


if __name__ == '__main__':
    main()
