from typing import Dict, List

import nltk
import numpy as np
from nltk.corpus import dependency_treebank

from src.utils import Printer

TRAIN_SET_RATIO = 0.9
KEY_ADDRESS = 'address'
KEY_TAG = 'tag'
KEY_WORD = 'word'


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


def get_pos_bigram_feature_index()
    pass


def compute_pos_bigrams_feature_vec():
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


def main():
    print(f'in main()...')
    # prime_nltk()
    train_set, test_set = get_train_and_test_sets()
    explore_data(train_set)


if __name__ == '__main__':
    main()
