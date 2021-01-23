from collections import namedtuple
from typing import List, Dict, Any, Tuple

import nltk
import numpy as np
from nltk.corpus import dependency_treebank

from src.Chu_Liu_Edmonds_algorithm import max_spanning_arborescence_nx
from src.utils import Printer

TOP = 'TOP'

KEY_HEAD = 'head'

COLS = 1

TRAIN_SET_RATIO = 0.9
KEY_ADDRESS = 'address'
KEY_TAG = 'tag'
KEY_WORD = 'word'

LR = 1
N_ITER = 2

Arc = namedtuple('Arc', 'tail head weight')
WordNode = namedtuple('WordNode', 'address word tag head')
WordFeature = namedtuple('WordFeature', 'word feature_vec')


def prime_nltk():
    print(f'\nin prime_nltk()...')
    nltk.download('dependency_treebank')


def get_train_and_test_sets() -> Tuple[List[Dict], List[Dict]]:
    print(f'\nin get_train_and_test_sets()...')
    parsed_sents = dependency_treebank.parsed_sents()
    sents_size = len(parsed_sents)
    print(f'size of parsed_sents = {sents_size}')
    print(f'type of parsed_sents = {type(parsed_sents)}')
    train_size = int(TRAIN_SET_RATIO * sents_size)
    test_size = sents_size - train_size
    train_set = parsed_sents[:train_size]
    print(f'size of train_set = {len(train_set)}')
    test_set = parsed_sents[-test_size:]
    print(f'size of test_set = {len(test_set)}')
    return train_set, test_set


def compute_feature_vec(first_word: WordNode, second_word: WordNode, sentence: List[WordNode], word_vocab: List[str],
                        tag_vocab: List[str]):
    word_bigram_vec = compute_word_bigrams_feature_vec(first_word, second_word, sentence, word_vocab)
    pos_bigram_vec = compute_tag_bigrams_feature_vec(first_word, second_word, sentence, tag_vocab)
    feature_vec = np.concatenate((word_bigram_vec, pos_bigram_vec))
    return feature_vec


def compute_word_bigrams_feature_vec(u: WordNode, v: WordNode, sentence: List[WordNode], word_vocab: List[str]):
    size_vocab = len(word_vocab)
    feature_vec = np.zeros(size_vocab * size_vocab)
    pair_index = get_word_bigram_feature_index(u, v, word_vocab)
    if pair_index != -1:
        feature_vec[pair_index] = 1
    return feature_vec


def get_word_bigram_feature_index(u: WordNode, v: WordNode, word_vocab: List[str]):
    '''
    :param u: first word
    :param v: second word
    :param word_vocab: vocabulary of unique words
    :return: the index of the corresponding entry, or -1 if at least one of the words is not in the vocab
    '''
    if u in word_vocab and v in word_vocab:
        size_vocab = len(word_vocab)
        u_index = word_vocab.index(u.word)
        v_index = word_vocab.index(v.word)
        index = u_index * size_vocab + v_index
    else:
        index = -1
    return index


def get_tag_bigram_feature_index(u: WordNode, v: WordNode, tag_vocab: List[str]):
    '''
        :param u: first word
        :param v: second word
        :param tag_vocab: vocabulary of unique tags
        :return: the index of the corresponding entry, or -1 if at least one of the tags is not in the vocab
        '''
    if u in tag_vocab and v in tag_vocab:
        size_vocab = len(tag_vocab)
        u_index = tag_vocab.index(u.tag)
        v_index = tag_vocab.index(v.tag)
        index = u_index * size_vocab + v_index
    else:
        index = -1
    return index


def compute_tag_bigrams_feature_vec(u: WordNode, v: WordNode, sentence: List[WordNode], tag_vocab: List[str]):
    size_vocab = len(tag_vocab)
    feature_vec = np.zeros(size_vocab * size_vocab)
    pair_index = get_word_bigram_feature_index(u, v, tag_vocab)
    if pair_index != -1:
        feature_vec[pair_index] = 1
    return feature_vec


def get_gold_tree(sentence: List[WordNode]) -> List[Arc]:
    arcs = []
    for u in sentence:
        for v in sentence:
            uv_arc = Arc(u, v, None)
            arcs.append(uv_arc)
    return arcs


def avg_perceptron(word_vocab: List[str], tag_vocab: List[str], train_set: List[List[WordNode]], lr: float,
                   n_iter: int):
    Printer.greet_function('avg_perceptron')
    n_examples = len(train_set)
    n_steps = n_iter * n_examples
    # sum of all weights
    sum_weights = None
    curr_weights = None
    for curr_iter in range(n_iter):
        print(f'in iter: {curr_iter}')
        for i, sentence in enumerate(train_set):
            print(f'\nin example: {i}')
            full_graph, arc_feature_dict = build_complete_graph(sentence, curr_weights, word_vocab, tag_vocab)
            pred_tree = find_best_tree(full_graph)
            gold_tree = get_gold_tree(sentence)
            curr_weights = find_new_weights(lr, curr_weights, pred_tree, gold_tree, arc_feature_dict)
            # if sum_weights has been initialized, add to it. Else, initialize it.
            sum_weights = sum_weights + curr_weights if sum_weights else curr_weights
    avg_weights = sum_weights / n_steps
    return avg_weights


def get_edge_weight(edge_feature_vec, weights):
    if weights is not None:
        weight = np.dot(weights, edge_feature_vec)
    else:
        weight = 0
    return weight


def build_complete_graph(sentence: List[WordNode], weights, word_vocab: List[str], tag_vocab: List[str]):
    print(f'building complete graph...')
    arcs = []
    arcs_feature_dict = dict()
    for u in sentence:
        for v in sentence:
            uv_feature_vec = compute_feature_vec(u, v, sentence, word_vocab, tag_vocab)
            uv_weight = get_edge_weight(uv_feature_vec, weights)
            uv_arc = Arc(u, v, uv_weight)
            arcs.append(uv_arc)
            arcs_feature_dict[uv_arc] = uv_feature_vec
    return arcs, arcs_feature_dict


def find_best_tree(complete_graph: List[Arc]):
    print('finding best tree...')
    best_tree = max_spanning_arborescence_nx(complete_graph)
    return best_tree


def get_sum_of_feature_vecs(tree: List[Arc], arc_feature_map: Dict[Arc, Any]):
    first_arc = tree[0]
    sum_tree_features = arc_feature_map[first_arc]
    for arc in tree[1:]:
        arc_feature_vec = arc_feature_map[arc]
        sum_tree_features += arc_feature_vec
    return sum_tree_features


def find_new_weights(prev_weights, lr: float, pred_tree: List[Arc],
                     gold_tree: List[Arc], arc_feature_dict: Dict[Arc, Any]):
    sum_pred_tree_features = get_sum_of_feature_vecs(pred_tree, arc_feature_dict)
    sum_gold_tree_features = get_sum_of_feature_vecs(gold_tree, arc_feature_dict)
    new_weights = prev_weights + lr * (sum_gold_tree_features - sum_pred_tree_features)
    return new_weights


def explore_data(data):
    # todo: sample and show random sentences from the data, and its label
    print(f'\nin explore_data()...')
    print(f'type of treebank is: {type(data)}')
    print(f'len of treebank is: {len(data)}')
    print(f'first element is of type: {type(data[0])}')
    print(f'first tree: {data[0].tree()}')
    print(f'first element is: {data[0]}')


def evaluate_tree(gold_tree, predicted_tree):
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


# def evaluate_model(weights, test_set):
#     Printer.greet_function('evaluate_learned_weights')
#     sum_score = 0
#     for sentence in test_set:
#         gold_tree = get_gold_tree_from_sentence()
#         pred_tree = predict_tree_from_sentence()
#         attachment_score = evaluate_tree(gold_tree, pred_tree)
#         sum_score += attachment_score
#     n_examples = len(test_set)
#     avg_score = sum_score / n_examples
#     return avg_score


def get_word_tag_vocab(data_set: List[List[WordNode]]):
    '''
    :param data_set: set of sentences to extract vocab from
    :return: sorted lists of unique words and tags
    '''
    Printer.greet_function('get_word_tag_vocab')
    # todo: what about commas and other punctuation? do we strip them?
    word_vocab = []
    tag_vocab = []
    for sentence in data_set:
        for word_node in sentence:
            # if word is not the root
            if word_node.word:
                word_vocab.append(word_node.word)
                tag_vocab.append(word_node.tag)
    word_vocab = sorted(list(set(word_vocab)))
    tag_vocab = sorted(list(set(tag_vocab)))
    print(f'word_vocab size is: {len(word_vocab)}')
    print(f'tag_vocab size is: {len(tag_vocab)}')
    return word_vocab, tag_vocab


def train_model(train_set, word_vocab, tag_vocab):
    Printer.greet_function('train_model')
    weights = avg_perceptron(word_vocab, tag_vocab, train_set, LR, N_ITER)
    return weights


def prep_data(tree_bank_data: list) -> List[List[WordNode]]:
    """
    :param tree_bank_data: the data received from 'tree_bank'.
    A dict of keys (indexes) holding sentences (dicts holding words)
    :return: preprocessed data: a list of sentences, where each sentence is a list of WordNodes
    """
    Printer.greet_function('arcs_feature_dict')
    sentences = []
    for sentence_graph in tree_bank_data:
        sentence_dict = sentence_graph
        sentence_list = []
        for j, word in sentence_dict.nodes.items():
            word_node = WordNode(j, word[KEY_WORD], word[KEY_TAG], word[KEY_HEAD])
            sentence_list.append(word_node)
        sentences.append(sentence_list)
    return sentences


# def get_sentences_embeddings(data: List[List[WordNode]]) -> List[list]:
#     word_vocab, tag_vocab = get_word_tag_vocab(data)
#     sentences_embeddings = []
#     for sentence in data:
#         curr_sentence_embedding = []
#         for word_node in sentence:
#             word_embedding = feat


def main():
    print(f'in main()...')
    # prime_nltk()
    train_set, test_set = get_train_and_test_sets()
    # explore_data(train_set)
    prep_train_set = prep_data(train_set)
    word_vocab, tag_vocab = get_word_tag_vocab(prep_train_set)
    # sentences_embeddings = get_sentences_embeddings(prep_train_set)
    weights = train_model(prep_train_set, word_vocab, tag_vocab)
    # score = evaluate_model(weights)
    # print(f'score for the model is: {score}')


if __name__ == '__main__':
    main()
