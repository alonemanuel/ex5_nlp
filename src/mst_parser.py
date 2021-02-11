from collections import defaultdict
from random import shuffle
from typing import List, Dict, Any, Iterator

import numpy as np
from tqdm import tqdm

from src.Chu_Liu_Edmonds_algorithm import max_spanning_arborescence_nx
from src.utils import *

KEY_NOT_FOUND = -1

TOP = 'TOP'

COLS = 1

TRAIN_SET_RATIO = 0.9
KEY_ADDRESS = 'address'
KEY_TAG = 'tag'
KEY_WORD = 'word'
KEY_HEAD = 'head'

LR = 1
N_ITER = 2


class MSTParser:
    def __init__(self, train_set: List[SentenceExample]):
        self._train_set: List[List[WordNode]] = train_set
        self._curr_weights = defaultdict(lambda: 0.0)
        self._sum_weights = defaultdict(lambda: 0.0)
        self._trained_weights = defaultdict(lambda: 0.0)

    def train(self, n_epochs: int, lr: float):
        print(f'training...')
        for epoch in tqdm(range(n_epochs),position=0, leave=True):
            for example_i, example in enumerate(tqdm(self._train_set,position=0, leave=True)):
                sentence = example.sentence
                gold_tree = example.gold_tree
                self._train_sentence(sentence, gold_tree, lr)
            shuffle(self._train_set)

        n_iters = n_epochs * len(self._train_set)
        self._trained_weights = dict((key, val / n_iters) for (key, val) in self._sum_weights.items())


    def _train_sentence(self, sentence: List[WordNode], gold_tree: List[TreeEdge], lr: float):
        '''
        trains a single sentence, i.e. returns the new weights after this single example
        '''
        sentence.sort(key=lambda tup: tup.index)
        pred_tree = self._get_predicted_tree(sentence)
        self._update_new_weights(sentence, pred_tree, gold_tree, lr)

    def _update_new_weights(self, sentence: List[WordNode], pred_tree: List[TreeEdge], gold_tree: List[TreeEdge],
                            lr: float):
        assert len(gold_tree) == len(pred_tree)
        for edge_index in range(1, len(gold_tree)):
            gold_edge = gold_tree[edge_index]
            pred_edge = pred_tree[edge_index]
            self._update_weights_with_one_edge(sentence, gold_edge, 1, lr)
            self._update_weights_with_one_edge(sentence, pred_edge, -1, lr)

    def _update_weights_with_one_edge(self, sentence: List[WordNode], edge: TreeEdge,
                                      val_to_add: float, lr: float):

        u_index = edge.u_index
        v_index = edge.v_index
        u_node = sentence[u_index]
        v_node = sentence[v_index]
        for feature_key in self._get_feature_keys(u_node, v_node):
            self._sum_weights[feature_key] += self._curr_weights[feature_key] + (lr * val_to_add)
            self._curr_weights[feature_key] += lr * val_to_add

    @staticmethod
    def _get_mst(complete_graph: List[WordArc]) -> List[TreeEdge]:
        '''
        :param complete_graph: the complete graph of arcs of the given sentence
        :return: an MST of the graph
        '''
        mst = max_spanning_arborescence_nx(complete_graph)
        # mst.sort(key=lambda tup: tup.v_index)
        tree = []
        for arc in mst:
            tree_node = TreeEdge(arc.u_index, arc.v_index)
            tree.append(tree_node)
        return tree

    def evaluate(self, test_set: List[SentenceExample]):
        print(f'in evaluate()...')
        sum_score = 0
        n_samples = len(test_set)
        for example in tqdm(test_set):
            sentence, gold_tree = example.sentence, example.gold_tree
            gold_tree.sort(key=lambda tup: tup.v_index)

            # print(f'gold tree is:\n{gold_tree}')
            pred_tree = self._get_predicted_tree(sentence)
            pred_tree.sort(key=lambda tup: tup.v_index)
            # print(f'pred tree is:\n{pred_tree}')
            curr_score = self._compute_attachment_score(pred_tree, gold_tree)
            sum_score += curr_score
        avg_score = sum_score / n_samples
        print(f'average score is: {avg_score}')
        return avg_score

    def _get_predicted_tree(self, sentence: List[WordNode]) -> List[TreeEdge]:
        full_graph = self._get_complete_graph(sentence)
        mst = self._get_mst(full_graph)
        return mst

    def _get_complete_graph(self, sentence: List[WordNode]) -> List[WordArc]:
        complete_graph = []

        for i, u in enumerate(sentence):
            for j, v in enumerate(sentence[1:]):
                uv_weight = self._get_edge_score(u, v)
                uv_arc = WordArc(u.index, v.index, uv_weight)
                complete_graph.append(uv_arc)
        return complete_graph

    def _get_edge_score(self, u: WordNode, v: WordNode) -> float:
        score = 0.0
        for feature_key in self._get_feature_keys(u, v):
            score += self._curr_weights[feature_key]
        return score

    @staticmethod
    def _get_feature_keys(u: WordNode, v: WordNode) -> Iterator[str]:
        if u.literal is None:
            yield f'u_is_none,v_literal:{v.literal}'
        else:
            yield f'u_literal:{u.literal},v_literal:{v.literal}'
        yield f'u_tag:{u.tag},v_tag:{v.tag}'

    @staticmethod
    def _compute_attachment_score(pred_tree: List[TreeEdge], gold_tree: List[TreeEdge]) -> float:
        assert len(pred_tree) == len(gold_tree)
        n_words = len(pred_tree)
        n_shared_arcs = 0
        for i in range(n_words):
            pred_arc, gold_arc = pred_tree[i], gold_tree[i]
            have_shared_arc = pred_arc.u_index == gold_arc.u_index
            n_shared_arcs += have_shared_arc
        attachment_score = n_words / n_shared_arcs if n_shared_arcs else 0
        return attachment_score


# def compute_feature_vec(first_word: WordNode, second_word: WordNode, word_to_i, tag_to_i):
#     # print(f'computing feature vec...')
#
#     # word_bigram_vec = compute_word_bigrams_feature_vec(first_word, second_word, sentence, word_vocab)
#     # pos_bigram_vec = compute_tag_bigrams_feature_vec(first_word, second_word, sentence, tag_vocab)
#     # feature_vec = np.concatenate((word_bigram_vec, pos_bigram_vec))
#     word_bigram_i = get_word_bigram_feature_index(first_word, second_word, word_to_i)
#     tag_bigram_i = get_tag_bigram_feature_index(first_word, second_word, tag_to_i)
#     feature_vec = {word_bigram_i, tag_bigram_i}
#     return feature_vec
#
#
# def compute_feature_vec2(first_word: WordNode, second_word: WordNode, sentence: List[WordNode], word_vocab: List[str],
#                          tag_vocab: List[str]):
#     print(f'computing feature vec...')
#
#     word_bigram_vec = compute_word_bigrams_feature_vec(first_word, second_word, sentence, word_vocab)
#     pos_bigram_vec = compute_tag_bigrams_feature_vec(first_word, second_word, sentence, tag_vocab)
#     feature_vec = np.concatenate((word_bigram_vec, pos_bigram_vec))
#     return feature_vec
#
#
# def compute_word_bigrams_feature_vec(u: WordNode, v: WordNode, sentence: List[WordNode], word_vocab: List[str]):
#     print(f'computing word bigrams...')
#
#     size_vocab = len(word_vocab)
#     feature_vec = np.zeros(size_vocab * size_vocab, dtype=np.bool)
#     pair_index = get_word_bigram_feature_index(u, v, word_vocab)
#     if pair_index != -1:
#         feature_vec[pair_index] = 1
#     return feature_vec
#
#
# def get_word_bigram_feature_index2(u: WordNode, v: WordNode, word_vocab: List[str]):
#     '''
#     :param u: first word
#     :param v: second word
#     :param word_vocab: vocabulary of unique words
#     :return: the index of the corresponding entry, or -1 if at least one of the words is not in the vocab
#     '''
#     if u in word_vocab and v in word_vocab:
#         size_vocab = len(word_vocab)
#         # u_index = word_vocab.index(u.word)
#         # v_index = word_vocab.index(v.word)
#         # todo: just for debug
#         u_index, v_index = 0, 0
#         index = u_index * size_vocab + v_index
#     else:
#         index = -1
#     return index
#
#
# def get_tag_bigram_feature_index(u, v, tag_dict):
#     u_index, v_index = tag_dict[u.tag], tag_dict[v.tag]
#     if u_index == KEY_NOT_FOUND or v_index == KEY_NOT_FOUND:
#         return -1
#     else:
#         size_vocab = len(tag_dict)
#         index = u_index * size_vocab + v_index
#         return index
#
#
# def get_word_bigram_feature_index(u, v, word_dict):
#     u_index, v_index = word_dict[u.word], word_dict[v.word]
#     if u_index == KEY_NOT_FOUND or v_index == KEY_NOT_FOUND:
#         return -1
#     else:
#         size_vocab = len(word_dict)
#         index = u_index * size_vocab + v_index
#         return index
#
#
# def get_tag_bigram_feature_index2(u: WordNode, v: WordNode, tag_vocab: List[str]):
#     '''
#         :param u: first word
#         :param v: second word
#         :param tag_vocab: vocabulary of unique tags
#         :return: the index of the corresponding entry, or -1 if at least one of the tags is not in the vocab
#         '''
#     if u in tag_vocab and v in tag_vocab:
#         size_vocab = len(tag_vocab)
#         # u_index = tag_vocab.index(u.tag)
#         # v_index = tag_vocab.index(v.tag)
#         # todo: just for debug
#         u_index, v_index = 0, 0
#         index = u_index * size_vocab + v_index
#     else:
#         index = -1
#     return index
#
#
# def compute_tag_bigrams_feature_vec(u: WordNode, v: WordNode, sentence: List[WordNode], tag_vocab: List[str]):
#     print(f'computing tag bigrams...')
#     size_vocab = len(tag_vocab)
#     feature_vec = np.zeros(size_vocab * size_vocab, dtype=np.bool)
#     pair_index = get_word_bigram_feature_index(u, v, tag_vocab)
#     if pair_index != -1:
#         feature_vec[pair_index] = 1
#     return feature_vec
#
#
# def create_dict_from_list(vocab_list):
#     indexes = list(range(len(vocab_list)))
#     words_to_index_dict = defaultdict(lambda: KEY_NOT_FOUND, zip(vocab_list, indexes))
#     return words_to_index_dict
#
#
# def build_complete_graph(sentence: List[WordNode], weights, word_to_i,
#                          tag_to_i):
#     print(f'building complete graph...')
#     arcs = []
#     arcs_feature_dict = dict()
#
#     for i, u in enumerate(sentence):
#         for j, v in enumerate(sentence):
#             # print(f'going over edge: ({i}, {j})...')
#             uv_feature_vec = compute_feature_vec(u, v, word_to_i, tag_to_i)
#             uv_weight = get_edge_weight(uv_feature_vec, weights)
#
#             # debug
#             # uv_feature_vec = 0
#             # uv_weight = 0
#
#             uv_arc = WordArc(u.address, v.address, uv_weight)
#             arcs.append(uv_arc)
#             arcs_feature_dict[uv_arc] = uv_feature_vec
#     return arcs, arcs_feature_dict
#
#
# def get_gold_tree(sentence: List[WordNode]) -> List[WordArc]:
#     arcs = []
#     for u in sentence:
#         uv_arc = WordArc(u.address, u.head, -1)
#         arcs.append(uv_arc)
#     return arcs
#
#
# def train_sentence(sentence, prev_weights, word_to_i, tag_to_i, lr):
#     full_graph, arc_feature_dict = build_complete_graph(sentence, prev_weights, word_to_i, tag_to_i)
#     pred_tree = find_best_tree(full_graph)
#     gold_tree = get_gold_tree(sentence)
#     curr_weights = find_new_weights(prev_weights, lr, pred_tree, gold_tree, arc_feature_dict)
#     return curr_weights
#
#
# def avg_perceptron(word_vocab: List[str], tag_vocab: List[str], train_set: List[List[WordNode]], lr: float,
#                    n_iter: int):
#     Printer.greet_function('avg_perceptron')
#     n_examples = len(train_set)
#     n_steps = n_iter * n_examples
#     # sum of all weights
#
#     word_to_i = create_dict_from_list(word_vocab)
#     tag_to_i = create_dict_from_list(tag_vocab)
#     weights_size = len(word_to_i) ** 2 + len(tag_to_i) ** 2
#     curr_weights = np.zeros(weights_size)
#     sum_weights = np.copy(curr_weights)
#     for curr_iter in range(n_iter):
#         print(f'in iter: {curr_iter}')
#         for i, sentence in enumerate(train_set):
#             print(f'\nin example: {i}')
#             curr_weights = train_sentence(sentence, curr_weights, word_to_i, tag_to_i, lr)
#             # if sum_weights has been initialized, add to it. Else, initialize it.
#             sum_weights = sum_weights + curr_weights
#     avg_weights = sum_weights / n_steps
#     return avg_weights
#
#
# def get_edge_weight(edge_feature_vec, weights):
#     if weights is not None:
#         edge_feature_vec = np.array(list(edge_feature_vec))
#         weight = np.sum(np.take(weights, edge_feature_vec))
#         # weight = np.dot(weights, edge_feature_vec)
#     else:
#         weight = 0
#     return weight
#
#
# def get_edge_weight2(edge_feature_vec, weights):
#     if weights is not None:
#         weight = np.dot(weights, edge_feature_vec)
#     else:
#         weight = 0
#     return weight
#
#
# def find_best_tree(complete_graph: List[WordArc]):
#     print('finding best tree...')
#     best_tree = max_spanning_arborescence_nx(complete_graph)
#
#     return list(best_tree.values())
#     # return []
#
#
# def get_sum_of_feature_vecs(tree: List[WordArc], arc_feature_dict):
#     sum_tree_features = arc_feature_dict[tree[0]]
#     for arc in tree[1:]:
#         sum_tree_features += arc_feature_dict[arc]
#     return sum_tree_features
#
#
# def get_sum_of_feature_vecs2(tree: List[WordArc], arc_feature_map: Dict[WordArc, Any]):
#     first_arc = tree[0]
#     sum_tree_features = arc_feature_map[first_arc]
#     for arc in tree[1:]:
#         arc_feature_vec = arc_feature_map[arc]
#         sum_tree_features += arc_feature_vec
#     return sum_tree_features
#
#
# def find_new_weights(prev_weights, lr: float, pred_tree: List[WordArc],
#                      gold_tree: List[WordArc], arc_feature_dict: Dict[WordArc, Any]):
#     sum_pred_tree_features = get_sum_of_feature_vecs(pred_tree, arc_feature_dict)
#     sum_gold_tree_features = get_sum_of_feature_vecs(gold_tree, arc_feature_dict)
#
#     indices = list(sum_gold_tree_features.difference(sum_pred_tree_features))
#     new_weights = np.copy(prev_weights)
#     new_weights[indices] = 1
#
#     return new_weights
#
#
# def evaluate_tree(gold_tree, predicted_tree):
#     n_edges = len(predicted_tree)
#     count_edge_shared = 0
#     for edge_i in range(n_edges):
#         gold_edge = gold_tree[edge_i]
#         pred_edge = predicted_tree[edge_i]
#         # add 1 to the count iff the gold edge agrees (head-wise) with the predicted edge
#         count_edge_shared += (gold_edge == pred_edge)
#     # todo: check if there's an easier oneliner for the above
#     attachment_score = count_edge_shared / n_edges
#     print(f'attachment score is: {attachment_score}')
#     return attachment_score


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

#
# def get_word_tag_vocab(data_set: List[List[WordNode]]):
#     '''
#     :param data_set: set of sentences to extract vocab from
#     :return: sorted lists of unique words and tags
#     '''
#     Printer.greet_function('get_word_tag_vocab')
#     # todo: what about commas and other punctuation? do we strip them?
#     word_vocab = []
#     tag_vocab = []
#     for sentence in data_set:
#         for word_node in sentence:
#             # if word is not the root
#             if word_node.word:
#                 word_vocab.append(word_node.word)
#                 tag_vocab.append(word_node.tag)
#     word_vocab = sorted(list(set(word_vocab)))
#     tag_vocab = sorted(list(set(tag_vocab)))
#     print(f'word_vocab size is: {len(word_vocab)}')
#     print(f'tag_vocab size is: {len(tag_vocab)}')
#     return word_vocab, tag_vocab

#
# def train_model(train_set, word_vocab, tag_vocab):
#     Printer.greet_function('train_model')
#     weights = avg_perceptron(word_vocab, tag_vocab, train_set, LR, N_ITER)
#     return weights
#
#
# class Tree(list):
#     pass
