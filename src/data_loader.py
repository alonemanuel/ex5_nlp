from typing import List, Tuple

import nltk
from nltk.corpus import dependency_treebank

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



class DataLoader:
    def __init__(self, data_source=dependency_treebank, train_set_ratio: float = TRAIN_SET_RATIO,
                 nltk_package_name='dependency_treebank'):
        print(f'in DataLoader init...')
        self._prime_nltk(nltk_package_name)
        self._data_source = data_source
        self._train_set_ratio = train_set_ratio
        self._raw_data = self._get_raw_data()
        self._prep_sentences: List[SentenceExample] = self._preprocess_data()

    def _prime_nltk(self, nltk_package_name):
        print(f'\nin prime_nltk()...')
        nltk.download(nltk_package_name)

    def _get_raw_data(self):
        return self._data_source.parsed_sents()

    def _preprocess_data(self) -> List[SentenceExample]:
        sentences = []
        for sentence_graph in self._raw_data:
            sentence_dict = sentence_graph
            sentence_list: List[WordNode] = []
            gold_tree: List[TreeEdge] = []
            for j, word in sentence_dict.nodes.items():
                word_node = WordNode(word[KEY_ADDRESS], word[KEY_WORD], word[KEY_TAG])
                sentence_list.append(word_node)
                if j!=0:
                    tree_edge = TreeEdge(word[KEY_HEAD], word[KEY_ADDRESS])
                    gold_tree.append(tree_edge)

            example = SentenceExample(sentence_list, gold_tree)
            sentences.append(example)

        return sentences

    def get_train_and_test_sets(self) -> Tuple[SentenceExample, SentenceExample]:
        print(f'in get_train_and_test_sets...')
        sents_size = len(self._prep_sentences)
        train_size = int(self._train_set_ratio * sents_size)
        test_size = sents_size - train_size
        train_set = self._prep_sentences[:train_size]
        test_set = self._prep_sentences[-test_size:]
        return train_set, test_set

    def explore_data(self):
        # todo: sample and show random sentences from the data, and its label
        data = self._data_source
        print(f'\nin explore_data()...')
        print(f'type of treebank is: {type(data)}')
        print(f'len of treebank is: {len(data)}')
        print(f'first element is of type: {type(data[0])}')
        print(f'first tree: {data[0].tree()}')
        print(f'first element is: {data[0]}')
