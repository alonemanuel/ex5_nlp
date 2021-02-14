import string
from collections import defaultdict, Counter
from random import shuffle
from typing import List, Iterator, Dict

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
        self._curr_weights = Counter()
        self._sum_weights = Counter()
        self._trained_weights = defaultdict(lambda: 0.0)
        self._weights_dicts_list = []
        self._sum_weights = Counter()

    def train(self, n_epochs: int, lr: float):
        print(f'training...')
        for epoch in tqdm(range(n_epochs), position=0, leave=True):
            for example_i, example in enumerate(tqdm(self._train_set, position=0, leave=True)):
                sentence = example.sentence
                gold_tree = example.gold_tree
                self._train_sentence(sentence, gold_tree, lr)
            shuffle(self._train_set)

        self._update_trained_weights(n_epochs * len(self._train_set))

    def _update_trained_weights(self, n_iters):
        print(f'updating trained weights...')
        for weight_key in tqdm(self._sum_weights, position=0, leave=True):
            weight = self._sum_weights[weight_key]
            self._trained_weights[weight_key] = weight / n_iters
            # for feature, weight in weights_dict.items():
            #     self._trained_weights[feature] += weight
        # trained_weights = dict((key, val / n_iters) for (key, val) in self._trained_weights.items())
        # self._trained_weights = defaultdict(lambda: 0.0, trained_weights)

    def _train_sentence(self, sentence: Dict[int, WordNode], gold_tree: Dict[int, TreeEdge], lr: float):
        '''
        trains a single sentence, i.e. returns the new weights after this single example
        '''
        pred_tree = self._get_predicted_tree(sentence)
        self._update_new_weights(sentence, pred_tree, gold_tree, lr)
        self._sum_weights.update(self._curr_weights)
        # self._weights_dicts_list.append(copy(self._curr_weights))

    def _update_new_weights(self, sentence: Dict[int, WordNode], pred_tree: Dict[int, TreeEdge],
                            gold_tree: Dict[int, TreeEdge],
                            lr: float):
        assert len(gold_tree) == len(pred_tree)
        for edge_index in gold_tree:
            gold_edge = gold_tree[edge_index]
            pred_edge = pred_tree[edge_index]
            # if edge shares same head, i.e. if it's the same edge
            if gold_edge.u_index == pred_edge.u_index:
                continue
            self._update_weights_with_one_edge(sentence, gold_edge, 1, lr)
            self._update_weights_with_one_edge(sentence, pred_edge, -1, lr)

    def _update_weights_with_one_edge(self, sentence: Dict[int, WordNode], edge: TreeEdge,
                                      val_to_add: float, lr: float):

        u_index = edge.u_index
        v_index = edge.v_index
        u_node = sentence[u_index]
        v_node = sentence[v_index]
        for feature_key in self._get_feature_keys(u_node, v_node, sentence):
            if self._curr_weights[feature_key] + (lr * val_to_add) == 0:
                del self._curr_weights[feature_key]
                continue
            self._curr_weights[feature_key] += lr * val_to_add

    def _get_predicted_tree(self, sentence: Dict[int, WordNode], evaluation=False) -> Dict[int, TreeEdge]:
        full_graph = self._get_complete_graph(sentence, evaluation)
        mst = max_spanning_arborescence_nx(full_graph)
        return mst

    def _get_complete_graph(self, sentence: Dict[int, WordNode], evaluation: bool) -> List[WordArc]:
        complete_graph = []

        for i, u in sentence.items():
            for j, v in sentence.items():
                if v.literal is None:
                    continue
                uv_weight = self._get_edge_score(u, v, sentence, evaluation)
                uv_arc = WordArc(i, j, uv_weight)
                complete_graph.append(uv_arc)
        return complete_graph

    def _get_edge_score(self, u: WordNode, v: WordNode, sentence: Dict[int, WordNode], evaluation: bool) -> float:
        score = 0.0
        for feature_key in self._get_feature_keys(u, v, sentence):
            if evaluation:
                # if feature_key in self._trained_weights:
                score += self._trained_weights[feature_key]
            else:
                # if feature_key in self._curr_weights:
                score += self._curr_weights[feature_key]
        return score

    @staticmethod
    def _get_feature_keys(u: WordNode, v: WordNode, sentence: [int, WordNode]) -> Iterator[str]:
        if u.literal is None:
            yield f'u_is_none,v_literal:{v.literal}'
        else:
            yield f'u_literal:{u.literal},v_literal:{v.literal}'
        yield f'u_tag:{u.tag},v_tag:{v.tag}'
        if u.index > 0:
            if v.index > 0:
                yield f'u_tag:{u.tag},u-1_tag:{sentence[u.index - 1].tag},v_tag:{v.tag},v-1_tag:{sentence[v.index - 1]}'
            else:
                yield f'u_tag:{u.tag},u-1_tag:{sentence[u.index - 1].tag},v_tag:{v.tag}'
        else:
            if v.index > 0:
                yield f'u_tag:{u.tag},v_tag:{v.tag},v-1_tag:{sentence[v.index - 1]}'
        if u.index > len(sentence)-1:
            if v.index > len(sentence)-1:
                yield f'u_tag:{u.tag},u+1_tag:{sentence[u.index + 1].tag},v_tag:{v.tag},v+1_tag:{sentence[v.index + 1]}'
            else:
                yield f'u_tag:{u.tag},u+1_tag:{sentence[u.index + 1].tag},v_tag:{v.tag}'
        else:
            if v.index > len(sentence)-1:
                yield f'u_tag:{u.tag},v_tag:{v.tag},v+1_tag:{sentence[u.index + 1]}'
        yield f'is_right_directed:{v.index > u.index}'
        yield f'uv_distance:{u.index - v.index}'
        yield f'uv_abs_distance:{abs(u.index - v.index)}'
        yield f'u_tag:{u.tag},v_literal:{v.literal}'
        yield f'u_literal:{u.literal},v_tag:{v.tag}'
        if u.literal is not None and u.literal in string.punctuation:
            yield f'u_is_punctuation'
        if v.literal is not None and v.literal in string.punctuation:
            yield f'v_is_punctuation'
        for i in range(min(u.index, v.index) + 1, max(u.index, v.index)):
            if sentence[i].literal in string.punctuation:
                yield f'has_punctuation_between'

    @staticmethod
    def _compute_attachment_score(pred_tree: Dict[int, TreeEdge], gold_tree: Dict[int, TreeEdge]) -> float:
        assert len(pred_tree) == len(gold_tree)
        n_arcs = len(pred_tree)
        n_shared_arcs = 0
        for key in pred_tree:
            pred_edge = pred_tree[key]
            gold_edge = gold_tree[key]
            have_shared_arc = pred_edge.u_index == gold_edge.u_index
            n_shared_arcs += have_shared_arc

        attachment_score = n_shared_arcs / n_arcs if n_shared_arcs else 0
        return attachment_score

    def evaluate(self, test_set: List[SentenceExample]):
        print(f'in evaluate()...')
        sum_score = 0
        n_samples = len(test_set)
        for example in tqdm(test_set, position=0, leave=True):
            sentence, gold_tree = example.sentence, example.gold_tree
            pred_tree = self._get_predicted_tree(sentence, evaluation=True)
            curr_score = self._compute_attachment_score(pred_tree, gold_tree)
            sum_score += curr_score
        avg_score = sum_score / n_samples
        print(f'average score is: {avg_score}')
        return avg_score
