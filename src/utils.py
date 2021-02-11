from collections import namedtuple


class Printer:
    @staticmethod
    def greet_function(function_name):
        print(f'\nin function: {function_name}...')


WordArc = namedtuple('WordArc', 'u_index v_index weight')
WordNode = namedtuple('WordNode', 'index literal tag')
WordFeature = namedtuple('WordFeature', 'word feature_vec')
TreeEdge = namedtuple('TreeEdge', 'u_index v_index')
SentenceExample = namedtuple('SentenceExample', 'sentence gold_tree')
