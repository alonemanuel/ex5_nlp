import nltk
from nltk.corpus import dependency_treebank

TRAIN_SET_RATIO = 0.9


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


def explore_data(data):
    # todo: sample and show random sentences from the data, and its label
    pass


def main():
    print(f'in main()...')
    # prime_nltk()
    train_set, test_set = get_train_and_test_sets()
    # explore_data(train_set)
    

if __name__ == '__main__':
    main()
