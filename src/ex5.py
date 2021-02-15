from src.data_loader import DataLoader
from src.mst_parser import MSTParser

N_EPOCHS = 2
LR = 1


def main():
    data_loader = DataLoader()
    train_set, test_set = data_loader.get_train_and_test_sets()
    mst_parser = MSTParser(train_set)
    mst_parser.train(N_EPOCHS, LR)
    mst_parser.evaluate(test_set)


if __name__ == '__main__':
    main()
