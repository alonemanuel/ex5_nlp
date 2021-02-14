from copy import copy


def sandbox0():
    dict0 = {0: 1, 1: 2, 3: 4}
    dict1 = {0: 1, 1: 2, 3: 4}
    print(f'dict0: {dict0}')
    print(f'dict1: {dict1}')
    fakecopy0 = dict0
    realcopy1 = copy(dict1)
    fakecopy0[1] = 7
    realcopy1[1] = 7
    print(f'dict0: {dict0}')
    print(f'dict1: {dict1}')


def update_dict(dict0):
    dict0['is'] += 1


def main():
    sandbox0()


if __name__ == '__main__':
    main()
