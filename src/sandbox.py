from collections import Counter


def sandbox0():
    keys = [0,1,2,3,4]
    vals = [3,4,5,23]
    dict0 = dict(zip(keys, vals))
    print (dict0)

    new_dict = dict((key, val*2) for (key,val) in dict0.items())
    print(new_dict)
def update_dict(dict0):
    dict0['is']+=1

def main():
    sandbox0()


if __name__ == '__main__':
    main()
