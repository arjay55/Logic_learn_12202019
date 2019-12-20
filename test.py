from sympy.logic import SOPform


def genboolcombi(numbits):
    '''
    :param numbits: number of bits
    :return: list
    '''
    return list(map(list, list(product(range(2), repeat=numbits))))


zz = ['c','b','a']
minterms = [[0, 0, 1],
            [0, 1, 0],
            [0, 1, 1]]

dontcares = [[1, 0, 0],
             [1, 0, 1],
             [1, 1, 0],
             [1, 1, 1]]
logfunc = SOPform(zz, minterms, dontcares)
