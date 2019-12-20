import numpy as np


def listbin(nonbin,numbits):
    """
    converts int() result to a numpy array
    :param nonbin:
    :param numbits: max number of bits for creation of tleading zeros
    :return: numpy array of binary values
    """
    strbin=list(format(nonbin,'0'+str(numbits)+'b'))
    arrbin=np.array(strbin)
    return arrbin.astype(np.bool)

def addition1(numbits1,numbits2,outbits):
    """
    generator function where it adds 2 integers.
    Standard nested for loop to produce all combinations.

    Testcase may suck. design will result to a buffer on first loop before redesigning itself.
    To be used for functional testing for now.

    :param numbits1: number of bits of 1st addend
    :param numbits2: number of bits of 2nd addend
    :param outbits: number of output bits
    :return: addend1,addend2,sum
    """

    stop1=2**numbits1
    stop2=2**numbits2
    for x in range(stop1):
        for y in range(stop2):
            sum=x+y
            yield np.append(listbin(x,numbits1),listbin(y,numbits2)),listbin(sum,outbits)

