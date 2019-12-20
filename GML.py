# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:16:05 2017

@author: TEUSER20
"""

import random
import string
from copy import deepcopy
from itertools import product, permutations
import math
import numpy as np
from sympy.logic import SOPform, false, true

import gmltests


# a boolean based non probabilistic GML where all inputs have no probabilistic factors in it.
# depth is a list or dict.
# fixed controlling time and

# use a class to encapsulate a boolean node consisting of minterms and dontcares
# stores the list of connectivity coordinates.

# Notes:
# 1. As a reminder, each training sample will create a layer, then simplification will occur for a definite time.
# 2. Removal of a node is part of simplification if it results to a buffer gate.


def toint(symbool):
    """
    converts sympy sympy true/false to 1/0
    :param symbool: symbool object
    :return: integer 1 or 0
    """
    if symbool.subs({}):  # 03112019 the braces don't contain arguments, and why does node_g_1 already true?
        return 1
    else:
        return 0


def tosym(integer):
    """
    transforms the value to a sympy true/false
    :param integer: 1 or 0
    :return: sympy true/false
    """

    if integer:
        return true
    else:
        return false


def genboolcombi(numbits):
    """
    :param numbits: number of bits
    :return: list
    """
    return list(map(list, list(product(range(2), repeat=numbits))))


def sortkeys(stringlist):
    """
    :param stringlist: list of keys
    :return: sorted list of keys where input nodes comes first
    """

    if "in" in stringlist:
        telist = sorted(stringlist[1:])
        return sortkeys_process_list(telist)

    else:
        telist = sorted(stringlist)
        return sortkeys_process_list(telist)


def sortkeys_process_list(telist):
    """
    for refactoring purposes only
    :param telist:
    :return:
    """
    telist = [int(x) for x in telist]
    telist.insert(0, 'in')
    return [str(x) for x in telist]


def getnodegrid(nodedict):
    """
    #parses keys of the nodes and returns 2d array with rows of letter strings and columns as depth. Fills empty cells with None of course
    :return:
    """
    xkeys = list(nodedict.keys())
    depthdict = {}
    for x in xkeys:
        te01 = x.split('_')
        if te01[-1] in depthdict.keys():
            depthdict[te01[-1]].append(x)
        else:
            depthdict[te01[-1]] = [x]
    size = [0, 0]
    for x in depthdict:
        size[0] = max(size[0], len(depthdict[x]))
        size[1] += 1
    dt = np.dtype('U16')
    nodegrid = np.empty(size, dtype=dt)  # nodegrid[depth,letterstring]
    # place to nodegrid
    laysorted = sortkeys(list(depthdict.keys()))
    for x in range(len(laysorted)):
        for z in range(len(depthdict[laysorted[x]])):
            nodegrid[z, x] = depthdict[laysorted[x]][z]
    return nodegrid


class gennamepart:
    def __init__(self):
        self.letters = [None]

    def letterall(self, thelist):
        """
        :param thelist--> processes the integers to actual ascii characters.
        :return:
        """
        for x in range(len(thelist)):
            self.letters[x] = string.ascii_letters[thelist[x]]

        return ''.join(self.letters)

    def reset(self):
        self.letters = [None]
        return


def lettergen():
    """
    letterclass--> class that generates the letter string
    generator that generates string.ascii indices prior to generating boolnode letter name part
    :return:
    """
    # to reset the generator, simply delete the generator and reassign again.
    letterclass = gennamepart()
    numletters = 26
    numlist = [0]

    for x in range(1, numletters):
        numlist[-1] = x
        yield letterclass.letterall(numlist)  # caused an exception

    numlist.insert(0, 1)
    numlist[-1] = 0
    letterclass.letters.append(None)
    yield letterclass.letterall(numlist)
    while (True):
        for x in range(1, numletters):
            numlist[-1] = x
            yield letterclass.letterall(numlist)
        for x in (len(numlist) - 2, -1, -1):
            if (numlist[x] < (numletters - 1)):
                numlist[x] += 1
                numlist[x + 1] = 0
                yield letterclass.letterall(numlist)
                break
            else:
                if (len(numlist) < (len(numlist) - x) + 1):
                    numlist.insert(0, 1)
                    letterclass.letters.append(None)
                    numlist[x] = 0
                    numlist[x + 1] = 0
                    yield letterclass.letterall(numlist)
                    break
                else:
                    continue


def trunc_inputs(allinputs, maxinput):
    """
    if allinputs exceed maxinput, shuffle and return
    :param allinputs:
    :param maxinput:
    :return:
    """
    if (len(allinputs) > maxinput):
        random.shuffle(allinputs)

    return allinputs[:maxinput]


def insert0(boolist):
    """
    :param boolist: list of binary lookups
    :return
    """
    if not boolist:
        return boolist

    npboolist = np.array(boolist)
    npboolist = np.insert(npboolist, 0, 0, axis=1)
    return npboolist.tolist()


class BoolNode:
    def __init__(self, inputnamelist, layno, nodepos, objname):
        # symbols are in letters. use ascii in python
        # initialize the values
        # perform initial simp. (just a buffer)
        # boolean logic is stored on a variable.
        self.nodelogic = deepcopy(inputnamelist)
        self.dontcares = genboolcombi(len(inputnamelist))
        self.minterms = []
        self.layno = layno
        self.nodepos = nodepos
        # self.classdict = getattr(globals[objname], 'classdict')
        self.output = false
        self.fanout = set()
        self.boolfunc = false
        self.objname = objname

    def getbooldict(self):
        """
        gets the attribute 'output' from the boolnode and forms a list
        :return: dict containing input names and its output values
        """
        booldict = {}
        for x in self.nodelogic:
            te00 = getattr(globals()[self.objname], 'classdict')
            booldict[x] = te00[x].output
        return booldict

    def SOPcompute(self):
        """
        refactored SOP computation of the nodes minterms and dontcares
        :return:
        """
        self.boolfunc = SOPform(self.nodelogic, self.minterms, self.dontcares)
        self.output = self.boolfunc.subs(self.getbooldict())


class InitNode:
    def __init__(self):
        self.output = false
        self.fanout = set()


class BoolAGI:

    def __init__(self, noinputs, nooutputs, maxinput, nodeprefix='node', maxpermute=-1):
        """
        --> basic model only simplify immediately after one training example.

        """
        self.batchsize = 5
        self.maxtimesimp = 120  # in seconds
        self.noinputs = noinputs
        self.nooutputs = nooutputs
        self.nodeprefix = nodeprefix
        self.classdict = {}
        self.inputlist = []
        self.outnodes = []
        self.arrout = np.empty(nooutputs, dtype=np.bool)
        self.depth = 0
        self.maxinput = maxinput
        self.nodegrid = 0
        self.dest_stack = []
        self.maxpermute = maxpermute  # -1 means no limit.
        # generate input containers
        xlettergen = lettergen()
        for _ in range(noinputs):
            testr = self.nodeprefix + '_' + next(xlettergen) + '_' + 'in'
            self.classdict[testr] = InitNode()
            self.inputlist.append(testr)
        self.nodegrid = getnodegrid(self.classdict)

        self.causation()

    def updatefanout(self, newnode, nodelist):
        """
        updates fanouts set of the node upon new connections
        does not apply on generatlization phase
        :param newnode: node to be connected with.
        :param nodelist: list of nodes to have the fanout attribute to be updated.
        :return:
        """
        for x in nodelist:
            self.classdict[x].fanout.add(newnode)

    def getcausal(self):
        """
        determines nodes to be connected from current output nodes.
        :return: a 2d list containing the outnode list as 0 and node to update at 1
        """
        self.nodegrid = getnodegrid(self.classdict)
        inputupd = []
        scale = 2  # higher number will make the network less forgetful.
        size = self.nodegrid.shape[1] - 2
        pdist = 1 / np.exp(np.arange(size, 0, -1) / scale)

        for x in range(len(pdist), 0, -1):
            te00 = self.nodegrid[:, x]
            namenz = np.nonzero(te00)
            namenz = te00[namenz[0]]
            for y in range(len(namenz)):
                te01 = []
                for z in self.outnodes:
                    if (np.random.random() < pdist[x - 1]):
                        te01.append(z)

                inputupd.append([te01, namenz[y]])

        return inputupd

    def updateinput(self, nodenet):
        """
        :param nodenet:--> list where 0 is the list of nets to be inserted and 1 is the node name
        :return:
        """
        if len(self.classdict[nodenet[1]].nodelogic) >= self.maxinput:
            return
        else:
            # do efficiency considerations first
            # insert 0 to existing lists

            selectednet = trunc_inputs(
                nodenet[0], self.maxinput - len(self.classdict[nodenet[1]].nodelogic))
            self.updatefanout(nodenet[1], selectednet)
            for x in selectednet:
                if x in self.classdict[nodenet[1]].nodelogic:
                    continue
                else:
                    self.classdict[nodenet[1]].dontcares = insert0(
                        self.classdict[nodenet[1]].dontcares)
                    self.classdict[nodenet[1]].minterms = insert0(
                        self.classdict[nodenet[1]].minterms)

                    # add new set of dontcares that are 1 in MSB
                    newdontcares = genboolcombi(
                        len(self.classdict[nodenet[1]].nodelogic))
                    npnewdontcares = np.array(newdontcares)
                    npnewdontcares = np.insert(npnewdontcares, 0, 1, axis=1)
                    newdontcares = npnewdontcares.tolist()
                    self.classdict[nodenet[1]].dontcares.extend(newdontcares)
                    # update the namelist
                    self.classdict[nodenet[1]].nodelogic.insert(0, x)
            return

    def update_nodes(self, nodenet):
        """
        updates the nodes from recurrent connectivities
        :param nodenet:--> list where 0 is the list of nets to be inserted and 1 is the node name
        :return:
        """
        # shall we do concurrency?
        # 1. generate the list
        # 2. if in the list, skip extending to dont cares
        for x in nodenet:
            self.updateinput(x)

    def causation(self):
        """
        constructs BoolNodes before output terminals.
        :return:
        """

        self.outnodes = []
        xlettergen = lettergen()
        # ct = 0

        # generate input connections
        self.nodegrid = getnodegrid(self.classdict)
        # input section
        curinputs = self.nodegrid[:, -1][np.nonzero(self.nodegrid[:, -1])]
        curinputst = np.transpose(curinputs).tolist()

        def gennodestd(curinputst):
            """
            refactored code for node generation.
            #first function in a method!
            :return:
            """

            curinputst = trunc_inputs(curinputst, self.maxinput)
            for _ in range(self.nooutputs):
                curletter = next(xlettergen)
                testr = self.nodeprefix + '_' + \
                        curletter + '_' + str(self.depth)
                self.classdict[testr] = BoolNode(
                    curinputst, self.depth, curletter, 'xagi')
                self.updatefanout(testr, self.inputlist)
                self.outnodes.append(testr)

            self.depth += 1

        gennodestd(curinputst)

    def sequential(self):
        """
        inserts requrrence nets on previous nodes. Does not evaluate
        :return:
        """
        nodeupd = self.getcausal()
        self.update_nodes(nodeupd)

    def toinputs(self, inputs):
        for x, y in zip(inputs, self.inputlist):
            self.classdict[y] = x

    # def train(self, twodarray):
    #     self.depth = 0
    #     print('[INFO-001] Start of training')
    #     self.ctdata = 0
    #     for x in range(self.batchsize):
    #         curdata = twodarray[self.ctdata:self.ctdata + self.batchsize]
    #         _ = self.causation()

    def gettrue(self, inputnames, oneddata):
        """
        evaluates if prediction is correct or not
        :param inputnames: node names
        :param oneddata: test data
        :return: boolean list
        """
        myarray = np.empty(len(inputnames), dtype=np.bool)

        for x in range(len(inputnames)):
            # TODO: must resolve this with good efficiency
            myarray[x] = toint(
                self.classdict[inputnames[x]].output) == oneddata[x]

        return myarray

    def getboollist(self, nodename):
        """
        gets the attribute 'output' from the boolnode and forms a list
        :param nodename: node names
        :return: list of Booleans
        """
        boollist = []
        try:
            for x in self.classdict[nodename].nodelogic:
                boollist.append(toint(self.classdict[x].output))
        except:
            # reached 1st layer
            return

        return boollist

    def logicsimp(self, inputname, boollist, bit):
        """
        perform logic simplification
        :param inputname: BoolNode name
        :param boollist: input terms
        :param bit: training bit
        :return:
        """
        print("logic simplification being performed")

        if (bit):
            self.classdict[inputname].minterms.append(boollist)
            del self.classdict[inputname].dontcares[self.classdict[inputname].dontcares.index(
                boollist)]
            self.classdict[inputname].SOPcompute()
        else:
            del self.classdict[inputname].dontcares[self.classdict[inputname].dontcares.index(
                boollist)]
            self.classdict[inputname].SOPcompute()

    def loaddata(self, inputarray):
        """
        loads input values to input nodes
        :return:
        """
        names = self.nodegrid[:, 0][np.nonzero(self.nodegrid[:, 0])]
        # for parallel
        for x in range(len(names)):
            self.classdict[names[x]].output = tosym(inputarray[x])

    # def destroy_conns
    def backup_gate(self, inputnames):
        """
        :param inputnames: name of the gates
        :return: list of gate object (dictionary type)
        """
        tedict = {}
        for x in inputnames:
            tedict[x] = deepcopy(self.classdict[x])

        return tedict  # recursive alert!

    def return_gate(self, backup_gates):
        """
        :param backup_gates: dictionary containg the original gate dictionaries
        :return:
        """

        for x in backup_gates:
            self.classdict[x] = backup_gates[x]

    def backpropogate_spec(self, inputnames, oneddata):
        """
        specific backprop for a list of inputs
        :param inputnames: list of nodes
        :param oneddata: array of one dim data
        :return: returns when one line succeeds or destroy conns and nodes to be occurred.
        v0.2--> has output permutations.
        """
        # backup gates first.
        # bk_gates = self.backup_gate(inputnames)
        leninputs = len(inputnames)
        permutgen = permutations(range(leninputs), leninputs)
        print("length is {}".format(leninputs))
        for iterlist in permutgen:
            break

        return self.backprop_unit(inputnames, iterlist, oneddata)

    def backprop_unit(self, inputnames, iterlist, oneddata):

        listtrue = self.gettrue(inputnames, oneddata)
        for x in iterlist:
            if not listtrue[x]:
                # condition 1, dontcare mapping exists.
                # 02172019: This may go to a convoluted flowchart
                boollist = self.getboollist(inputnames[x])
                if boollist is None:
                    # hit the input nodes
                    continue

                elif boollist in self.classdict[inputnames[x]].dontcares:
                    self.logicsimp(inputnames[x], boollist, oneddata[x])
                    listtrue = self.gettrue(inputnames, oneddata)

                elif len(self.classdict[inputnames[x]].dontcares) > 0:
                    # this complicates the matter on iterations...
                    if self.backpropogate_spec(self.classdict[inputnames[x]].nodelogic,
                                               self.classdict[inputnames[x]].dontcares[0]) == -1:
                        break
                else:
                    continue

        if not np.all(listtrue):
            return -1  # unit backpropagation failure.

    def backpropagation(self, stimuli):
        """
        v0.1--> no mistake memory, meaning, if connectivity collapse occurs, it is not remembered.
        :param stimuli: target data
        """
        # 05102019 add another loop for different permutations.
        names = self.nodegrid[:, -1][np.nonzero(self.nodegrid[:, -1])]
        return self.backpropogate_spec(names, stimuli)

    def getoutput(self):
        """
        creates an array based on output nodes.
        :return: array based on output nodes.
        """
        for x in range(len(self.outnodes)):
            nodename = self.nodegrid[x, -1]
            self.arrout[x] = toint(self.classdict[nodename].output)

    def forwardpropagation(self, inputarray, traindata):
        """
        perform forward propagation
        prime candidate for parallelization.
        :param inputarray:
        :param traindata:
        :return: array of output values (numpy array)
        """

        # will  attempt to reuse some created functions
        # 1. getboollist
        # 2.
        self.nodegrid = getnodegrid(self.classdict)
        self.loaddata(inputarray)
        for x in range(1, self.nodegrid.shape[1]):
            curinputs = self.nodegrid[:, x][np.nonzero(self.nodegrid[:, x])]
            for y in curinputs:
                self.classdict[y].output = self.classdict[y].boolfunc.subs(
                    self.classdict[y].getbooldict())

        return self.getoutput()


def maintrain(gmlmainobj, indepvar, depvar):
    """
    wrapper for standard gml training
    :param gmlmainobj: main gml object
    :param indepvar: input vars
    :param depvar: outputvars
    :return:
    """
    gmlmainobj.loaddata(x)
    # causation and sequential
    gmlmainobj.causation()
    gmlmainobj.sequential()
    return gmlmainobj.backpropagation(depvar)


def mtrain_caus0_seq0(gmlmainobj, depvar):
    """
    wrapper for standard gml training
    no sequential
    :param gmlmainobj: main gml object
    :param indepvar: input vars
    :param depvar: outputvars
    :return:
    """
    gmlmainobj.loaddata(x)
    return gmlmainobj.backpropagation(depvar)


if __name__ == '__main__':
    # test pattern 1:
    # addition of 2 8 bit numbers
    # single sample training
    # full causation, sequential
    # inference will occur iteratively
    # training will occur on the error, then inferencing will restart
    # above methodology can also be a reinforcement learning technique as well.
    # TODO: [FEATURE] Architectural creation when noinputs > maxinput
    # TODO: add separate attribute max feedforward input where max_feed < maxinput
    oneaddend = 5
    inputbits = oneaddend * 2
    maxtrain = 2 ** 16
    outputbits = oneaddend + 1

    xagi = BoolAGI(inputbits, outputbits,
                   math.ceil(math.log(350e3, 2)), 'node')

    # 2 feedforward layers
    xagi.causation()
    endtrain = 0
    fullstop = 0
    while endtrain < maxtrain:
        # restart generator
        if fullstop == -1:
            print('[last loop] failure to learn. Need topological changes')
            break
        xaddition1 = gmltests.addition1(oneaddend, oneaddend, outputbits)
        ct = 0
        for x, y in xaddition1:
            if fullstop == -1:
                print('failure to learn. Need topological changes')
                break
            else:
                xagi.forwardpropagation(x, y)
                if np.bitwise_xor(xagi.arrout, y).any():
                    if mtrain_caus0_seq0(xagi, y) == -1:
                        fullstop = -1
                    endtrain = ct
                    print('Fix attempted. Correct {} values from the start'.format(endtrain))
                    endtrain
                    break
                else:
                    mtrain_caus0_seq0(xagi, y)
                    ct += 1
