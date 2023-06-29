# generate unlabeled circuit DAG datasets for training autoencoders
import random
import numpy as np
from util import *

set_num           = 10000

if __name__=="__main__":
    
    datafile = open('./dataset_withoutY_1w.txt','w')
    #datafile = open('./dataset_test.txt','w')
    #datafile = open('./dataset_fullRandom.txt','w')

    for c in range(1, set_num+1):
        topo_vector = sample_topo_vector(CIRCUIT_EDGE_NUM)
        CIRCUIT_DAG = vector2row(topo_vector)
        datafile.writelines(str(CIRCUIT_DAG)+'\n')

    '''
    for c in range(1, set_num+1):
        topo_vector = sample_full_random(10)
        CIRCUIT_DAG = vector2row(topo_vector)
        datafile.writelines(str(CIRCUIT_DAG)+'\n')
    '''

    print('finish {} random circuit DAGs generating !'.format(set_num))
    datafile.close()
