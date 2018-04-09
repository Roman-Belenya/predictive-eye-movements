import numpy as np


def dispersion(eyex, eyez, win):

    d = ( max(eyex[win[0]:win[1]]) - min(eyex[win[0]:win[1]]) ) + \
        ( max(eyez[win[0]:win[1]]) - min(eyez[win[0]:win[1]]) )

    return d


def distance(p1, p0):

    p0 = np.array(p0); p1 = np.array(p1)
    return np.sqrt(np.sum( (p1 - p0)**2, axis = 0 ))
