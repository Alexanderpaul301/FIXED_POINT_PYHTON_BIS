import numpy as np
from numpy.linalg import inv


def noLoadV(network, v0mags=None, v0phases=None):
    if v0mags is None or v0phases is None:
        v0mags = np.array([1, 1, 1])
        v0phases = np.array([0, -120, 120])

    # extract network quantities
    Y = network["Y"]
    Y_NS = network["Y_NS"]
    N = network["N"]
    avBusIndices = network["avBusInd"]

    # Setting up slack fixed voltage and finding the no-load solution
    v0phases = np.deg2rad(v0phases)
    v0 = np.expand_dims(v0mags * np.exp(1j*v0phases), axis=1)  # phasor

    # w = Z @ Y_{NS} @ vs (source)
    w = -inv(Y) @ Y_NS @ v0

    wCheck = np.zeros((3*(N+1), 1), dtype=complex)
    wCheck[np.ix_(avBusIndices)] = np.vstack([w, v0])

    # wCheck(missingBusIndices,1) = NaN
    w3phase = np.reshape(wCheck, (3, N+1), order="F").T

    # save
    noLoadQant = dict()
    noLoadQant["v0mags"] = v0mags 
    noLoadQant["v0phases"] = v0phases
    noLoadQant["v0"] = v0
    noLoadQant["w"] = w
    noLoadQant["wCheck"] = wCheck
    noLoadQant["w3phases"] = w3phase

    return noLoadQant
