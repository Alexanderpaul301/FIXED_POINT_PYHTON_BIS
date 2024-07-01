import numpy as np
import py_dss_interface
from numpy.linalg import inv
from numpy.matlib import repmat


class Zbus:

    def __init__(self, network, noLoadQant, loadQant):
        self.avBusInd = network["avBusInd"]
        self.N = network["N"]
        self.Y = network["Y"]
        self.Y_NS = network["Y_NS"]
        self.v0 = noLoadQant["v0"]
        self.loadQant = loadQant
        self.YL = loadQant["YL"]

    def perform_Zbus(self, maxIter = 1000):
        # initialization with flat start
        vPr = repmat(self.v0, self.N, 1)
        vPr = vPr[np.ix_(self.avBusInd[:-3])]
        vPr_fv = np.squeeze(vPr)
        vIterations = np.zeros((len(vPr), maxIter), dtype=complex)
        vIterations[:, 0] = vPr_fv
        fv = self.__compIPQII(vPr_fv)
        err = np.sum(abs((self.Y + self.YL) @ vPr + fv + self.Y_NS @ self.v0))
        it = 0
        while err > 1e-6 and it < maxIter:
            vNew = inv(self.Y + self.YL) @ (- fv - self.Y_NS @ self.v0)
            vNew_fv = np.squeeze(vNew)
            vIterations[:, it+1] = vNew_fv
            fv = self.__compIPQII(vNew_fv)
            err = np.sum(abs((self.Y + self.YL) @ vNew + fv + self.Y_NS @ self.v0))
            it += 1

        vsol = np.vstack([vNew, self.v0])
        v = np.zeros((3*(self.N+1), 1), dtype=complex)
        v[self.avBusInd] = vsol
        return v

    # helper methods
    def __compIPQII(self, vPr):
        gMat = self.loadQant["gMat"]
        cMat = self.loadQant["cMat"]
        ePage = self.loadQant["ePage"]
        sL = self.loadQant["sL"]
        iL = self.loadQant["iL"]
        yL = self.loadQant["yL"]
        # set the gMat column to zero since yL is already considered in the
        # Ybus using getYL method in setLoad class
        gMat[:, 2] = 0
        nvars = len(vPr)
        fv = np.zeros((nvars, 1), dtype=complex)

        for i in range(0, nvars):
            eMat = np.reshape(ePage[i, :, :], (nvars, 2), order="F")
            eVec1 = eMat[:, 0]
            eVec2 = eMat[:, 1]
            # compute constant power currents
            auxPQ = [self.__fPQ(eVec1 @ vPr, sL[i, 0]),
                     self.__fPQ(eVec2 @ vPr, sL[i, 1])]
            iL_PQ = cMat[i, :] @ np.array(auxPQ)
            # compute constant current
            auxI = np.array([iL[i, 0], iL[i, 1]])
            iL_I = cMat[i, :] @ auxI
            # compute constant impedance current
            auxZ = [self.__fZ(eVec1 @ vPr, yL[i, 0]),
                    self.__fZ(eVec2 @ vPr, yL[i, 1])]
            iL_Y = cMat[i, :] @ np.array(auxZ)
            fv[i] = gMat[i, :] @ np.array([iL_PQ, iL_I, iL_Y])
        return fv

    def __fPQ(self, v, s):
        """
        FPQ: Constant PQ Load i=fPQ(v,s) outputs the single phase current
        i, given the single phase inputs v, and the single phase apparent power
        consumption s
        """
        if abs(v) > 0:
            out = np.conj(s/v)
        else:
            out = 0 + 0 * 1j
        return out

    def __fZ(self, v, y):
        """
        FZ: Constant Z Load i=fZ(v,y) outputs the single phase current
        i, given the single phase inputs v, and the single phase addmitance y
        """
        out = y * v
        return out
