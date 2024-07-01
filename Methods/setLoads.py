import numpy as np
import pandas as pd
import py_dss_interface


class setLoads:
    def __init__(self, dss, network, noLoadQant):
        self.dss = dss
        # extract network parameters
        self.N = network["N"]
        self.avBusInd = network["avBusInd"]
        self.busNames = network["busNames"]
        self.Sbase = network["Sbase"]
        # no load quants
        self.v0 = np.squeeze(noLoadQant["v0"])

    # methods
    def get_Load(self):
        "Method to extract loads from a feeder"

        # preallocate
        nBuses = len(self.busNames)
        JbusSet = np.arange(0, 3*nBuses)
        phNodesLinInd = np.reshape(JbusSet, (3, nBuses), order="F").T
        self.sL = np.zeros((self.N*3, 2), dtype=complex) # constant power loads
        self.yL = np.zeros((self.N*3, 2), dtype=complex) # constant impedance loads
        self.iL = np.zeros((self.N*3, 2), dtype=complex) # constant current loads

        self.cMat = np.zeros((3*self.N, 2))

        # gMat defines the load type  gVec(1) PQ, gVec(2) I , gVec(3) Y
        self.gMat = np.zeros((3*self.N, 3))
        self.ePage = np.zeros((3*self.N, 3*self.N, 2))

        # create a dictionary from node names
        elems = self.dss.circuit_all_element_names()

        for i, elem in enumerate(elems):
            self.dss.circuit_set_active_element(elem)

            if "Load" in elem:
                # get bus name
                buses = self.dss.cktelement_read_bus_names()
                bus = buses[0].split(".")[0]

                # get nodes
                nodes = self.dss.cktelement_node_order()
                rows = [n - 1 for n in nodes if n != 0]

                # get number of nodes including reference
                n = len(rows)

                # reorder nodes
                # define columns and rows
                # extract load name
                loadName = elem.split(".")[1]
                
                # write load name
                self.dss.loads_write_name(loadName)
                Model = self.dss.loads_read_model()
                isDelta = self.dss.loads_read_is_delta()

                if isDelta != 0:
                    rows.pop(1)

                # indices
                nIndx = self.busNames.index(bus)
                linIdx0 = phNodesLinInd[nIndx, 0]
                linIdx1 = phNodesLinInd[nIndx, 1]
                linIdx2 = phNodesLinInd[nIndx, 2]

                # extract values
                if n == 3:
                    kw = self.dss.loads_read_kw() / n
                    kvar = self.dss.loads_read_kvar() / n
                else:
                    kw = self.dss.loads_read_kw()
                    kvar = self.dss.loads_read_kvar()

                # active power
                pLoad = np.zeros(3)
                pLoad[np.ix_(rows)] = kw * 1000 / self.Sbase

                # reactive power
                qLoad = np.zeros(3)
                qLoad[np.ix_(rows)] = kvar * 1000 / self.Sbase

                # aparent powero
                sLoad = pLoad + qLoad * 1j

                # cases
                if isDelta:
                    # define index matrices
                    self.__DindMatrices(linIdx0, linIdx1, linIdx2)

                    if Model == 1:  # constant power
                        self.__DPQ(linIdx0, linIdx1, linIdx2, sLoad)

                    if Model == 5:  # constant current
                        self.__DI(linIdx0, linIdx1, linIdx2, sLoad)

                    if Model == 2:  # constant impedance
                        self.__DZ(linIdx0, linIdx1, linIdx2, sLoad)
                else:
                    # define index matrices
                    self.__YindMatrices(linIdx0, linIdx1, linIdx2)

                    if Model == 1:  # constant power
                        self.__YPQ(linIdx0, linIdx1, linIdx2, sLoad)

                    elif Model == 5:  # constant current
                        self.__YI(linIdx0, linIdx1, linIdx2, sLoad)

                    elif Model == 2:  # constant impedance
                        self.__YZ(linIdx0, linIdx1, linIdx2, sLoad)

        sL_load = self.sL[np.ix_(self.avBusInd[:-3])]
        iL_load = self.iL[np.ix_(self.avBusInd[:-3])]
        gMat = self.gMat[np.ix_(self.avBusInd[:-3])]
        yL_load = self.yL[np.ix_(self.avBusInd[:-3])]
        ePage = self.ePage[np.ix_(self.avBusInd[:-3], self.avBusInd[:-3])]
        cMat = self.cMat[np.ix_(self.avBusInd[:-3])]

        # compute YL
        YL = self.__computeYL(yL_load, ePage, cMat) 

        # store results and return
        loadQant = dict()
        loadQant["YL"] = YL
        loadQant["yL"] = yL_load
        loadQant["ePage"] = ePage
        loadQant["cMat"] = cMat
        loadQant["sL"] = sL_load
        loadQant["iL"] = iL_load
        loadQant["gMat"] = gMat

        return loadQant

    # helper methods
    def __computeYL(self, yL_load, ePage, cMat):
        YL = cMat[:, 0] * yL_load[:, 0] * ePage[:, :, 0] + cMat[:, 1] * yL_load[:, 1] * ePage[:, :, 1]
        return YL 

    def __YindMatrices(self, linIdx0, linIdx1, linIdx2):
        # the following matrices help to accomodate powers
        # in matrix notation they are shared for all Y loads
        self.cMat[linIdx0, 0] = 1
        self.cMat[linIdx1, 0] = 1
        self.cMat[linIdx2, 0] = 1
        # ePage matrix
        self.ePage[linIdx0, linIdx0, 0] = 1
        self.ePage[linIdx1, linIdx1, 0] = 1
        self.ePage[linIdx2, linIdx2, 0] = 1

    def __DindMatrices(self, linIdx0, linIdx1, linIdx2):
        # the following matrices help to accomodate powers
        # in matrix notation they are shared for all Y loads
        self.cMat[linIdx0, 0] = 1
        self.cMat[linIdx1, 0] = 1
        self.cMat[linIdx2, 0] = 1
        self.cMat[linIdx0, 1] = -1
        self.cMat[linIdx1, 1] = -1
        self.cMat[linIdx2, 1] = -1
        # ePage matrix first page
        self.ePage[linIdx0, linIdx0, 0] = 1
        self.ePage[linIdx0, linIdx1, 0] = -1
        self.ePage[linIdx1, linIdx1, 0] = 1
        self.ePage[linIdx1, linIdx2, 0] = -1
        self.ePage[linIdx2, linIdx2, 0] = 1
        self.ePage[linIdx2, linIdx0, 0] = -1
        # ePage matrix second page
        self.ePage[linIdx0, linIdx2, 1] = 1
        self.ePage[linIdx0, linIdx0, 1] = -1
        self.ePage[linIdx1, linIdx0, 1] = 1
        self.ePage[linIdx1, linIdx1, 1] = -1
        self.ePage[linIdx2, linIdx1, 1] = 1
        self.ePage[linIdx2, linIdx2, 1] = -1

    def __YPQ(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Yload with cte power """
        self.sL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += sLoad
        # keeping track
        self.gMat[linIdx0, 0] = 1
        self.gMat[linIdx1, 0] = 1
        self.gMat[linIdx2, 0] = 1

    def __YI(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Yload with cte current """
        self.iL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += np.conj(sLoad/self.v0)
        # keeping track
        self.gMat[linIdx0, 1] = 1
        self.gMat[linIdx1, 1] = 1
        self.gMat[linIdx2, 1] = 1

    def __YZ(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Yload with cte impedance """
        # in general the impedance should be "constant" thus yl = sLoad*/1pu
        self.yL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += np.conj(sLoad)
        # keeping track
        self.gMat[linIdx0, 2] = 1
        self.gMat[linIdx1, 2] = 1
        self.gMat[linIdx2, 2] = 1

    def __DPQ(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Dload with cte power """
        self.sL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += sLoad
        self.sL[linIdx0, 1] += sLoad[2]
        self.sL[linIdx1, 1] += sLoad[0]
        self.sL[linIdx2, 1] += sLoad[1]
        # keeping track
        self.gMat[linIdx0, 0] = 1
        self.gMat[linIdx1, 0] = 1
        self.gMat[linIdx2, 0] = 1

    def __DI(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Dload with cte current """
        v0ab = self.v0[0] - self.v0[1]
        v0bc = self.v0[1] - self.v0[2]
        v0ca = self.v0[2] - self.v0[0]
        v0d = np.array([v0ab, v0bc, v0ca])

        self.iL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += np.conj(sLoad / v0d)
        self.iL[linIdx0, 1] += np.conj(sLoad[2] / v0ca)
        self.iL[linIdx1, 1] += np.conj(sLoad[0] / v0ab)
        self.iL[linIdx2, 1] += np.conj(sLoad[1] / v0bc)
        # keeping track
        self.gMat[linIdx0, 1] = 1
        self.gMat[linIdx1, 1] = 1
        self.gMat[linIdx2, 1] = 1

    def __DZ(self, linIdx0, linIdx1, linIdx2, sLoad):
        """ Dload with cte impedance """
        # 3 is for the delta z relationship (line to line quantities)
        self.yL[np.ix_([linIdx0, linIdx1, linIdx2]), 0] += np.conj(sLoad / 3)
        self.yL[linIdx0, 1] += np.conj(sLoad[2] / 3)
        self.yL[linIdx1, 1] += np.conj(sLoad[0] / 3)
        self.yL[linIdx2, 1] += np.conj(sLoad[1] / 3)
        # keeping track
        self.gMat[linIdx0, 2] = 1
        self.gMat[linIdx1, 2] = 1
        self.gMat[linIdx2, 2] = 1
