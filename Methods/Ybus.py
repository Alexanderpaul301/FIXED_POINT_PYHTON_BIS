import py_dss_interface
import numpy as np
import pandas as pd
from scipy import sparse
from numpy.linalg import inv


class Ybus:
    def __init__(self, dss, regTypes='ideal', epsilon=1e-5, kva_base=5000, kvll_base=4.16):
        # initialization
        self.dss = dss
        self.regTypes = regTypes
        self.epsilon = epsilon

        # processing
        self.Sbase = kva_base*1000  # from the substation transformer
        self.Vbase = (kvll_base * 1000)/np.sqrt(3)  # secondary of the substation transformer (L-N)
        self.Zbase = self.Vbase**2/self.Sbase
        self.Ybase = 1/self.Zbase

    # Methods
    def define_Ybus(self):

        # dss elements
        elements = self.dss.circuit_all_element_names()

        # get buses names per type of element (lines, sws, trafos, caps.)
        buses_pd, busesReg_pd, sBus, capNames, capBuses = self.__getBusNames(elements)

        # unique buses (\mathcal{N})
        busNames = np.unique(buses_pd.values.reshape((buses_pd.size, 1), order='F')).tolist()
        busNamesWithRegs = np.unique(busesReg_pd.values.reshape((busesReg_pd.size, 1), order='F')).tolist()

        # move substation bus at the end
        busNames.remove(sBus)
        busNames.append(sBus)  # assign substation at the end
        busNamesWithRegs.remove(sBus)
        busNamesWithRegs.append(sBus)  # assign substation at the end

        # number of buses
        nBuses = len(busNames)

        # number of buses minus substation
        N = nBuses - 1

        # number of branches
        nBr = len(buses_pd)  

        # The Ybus will be created by expanding the buses using 3 by 3 blocks (to account for three phases).  
        self.CIndices = np.zeros((nBr*3, nBuses*3), dtype=complex)
        self.Ytilde = np.zeros((3*nBuses, 3*nBuses), dtype=complex)
        self.Ybranch = np.zeros((3*nBr, 3*nBr), dtype=complex)

        # loop over organized elements 
        for ii, element in enumerate(buses_pd.index):

            # define the linear index (for each phase of each bus) for this element
            jBranchIdx = np.arange(ii*3, (ii+1)*3)

            # get the buses for this element
            bus1 = buses_pd.loc[element, 0]
            bus2 = buses_pd.loc[element, 1]

            # get the index for these buses from the unique buses vector
            nIdx = busNames.index(bus1)
            mIdx = busNames.index(bus2)

            # define the position in which the phase indices will be located
            jnIdx = np.arange(nIdx*3, (nIdx+1)*3)
            jmIdx = np.arange(mIdx*3, (mIdx+1)*3)

            if element == "Line.sw1":  # reg1 is connected to sw1, in the formulation these form a single element
                # 3-phase gang reg
                tap = [7, 7, 7]
                arA = 1-0.00625*tap[0]
                arB = 1-0.00625*tap[1]
                arC = 1-0.00625*tap[2]

                # gain
                Av = np.array([[arA, 0, 0], [0, arB, 0], [0, 0, arC]])

                # gain inverse
                Ai = inv(Av)

                # regulator impedance impedance
                ztReg = (self.Sbase/5000000)*3*0.00001j
                Zreg = np.array([[ztReg/arA, 0, 0], [0, ztReg/arB, 0], [0, 0, ztReg/arC]])

                # modify network matrices
                self.__getRegYprim(element, Av, Ai, Zreg, jnIdx, jmIdx, jBranchIdx)

            elif element == "Line.l11": # Line l11 is connected after regulator 2, in the formulation these two form a single element
                # single-phase reg
                tapA = -1
                arA = 1-0.00625*tapA

                # gain
                Av = np.array([[arA, 0, 0], [0, 0, 0], [0, 0, 0]])
                Ai = np.array([[1/arA, 0, 0], [0, 0, 0], [0, 0, 0]])

                # impedande
                ztReg = (self.Sbase/2000000)*1*0.0001j
                Zreg = np.array([[ztReg/arA, 0, 0], [0, 0, 0], [0, 0, 0]])

                # modify network
                self.__getRegYprim(element, Av, Ai, Zreg, jnIdx, jmIdx, jBranchIdx)

            elif element == "Line.l25": # Line l25 is connected after regulator 3, in the formulation these two form a single element  # reg3
                # single-phase reg
                tapA = 0
                tapC = -1
                arA = 1-0.00625*tapA
                arC = 1-0.00625*tapC

                # gain
                Av = np.array([[arA, 0, 0], [0, 0, 0], [0, 0, arC]])
                Ai = np.array([[1/arA, 0, 0], [0, 0, 0], [0, 0, 1/arC]])

                # regulator impedande
                ztReg = (self.Sbase/2000000)*1*0.0001j
                Zreg = np.array([[ztReg/arA, 0, 0], [0, 0, 0], [0, 0, ztReg/arC]])

                # modify network
                self.__getRegYprim(element, Av, Ai, Zreg, jnIdx, jmIdx, jBranchIdx)

            elif element == "Line.l117": # Line l117 is connected after regulator 4, in the formulation these two form a single element
                # 3phase individual reg
                tap = [8, 1, 5]
                arA = 1-0.00625*tap[0]
                arB = 1-0.00625*tap[1]
                arC = 1-0.00625*tap[2]

                # gain
                Av = np.array([[arA, 0, 0], [0, arB, 0], [0, 0, arC]])

                # gain inverse
                Ai = inv(Av)

                # regulator impedance
                ztReg = (self.Sbase/2000000)*1*0.0001j
                Zreg = np.array([[ztReg/arA, 0, 0], [0, ztReg/arB, 0], [0, 0, ztReg/arC]])

                # modify network
                self.__getRegYprim(element, Av, Ai, Zreg, jnIdx, jmIdx, jBranchIdx)

            elif element == "Transformer.xfm1":
                self.__getTrafoYprim(jnIdx, jmIdx, jBranchIdx)

            else:
                self.__getLineYprim(element, jnIdx, jmIdx, jBranchIdx)

        # add capacitors
        Ycap = np.zeros(self.Ytilde.shape, dtype=complex)
        for ii, cap in enumerate(capNames):

            # capacitors are connected to already existing buses, find them from \mathcal{N}:
            nIdx = busNames.index(capBuses[cap])

            # extendt to 3phase case
            capIdx = np.arange(nIdx*3, (nIdx+1)*3)

            # set cap as active element
            self.dss.circuit_set_active_element(cap)

            # get nodes (phases) to which each cap is connected
            nodes = self.dss.cktelement_node_order()

            # get number of nodes including reference
            n = len(nodes)

            # reorder nodes
            nodes = np.asarray(nodes).reshape((int(len(nodes)/2), -1), order="F")
            row = nodes[:, 0] - 1
            col = nodes[:, 0] - 1

            # preallocate
            ycap = np.zeros((3, 3), dtype=complex)

            # set active cap
            self.dss.capacitors_write_name(cap.split(".")[1])

            # ybus from the capacitors is modeled as constant yprim:
            kva = (self.dss.capacitors_read_kvar() / (n/2)) * np.eye(int(n/2)) 

            # include in the preallocated ycap
            ycap[np.ix_(row, col)] = (kva *1j*1000)/self.Sbase

            # add to the specific position in Ynet
            Ycap[np.ix_(capIdx, capIdx)] = ycap

        # Finding available phases
        avBranchInd = np.where(self.CIndices.any(axis=1))[0]
        avBusAppInd = np.where(self.CIndices.any(axis=0))[0]
        avBusInd = np.where(self.Ytilde.any(axis=1))[0]

        # availableBusAppIndices and availableBusIndices should match
        YbusS = self.Ytilde[np.ix_(avBusInd, avBusInd)]
        YcapS = Ycap[np.ix_(avBusInd, avBusInd)]


        # define Ynet
        Ynet = YbusS + YcapS
        Y = Ynet[:-3, :-3]
        Y_NS = Ynet[:-3, -3:]
        Y_SS = Ynet[-3:, -3:]

        # the rank deficient admittance matrix
        Ybus = Y

        # save outputs in dict
        network = dict()
        network["busNames"] = busNames
        network["Sbase"] = self.Sbase
        network["N"] = N
        network["Y"] = Y
        network["Y_NS"] = Y_NS
        network["Y_SS"] = Y_SS
        network["Ybus"] = Ybus
        network["avBusInd"] = avBusInd

        return network

    # Helper methods

    def __getBusNames(self, elements):
        """"This method is designed to get the buses from all power delivery elements in a distribution circuit, 
            The ouput are data frames of the following format:
            
            buses_pd:             bus 1,   bus 2
                      element 1
                      elemment 2
                      ....
            
            buses_reg_pd:         bus 1,   bus 2
                      element 1
                      elemment 2
                      ....
            
            sBus => id
            capNames => list 
            capBuses => dict[id] = [bus]
            """

        # extract element names from OpenDSS
        # for lines
        lineNames = list()
        lineBuses = dict()

        # for Switches
        swNames = list()
        swBuses = dict()

        # for Trafos
        trafoNames = list()
        trafoBuses = dict()

        # for Regulation devices
        regNames = list()
        regBuses = dict()

        # for Capacitors
        capNames = list()
        capBuses = dict()

        for i, element in enumerate(elements):
            # set element as active in opendss
            self.dss.circuit_set_active_element(element)

            # read the buses from active element
            buses = self.dss.cktelement_read_bus_names()

            # get bus connected at the begining of the element
            buses1 = buses[0]

            # try to get bus connected at the end (not all elements have a bus at the end, e.g., loads)
            if np.size(buses) > 1:
                buses2 = buses[1]

            # check element type
            if "Vsource" in element:
                sName = element
                sBus = buses1.split('.')[0]

            elif "Line" in element:
                senBus = buses1.split('.')[0].replace("r", "")
                recBus = buses2.split('.')[0].replace("r", "")
                if "sw" in element:
                    swNames.append(element)
                    swBuses[element] = [senBus, recBus]
                else:
                    lineNames.append(element)
                    lineBuses[element] = [senBus, recBus]

            elif "Transformer" in element:
                senBus = buses1.split('.')[0]
                recBus = buses2.split('.')[0]
                if "reg" in element:
                    regNames.append(element)
                    regBuses[element] = [senBus, recBus]
                else:
                    trafoNames.append(element)
                    trafoBuses[element] = [senBus, recBus]

            elif "Capacitor" in element:
                capNames.append(element)
                capBuses[element] = buses1.split('.')[0]

        # create data frame for lines
        lineBuses_pd = pd.DataFrame(np.asarray([lineBuses[line] for line in lineNames]), index=lineNames)

        # create data frame for sws
        swBuses_pd = pd.DataFrame(np.asarray([swBuses[line] for line in swNames]), index=swNames)
        swBuses_pd = swBuses_pd.drop('Line.sw7')  # normally open sws (ignore)
        swBuses_pd = swBuses_pd.drop('Line.sw8')  # normally open sws (ignore)

        # create data frame for Trafos
        trafoBuses_pd = pd.DataFrame(np.asarray([trafoBuses[line] for line in trafoNames]), index=trafoNames)

        # create data frame for Regs
        regBuses_pd = pd.DataFrame(np.asarray([regBuses[line] for line in regNames]), index=regNames)

        # concatenate buses, not including regulators
        buses_pd = pd.concat([lineBuses_pd, swBuses_pd, trafoBuses_pd])
        
        # concatenate including regulators
        buses_reg_pd = pd.concat([buses_pd, regBuses_pd])

        return buses_pd, buses_reg_pd, sBus, capNames, capBuses

    def __getRegYprim(self, element, Av, Ai, Zreg, jnIdx, jmIdx, jBranchIdx):

        # get parts
        YtildeBlocks, YbranchII, Cmat = self.__getYparts(element)
        # reorder for better readability
        YtildeNMn = YtildeBlocks[0]
        YtildeNMm = YtildeBlocks[1]
        YtildeMNn = YtildeBlocks[2]
        YtildeMNm = YtildeBlocks[3]

        # Ytilde
        if self.regTypes == 'ideal':
            self.Ytilde[np.ix_(jnIdx, jnIdx)] += Ai @ (YtildeNMn) @ Ai.T
            self.Ytilde[np.ix_(jnIdx, jmIdx)] -= Ai @ YtildeNMm
            self.Ytilde[np.ix_(jmIdx, jnIdx)] -= YtildeMNn @ Ai.T
            self.Ytilde[np.ix_(jmIdx, jmIdx)] += YtildeMNm

        else:  
            # eq. 12 from Bazrafshan et al 2018
            Fr = np.eye(3) + YtildeNMn @ Ai.T @ Zreg @ Ai

            # in paragraph after eq. 18d from Bazrafshan et al 2018
            FriT = np.eye(3) - Ai.T @ Zreg @ Ai @ inv(Fr) @ YtildeNMn

            self.Ytilde[np.ix_(jnIdx, jnIdx)] += Ai @ inv(Fr) @ YtildeNMn @ Ai.T
            self.Ytilde[np.ix_(jnIdx, jmIdx)] -= Ai @ inv(Fr) @ YtildeNMm
            self.Ytilde[np.ix_(jmIdx, jnIdx)] -= YtildeMNn @ FriT @ Ai.T
            self.Ytilde[np.ix_(jmIdx, jmIdx)] += YtildeMNm - YtildeMNn @ Ai.T @ Zreg @ Ai @ inv(Fr) @ YtildeNMm

        # Ybranch
        self.Ybranch[np.ix_(jBranchIdx, jBranchIdx)] = YbranchII
        self.CIndices[np.ix_(jBranchIdx, jnIdx)] = Cmat
        self.CIndices[np.ix_(jBranchIdx, jmIdx)] = -Cmat

    def __getTrafoYprim(self, jnIdx, jmIdx, jBranchIdx):
        # prelocate
        YtildeNMn = np.zeros((3, 3), dtype=complex)
        YtildeNMm = np.zeros((3, 3), dtype=complex)
        YtildeMNn = np.zeros((3, 3), dtype=complex)
        YtildeMNm = np.zeros((3, 3), dtype=complex)
        YbranchII = np.zeros((3, 3), dtype=complex)
        Cmat = np.zeros((3, 3))

        # define impedances
        zt = 0.01*(1.27+2.72j)*(self.Sbase/150000)*3
        yt = 1/zt
        Y1 = np.diag([yt, yt, yt])
        Y2 = (1/3)*np.array([[2*yt, -yt, -yt],
                            [-yt, 2*yt, -yt],
                            [-yt, -yt, 2*yt]])
        # build Yhat
        Y2hat1 = Y2 + abs(yt)*self.epsilon*np.eye(3)
        Y2hat2 = Y2 + abs(yt)*(self.epsilon/2)*np.eye(3)

        row = col = np.array([0, 1, 2])

        # Ytilde
        YtildeNMn[np.ix_(row, col)] = Y2hat1
        YtildeNMm[np.ix_(row, col)] = Y2hat2
        YtildeMNn[np.ix_(row, col)] = Y2hat2
        YtildeMNm[np.ix_(row, col)] = Y2hat1

        # Ybranch
        YbranchII[np.ix_(row, col)] = Y2
        # Cmat
        Cmat[np.ix_(row, col)] = np.eye(len(row))

        # Ytilde
        self.Ytilde[np.ix_(jnIdx, jnIdx)] += YtildeNMn  # YtildeNMn
        self.Ytilde[np.ix_(jnIdx, jmIdx)] = -YtildeNMm
        self.Ytilde[np.ix_(jmIdx, jnIdx)] = -YtildeMNn
        self.Ytilde[np.ix_(jmIdx, jmIdx)] += YtildeMNm

        # Ybranch
        self.Ybranch[np.ix_(jBranchIdx, jBranchIdx)] = YbranchII

        # Cmat
        self.CIndices[np.ix_(jBranchIdx, jnIdx)] = Cmat
        self.CIndices[np.ix_(jBranchIdx, jmIdx)] = -Cmat

    def __getLineYprim(self, elem, jnIdx, jmIdx, jBranchIdx):
        # update Ytilde and Ybranch
        # get parts
        YtildeBlocks, YbranchII, Cmat = self.__getYparts(elem)

        # Ytilde
        self.Ytilde[np.ix_(jnIdx, jnIdx)] += YtildeBlocks[0]  # YtildeNMn
        self.Ytilde[np.ix_(jnIdx, jmIdx)] = -YtildeBlocks[1]  # YtildeNMm
        self.Ytilde[np.ix_(jmIdx, jnIdx)] = -YtildeBlocks[2]  # YtildeMNn
        self.Ytilde[np.ix_(jmIdx, jmIdx)] += YtildeBlocks[3]  # YtildeMNm

        # Ybranch
        self.Ybranch[np.ix_(jBranchIdx, jBranchIdx)] = YbranchII

        # Cmat
        self.CIndices[np.ix_(jBranchIdx, jnIdx)] = Cmat
        self.CIndices[np.ix_(jBranchIdx, jmIdx)] = -Cmat

    def __getYparts(self, elem):

        # preallocate
        YtildeNMn = np.zeros((3, 3), dtype=complex)
        YtildeNMm = np.zeros((3, 3), dtype=complex)
        YtildeMNn = np.zeros((3, 3), dtype=complex)
        YtildeMNm = np.zeros((3, 3), dtype=complex)
        YbranchII = np.zeros((3, 3), dtype=complex)
        Cmat = np.zeros((3, 3))

        # set the active circuit element
        self.dss.circuit_set_active_element(elem)

        # get nodes (phases) from the active element
        nodes = self.dss.cktelement_node_order()

        # get number of nodes including reference
        n = len(nodes)

        # reorder nodes
        nodes = np.asarray(nodes).reshape((int(len(nodes)/2), -1), order="F")

        # extract and organize yprim
        if "sw" in elem:
            Zseries = np.array([[0.4576+1.0780*1j, 0.1560+0.5017*1j, 0.1535+0.3849*1j],
                                [0.1560+0.5017*1j, 0.4666+1.0482*1j, 0.1580+0.4236*1j],
                                [0.1535+0.3849*1j, 0.1580+0.4236*1j, 0.4615+1.0651*1j]]) / 5280   # ohm per unit lenght

            Yshunt = 1j*(1e-6)*np.array([[5.6765, -1.8319, -0.6982],
                                         [-1.8319, 5.9809, -1.1645],
                                         [-0.6982, -1.1645, 5.3971]]) / 5280  # siemens per unit lenght

            lineLength = 1e-3

            # shunt elements only affect diagonal submatrices.
            yprim_diag = inv(Zseries * lineLength) + 0.5 * (Yshunt * lineLength)
            yprim_off = inv(Zseries * lineLength)

        else:
            # get actual line yprimitive from active element
            yprim_tmp = self.dss.cktelement_y_prim()
            yprim_tmp = np.asarray(yprim_tmp).reshape((2*n, n), order="F")
            yprim_tmp = yprim_tmp.T

            # accomodate diagonal and off-diagonal block matrices 
            yprim_diag = yprim_tmp[:int(n/2), :n]
            yprim_diag = yprim_diag[:, 0::2] + 1j * yprim_diag[:, 1::2] # create complex

            
            yprim_off = -yprim_tmp[:int(n/2), n:]  # negative in OpenDSS
            yprim_off = yprim_off[:, 0::2] + 1j * yprim_off[:, 1::2] # create complex
        
        # define columns and rows (opendss is 1 index)
        row = nodes[:, 0] - 1
        col = nodes[:, 1] - 1

        # Ytilde: ix defines at which phases the yprim actually belongs (to account for single or multiple phases)
        YtildeNMn[np.ix_(row, col)] = yprim_diag / self.Ybase
        YtildeNMm[np.ix_(row, col)] = yprim_off / self.Ybase
        YtildeMNn[np.ix_(row, col)] = yprim_off / self.Ybase
        YtildeMNm[np.ix_(row, col)] = yprim_diag / self.Ybase
        YtildeBlocks = [YtildeNMn, YtildeNMm, YtildeMNn, YtildeMNm]

        # Ybranch
        YbranchII[np.ix_(row, col)] = yprim_off / self.Ybase

        # Cmat
        Cmat[np.ix_(row, col)] = np.eye(int(n/2))

        return YtildeBlocks, YbranchII, Cmat
