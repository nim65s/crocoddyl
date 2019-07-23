import crocoddyl
from crocoddyl import DifferentialActionModelAbstract, DifferentialActionDataAbstract, StateVector, m2a, a2m
from pinocchio import SE3
import numpy as np
from collections import OrderedDict

class DifferentialActionModelCentroidal(DifferentialActionModelAbstract):
    """ Centroidal model for locomotion.

    State: com c, momentum integral r, com velocity cdot, angular momentum sigma
    Control:  force f_i  for i contact at placement o^M_i = [ o^R_i o^OI ]
    
    cddot = 1/m sum o^f_i = 1/m sum oRi i^f_i
    sigmadot  = sum (-o^OC + o^OI) x o^f_i  = sum (p_i-c)x (oRi ^if_i)

    NB: cXi  = [ oRi 0 ; o^CIx oRi   oRi ]
    
    """
    def __init__(self, contacts, mass, comRef = np.zeros(3), vcomRef = np.zeros(3)):
        assert( isinstance(contacts,OrderedDict) )
        assert( len(contacts)==0 or isinstance(contacts.values()[0],SE3) )
      
        self.ncontact = len(contacts)
        DifferentialActionModelAbstract.__init__(self, nq=6, nv=6, nu=3*self.ncontact)
        self.DifferentialActionDataType = DifferentialActionDataCentroidal
        self.State = StateVector(self.nx)

        self.contacts = contacts
        self.mass = mass

        # --- COSTS ---
        class CentroidalCost:
            def __init__(self,idx,ncost,ref=None):
                self.idx = idx
                self.ncost = ncost
                self.idxs = range(idx,idx+ncost)
                self.ref = ref if ref is not None else np.zeros(ncost)
            def setWeights(self,weights):
                self.weights = weights[self.idxs]
                return self.weights
        
        self.ncost = 0
        self.costs = {}
    
        self.costs['com'] = CentroidalCost(self.ncost,3,comRef); self.ncost+=3
        self.costs['vcom'] = CentroidalCost(self.ncost,3,vcomRef); self.ncost+=3
        for c in self.contacts:
            self.costs['f%s'%c] = CentroidalCost(self.ncontact,3); self.ncost+=3
        
        self.weights = np.zeros(self.ncost)
        self.costs['com'].setWeights(self.weights)[:] = 1.
        self.costs['vcom'].setWeights(self.weights)[:] = .1
        for c in self.contacts:
            self.costs['f%s'%c].setWeights(self.weights)[:] = .01
        
      
    def calc(model,data,x,u=None):
        if u is None: u = np.zeros(0)
        c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
        fs = [ np.dot(m2a(M.rotation),f) for f,M in zip(np.split(u,[3]*model.ncontact),
                                                        model.contacts.values()) ]

        # --- DYN ---
        cddot    = 1/model.mass * sum(fs)
        sigmadot = sum([ np.cross(m2a(M.translation)-c, f) for f,M in zip(fs,model.contacts.values()) ])
        
        data.xout[3:] = cddot
        data.xout[:3] = sigmadot

        # --- COST ---
        c='com'; cm,cd = model.costs[c],data.costs[c]
        cd.r[:] = c-cm.ref

        c='vcom'; cm,cd = model.costs[c],data.costs[c]
        cd.r[:] = cdot-cm.ref

        for contact in contacts:
            c='f%s'%contact; cm,cd = model.costs[c],data.costs[c]
            cd.r[:] = cdot-cm.ref

        
        
        data.cost = .5*sum(data.costResiduals**2)
        
        return data.xout, data.cost
  
    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
      

class DifferentialActionDataCentroidal(DifferentialActionDataAbstract):
    def __init__(self,model):
        DifferentialActionDataAbstract.__init__(self, model)

        # --- COSTS ---
        class CentroidalCost:
            def __init__(self,idxs,r,Rx,Ru):
                self.idxs = idxs
                self.r = r[idxs]
                self.Rx = Rx[idxs,:]
                self.Ru = Ru[idxs,:]

        self.costs = {}
        for n,c in model.costs.items():
            self.costs[n] = CentroidalCost(c.idxs,self.costResiduals,self.Rx,self.Ru)

            

#########################################################################33
#########################################################################33
#########################################################################33

from pinocchio.utils import eye,rand,zero
dmodel = DifferentialActionModelCentroidal(mass=1., contacts={ 'lleg': SE3(eye(3),rand(3)) })
ddata  = dmodel.createData()

x = dmodel.State.rand()
u = np.zeros(3)

dmodel.calc(ddata,x,u)

