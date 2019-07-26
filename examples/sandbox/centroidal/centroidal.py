import crocoddyl
from crocoddyl import DifferentialActionModelAbstract, DifferentialActionDataAbstract, StateVector, m2a, a2m, IntegratedActionModelEuler, CallbackDDPLogger, CallbackDDPVerbose, SolverBoxDDP
import pinocchio as pio
import numpy as np
from collections import OrderedDict


# --- CONTACT ---------------------------------------------------------------
# --- CONTACT ---------------------------------------------------------------
# --- CONTACT ---------------------------------------------------------------
class CentroidalContactModel:
    def __init__(self,name,indexes,placement):
        self.name = name
        self.indexes = indexes
        self.placement = placement
    def createData(self):
        return CentroidalContactData(self)
    
class CentroidalContactData:
    def __init__(self,model):
        pass

    
# --- COST ------------------------------------------------------------------
# --- COST ------------------------------------------------------------------
# --- COST ------------------------------------------------------------------

class CentroidalCostModel:
    def __init__(self,idx,ncost,ref=None):
        self.idx = idx
        self.ncost = ncost
        self.ref = ref if ref is not None else np.zeros(ncost)
    def setWeights(self,weights):
        self.weights = weights[self.idx:self.idx+self.ncost]
        return self.weights

class CentroidalCostData:
    def __init__(self,idx,ncost,r,Rx,Ru):
        self.idx = idx
        self.ncost = ncost
        self.r = r[idx:idx+ncost]
        self.Rx = Rx[idx:idx+ncost,:]
        self.Ru = Ru[idx:idx+ncost,:]

# --- DAM -------------------------------------------------------------------
# --- DAM -------------------------------------------------------------------
# --- DAM -------------------------------------------------------------------
class DifferentialActionModelCentroidal(DifferentialActionModelAbstract):
    """ Centroidal model for locomotion.

    State: com c, momentum integral r, com velocity cdot, angular momentum sigma
    Control:  force f_i  for i contact at placement o^M_i = [ o^R_i o^OI ]
    
    cddot = 1/m sum o^f_i = 1/m sum oRi i^f_i
    sigmadot  = sum (-o^OC + o^OI) x o^f_i  = sum (p_i-c)x (oRi ^if_i)

    NB: cXi  = [ oRi 0 ; o^CIx oRi   oRi ]
    
    """
    def __init__(self, contacts, mass, comRef = np.zeros(3), vcomRef = np.zeros(3)):
      
        self.ncontact = len(contacts)
        DifferentialActionModelAbstract.__init__(self, nq=6, nv=6, nu=3*self.ncontact)
        self.DifferentialActionDataType = DifferentialActionDataCentroidal
        self.State = StateVector(self.nx)

        self.contacts = contacts
        self.mass = mass
        self.grav = np.array([0,0,-9.81])
        self.withCostResiduals = False
        
        # --- COSTS ---
         
        self.ncost = 0
        self.costs = {}
    
        self.costs['com'] = CentroidalCostModel(self.ncost,3,comRef); self.ncost+=3
        self.costs['vcom'] = CentroidalCostModel(self.ncost,3,vcomRef); self.ncost+=3
        self.costs['angmom'] = CentroidalCostModel(self.ncost,3,vcomRef); self.ncost+=3
        for c in self.contacts:
            self.costs['f%s'%c] = CentroidalCostModel(self.ncost,3); self.ncost+=3
        
        self.weights = np.zeros(self.ncost)
        self.costs['com'].setWeights(self.weights)[:] = 1.
        self.costs['vcom'].setWeights(self.weights)[:] = 1.
        self.costs['angmom'].setWeights(self.weights)[:] = 1.
        for c,contact in self.contacts.items():
            self.costs['f%s'%c].setWeights(self.weights)[:] = 1.
            self.costs['f%s'%c].forceIndexes = contact.indexes

    def disp(self,x,u=None,delay=0.1):
        from display_centroidal import DisplayCentroidal
        import time
        if 'displayCentroidal' not in self.__dict__: self.displayCentroidal = DisplayCentroidal()
        
        if u is None: u = np.zeros(self.ncontact*3)
        c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
        contacts = { contact.name:[ contact.placement,np.dot(m2a(contact.placement.rotation),f) ]
                     for f,contact in zip(np.split(u,self.ncontact),
                                          self.contacts.values()) }
        self.displayCentroidal(c,cdot,sigma,contacts)
        time.sleep(delay)
        
    def calc(model,data,x,u=None):
        if u is None: u = np.zeros(model.ncontact*3)
        c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
        fs = [ np.dot(m2a(contact.placement.rotation),f)
               for f,contact in zip(np.split(u,model.ncontact),
                                    model.contacts.values()) ]

        # --- DYN ---
        cddot    = 1/model.mass * sum(fs) + model.grav
        sigmadot = sum([ np.cross(m2a(contact.placement.translation)-c, f)
                         for f,contact in zip(fs,model.contacts.values()) ])
        
        data.xout[:3] = cddot
        data.xout[3:] = sigmadot

        # --- COST ---
        cname='com'; cm,cd = model.costs[cname],data.costs[cname]
        cd.r[:] = cm.weights*(c-cm.ref)
        
        cname='vcom'; cm,cd = model.costs[cname],data.costs[cname]
        cd.r[:] = cm.weights*(cdot-cm.ref)

        cname='angmom'; cm,cd = model.costs[cname],data.costs[cname]
        cd.r[:] = cm.weights*(sigma)

        for contact,f in zip(model.contacts,fs):
            cname='f%s'%contact; cm,cd = model.costs[cname],data.costs[cname]
            cd.r[:] = cm.weights*(f-cm.ref)
        
        data.cost = .5*sum(data.costResiduals**2)
        
        return data.xout, data.cost
  
    def calcDiff(model,data,x,u=None,recalc=True):
        if recalc: xout,cost = model.calc(data,x,u)
        if u is None: u = np.zeros(model.ncontact*3)

        c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
        fs = [ np.dot(m2a(contact.placement.rotation),f)
               for f,contact in zip(np.split(u,model.ncontact),
                                    model.contacts.values()) ]

        # --- DYN ---
        data.Fx[3:,:3] = pio.utils.skew(sum(fs))
        for contact in model.contacts.values():
            np.fill_diagonal(data.Fu[:3,contact.indexes[0]:contact.indexes[-1]+1],1/model.mass)
            lever = m2a(contact.placement.translation)-c
            data.Fu[3:,contact.indexes[0]:contact.indexes[-1]+1] = pio.utils.skew(lever)
            
        # --- COST ---
        cname= 'com'; cm,cd = model.costs[cname],data.costs[cname]
        if model.withCostResiduals: np.fill_diagonal(cd.Rx[:,:3],cm.weights)
        data.Lx[:3] = cm.weights*cd.r
        np.fill_diagonal(data.Lxx[:3,:3],cm.weights**2)

        cname= 'vcom'; cm,cd = model.costs[cname],data.costs[cname]
        if model.withCostResiduals: np.fill_diagonal(cd.Rx[:,6:9],cm.weights)
        data.Lx[6:9] = cm.weights*cd.r
        np.fill_diagonal(data.Lxx[6:9,6:9],cm.weights**2)
        
        cname= 'angmom'; cm,cd = model.costs[cname],data.costs[cname]
        if model.withCostResiduals: np.fill_diagonal(cd.Rx[:,9:12],cm.weights)
        data.Lx[9:12] = cm.weights*cd.r
        np.fill_diagonal(data.Lxx[9:12,9:12],cm.weights**2)
        
        for contact,f in zip(model.contacts,fs):
            cname='f%s'%contact; cm,cd = model.costs[cname],data.costs[cname]
            if model.withCostResiduals:
                np.fill_diagonal(cd.Ru[:,cm.forceIndexes[0]:cm.forceIndexes[-1]+1],cm.weights)
            data.Lu[cm.forceIndexes[0]:cm.forceIndexes[-1]+1] = cm.weights*cd.r
            np.fill_diagonal(data.Luu[cm.forceIndexes[0]:cm.forceIndexes[-1]+1,
                                      cm.forceIndexes[0]:cm.forceIndexes[-1]+1],cm.weights**2)

class DifferentialActionDataCentroidal(DifferentialActionDataAbstract):
    def __init__(self,model):
        DifferentialActionDataAbstract.__init__(self, model)

        self.costs = {}
        for n,c in model.costs.items():
            self.costs[n] = CentroidalCostData(c.idx,c.ncost,self.costResiduals,self.Rx,self.Ru)
            

#########################################################################33
#########################################################################33
#########################################################################33

