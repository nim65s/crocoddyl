import crocoddyl
from crocoddyl import DifferentialActionModelAbstract, DifferentialActionDataAbstract, StateVector, m2a, a2m, IntegratedActionModelEuler, CallbackDDPLogger, CallbackDDPVerbose, SolverBoxDDP
from pinocchio import SE3
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
        if 'displayCentroidal' not in self.__dict__: self.displayCentroidal = DisplayCentroidal()
        
        if u is None: u = np.zeros(self.ncontact*3)
        c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
        contacts = { contact.name:[ contact.placement,np.dot(m2a(contact.placement.rotation),f) ]
                     for f,contact in zip(np.split(u,self.ncontact),
                                          self.contacts.values()) }
        self.displayCentroidal(c,cdot,sigma,contacts)
        import time
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
        # data.Lx[:] = np.dot(data.Rx.T,data.costResiduals)
        # data.Lu[:] = np.dot(data.Ru.T,data.costResiduals)
        # data.Lxx[:] = np.dot(data.Rx.T,data.Rx)
        # data.Luu[:] = np.dot(data.Ru.T,data.Ru)
        

class DifferentialActionDataCentroidal(DifferentialActionDataAbstract):
    def __init__(self,model):
        DifferentialActionDataAbstract.__init__(self, model)

        self.costs = {}
        for n,c in model.costs.items():
            self.costs[n] = CentroidalCostData(c.idx,c.ncost,self.costResiduals,self.Rx,self.Ru)
            

#########################################################################33
#########################################################################33
#########################################################################33

from pinocchio.utils import eye,rand,zero
from crocoddyl import DifferentialActionModelNumDiff,ActionModelNumDiff
from testutils import NUMDIFF_MODIFIER, assertNumDiff
from gviewserver import GepettoViewerServer
gv = GepettoViewerServer()

contacts = { 'lfoot': CentroidalContactModel('lfoot', range(3), SE3(eye(3), zero(3)) ) }


dmodel = DifferentialActionModelCentroidal(mass=1., contacts=contacts)
ddata  = dmodel.createData()

x = dmodel.State.rand()
u = np.zeros(3)

dmodel.calc(ddata,x,u)

damnd = DifferentialActionModelNumDiff(dmodel,withGaussApprox=True)
dadnd = damnd.createData()

x = dmodel.State.rand()
u = np.random.rand(3)

dmodel.costs['com'].weights[:] = np.random.rand(3)
dmodel.costs['vcom'].weights[:] = np.random.rand(3)
dmodel.costs['angmom'].weights[:] = np.random.rand(3)
dmodel.costs['flfoot'].weights[:] = np.random.rand(3)
dmodel.withCostResiduals = True
dmodel.calcDiff(ddata,x,u)
damnd .calcDiff(dadnd,x,u)
assertNumDiff(ddata.Fx,dadnd.Fx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Fu,dadnd.Fu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Rx,dadnd.Rx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Ru,dadnd.Ru,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lx,dadnd.Lx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lu,dadnd.Lu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lxx,dadnd.Lxx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Luu,dadnd.Luu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lxu,dadnd.Lxu,NUMDIFF_MODIFIER * damnd.disturbance)

# --- INTEGRATION
model = IntegratedActionModelEuler(dmodel,timeStep=1e-2,withCostResiduals=True)
data  = model.createData()

x = dmodel.State.zero()
u = np.zeros(3)

xnext,cost = model.calc(data,x,u)

assertNumDiff(xnext[:3],  model.timeStep**2 * dmodel.grav,1e-6) 
assertNumDiff(xnext[6:9], model.timeStep    * dmodel.grav,1e-6) 

modelnd = ActionModelNumDiff(model,withGaussApprox=True)
datand = modelnd.createData()

x = dmodel.State.rand()
u = np.random.rand(3)

model   .calcDiff(data,  x,u)
modelnd .calcDiff(datand,x,u)

assertNumDiff(data.Fx,datand.Fx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(data.Fu,datand.Fu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(data.Lx,datand.Lx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(data.Lu,datand.Lu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lxx,dadnd.Lxx,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Luu,dadnd.Luu,NUMDIFF_MODIFIER * damnd.disturbance)
assertNumDiff(ddata.Lxu,dadnd.Lxu,NUMDIFF_MODIFIER * damnd.disturbance)

# --- OCP
class Phase:
    def __init__(self, contacts, duration):
        self.contacts = contacts
        self.duration = duration
footGap = .15
def right(x): return SE3(eye(3),np.matrix([x, footGap/2,0.]).T)
def left (x): return SE3(eye(3),np.matrix([x,-footGap/2,0.]).T)

phases = [
    Phase( { 'rfoot': right(0), 'lfoot': left(0.0) }, 1.5 ),
    Phase( { 'rfoot': right(0)                     }, 0.8 ),
    Phase( { 'rfoot': right(0), 'lfoot': left(0.2) }, 1.5 ),
]

if 0:
    phases = [
        Phase( { 'rfoot': right(0.0), 'lfoot': left(0.0) }, 1.5 ),
        Phase( { 'rfoot': right(0.0)                     }, 0.8 ),
        Phase( { 'rfoot': right(0.0), 'lfoot': left(0.2) }, 0.2 ),
        Phase( { 'rfoot': left (0.2)                     }, 0.8 ),
        Phase( { 'rfoot': right(0.4), 'lfoot': left(0.2) }, 0.2 ),
        Phase( { 'rfoot': right(0.4)                     }, 0.8 ),
        Phase( { 'rfoot': right(0.4), 'lfoot': left(0.6) }, 0.2 ),
        Phase( { 'rfoot': left (0.6)                     }, 0.8 ),
        Phase( { 'rfoot': right(0.8), 'lfoot': left(0.6) }, 1.5 ),
    ]

x0 =    np.array([ 0., 0.,1.,    0.,0.,0.,    0.,0.,0.,    0.,0.,0. ])
xterm = np.array([ 0.1,0.,1.,    0.,0.,0.,    0.,0.,0.,    0.,0.,0. ])

models = []
DT = 1e-1
MASS = 50.

from copy import copy

for phase in phases:
    contacts = {}
    nc = 0
    for name,placement in phase.contacts.items():
        contacts.update({ name: CentroidalContactModel(name,placement=placement,
                                                       indexes=range(nc,nc+3)) })
        nc += 3

    dam   = lambda : DifferentialActionModelCentroidal(mass=MASS,contacts=contacts)
    #damnd = lambda d: DifferentialActionModelNumDiff(d,withGaussApprox=True)
    nshoot = int(round(phase.duration/DT))
    assert( abs(nshoot*DT-phase.duration)<DT/10 )
    models += [
        IntegratedActionModelEuler(dam(),DT) for _ in range(nshoot) ]
    #print models
assert( abs(sum([ph.duration for ph in phases])-DT*len(models))<DT/10 )

for t,im in enumerate(models):
    m = im.differential
    m.costs['com'].ref = x0[:3] + (xterm[:3]-x0[:3])*float(t)/len(models)
    #print t,x0[:3] + (xterm[:3]-x0[:3])*float(t)/(len(models)-1.)
    m.costs['com']          .weights[:] = 0.1
    m.costs['vcom']         .weights[:] = 1
    m.costs['angmom']       .weights[:] = .2
    if 'frfoot' in m.costs: m.costs['frfoot'].weights[:] = .001
    if 'flfoot' in m.costs: m.costs['flfoot'].weights[:] = .001
    im.ul = np.array([ -10000.,-10000.,-10000. ]*m.ncontact)
    
from display_centroidal import DisplayCentroidal
disp1 = DisplayCentroidal()

from crocoddyl import ShootingProblem,SolverFDDP
ocp = ShootingProblem(x0,models[:-1],models[-1])
ocp.terminalModel.differential.costs['vcom'].weights[:] = 100.
ocp.terminalModel.differential.costs['angmom'].weights[:] = 1.

from crocoddyl.qpsolvers import quadprogWrapper
ddp = SolverBoxDDP(ocp)
ddp.setCandidate()
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]

ddp.solve(maxiter=5)

fddp = SolverFDDP(ocp)
fddp.solve(maxiter=1)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
#stophere

m=models[10].differential
d=ddp.datas() [10].differential
u=ddp.us[10]
x=ddp.xs[10]


def disp():
    for m,d,x,u in zip(ddp.models(),ddp.datas(),ddp.xs,ddp.us):
        m.differential.disp(x,u)


import matplotlib.pylab as plt
from plot_centroidal import plotf,plotx
plt.ion()

xs = ddp.xs
us = ddp.us
ms = [ m.differential for m in ddp.models() ]
ds = [ d.differential for d in ddp.datas() ]

lf = m2a(phases[-1].contacts['lfoot'].translation)
c  = xs[22][:3]

m=ddp.models()[0]
d=ddp.datas()[0]
x=ddp.xs[0]
u=ddp.us[0]
m.differential.calcDiff(d.differential,x,u)

stophere
oddp = ddp
# --------------- time opt
from timeopt_action import IntegratedActionModelTimeOptEuler

models = [ IntegratedActionModelTimeOptEuler(m.differential,timeStep=m.timeStep) for m in models ]
for m in models:
    m.ul = np.array([ -100000. ]*3*m.differential.ncontact + [5e-3/m.timeScale])
    m.uu = np.array([  100000. ]*3*m.differential.ncontact + [DT*5/m.timeScale])
    m.weightTS = 1.
    
ocp = ShootingProblem(x0,models[:-1],models[-1])

model = models[0]
data  = model.createData()
x = x0.copy()
u = np.concatenate([np.random.rand(6)*100,np.array([1e-2])])

ddp = SolverBoxDDP(ocp)
ddp = SolverFDDP(ocp)
ddp.setCandidate()
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]

def disp():
    for m,d,x,u in zip(ddp.models(),ddp.datas(),ddp.xs,ddp.us):
        m.differential.disp(x,u[:-1])

ddp.solve(init_xs=oddp.xs,
          init_us= [ np.concatenate([u,np.array([DT/m.timeScale])]) for u,m in zip(oddp.us,ddp.models()) ],
          isFeasible=True,maxiter=1000)

