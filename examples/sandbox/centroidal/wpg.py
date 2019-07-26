import numpy as np
from pinocchio.utils import eye,rand,zero
from pinocchio import SE3
from crocoddyl import DifferentialActionModelNumDiff,ActionModelNumDiff,IntegratedActionModelEuler,SolverBoxDDP,SolverFDDP,CallbackDDPLogger,CallbackDDPVerbose,m2a,a2m,ShootingProblem
from centroidal import DifferentialActionModelCentroidal,CentroidalContactModel

from testutils import NUMDIFF_MODIFIER, assertNumDiff
from gviewserver import GepettoViewerServer
gv = GepettoViewerServer()

# --- LOCOMOTION PROBLEM ----------------------------------------------
# --- LOCOMOTION PROBLEM ----------------------------------------------
# --- LOCOMOTION PROBLEM ----------------------------------------------
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

# --- OCP -------------------------------------------------------------
# --- OCP -------------------------------------------------------------
# --- OCP -------------------------------------------------------------

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
assert( abs(sum([ph.duration for ph in phases])-DT*len(models))<DT/10 )

for t,im in enumerate(models):
    m = im.differential
    m.costs['com'].ref = x0[:3] + (xterm[:3]-x0[:3])*float(t)/len(models)
    m.costs['com']          .weights[:] = 0.1
    m.costs['vcom']         .weights[:] = 1
    m.costs['angmom']       .weights[:] = .2
    if 'frfoot' in m.costs: m.costs['frfoot'].weights[:] = .001
    if 'flfoot' in m.costs: m.costs['flfoot'].weights[:] = .001
    im.ul = np.array([ -10000.,-10000.,-10000. ]*m.ncontact)
    
from display_centroidal import DisplayCentroidal
disp1 = DisplayCentroidal()

ocp = ShootingProblem(x0,models[:-1],models[-1])
ocp.terminalModel.differential.costs['vcom'].weights[:] = 100.
ocp.terminalModel.differential.costs['angmom'].weights[:] = 1.

# --- SOLVER ----------------------------------------------------------
# --- SOLVER ----------------------------------------------------------
# --- SOLVER ----------------------------------------------------------

ddp = SolverBoxDDP(ocp)
ddp.setCandidate()
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]

ddp.solve(maxiter=5)

fddp = SolverFDDP(ocp)
fddp.solve(maxiter=1)

# --- DEBUG ---
# --- DEBUG ---
# --- DEBUG ---

np.set_printoptions(precision=3, linewidth=200, suppress=True)

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

