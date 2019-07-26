import numpy as np
from pinocchio.utils import eye,rand,zero
from pinocchio import SE3
from crocoddyl import DifferentialActionModelNumDiff,ActionModelNumDiff,IntegratedActionModelEuler
from centroidal import DifferentialActionModelCentroidal,CentroidalContactModel
from testutils import NUMDIFF_MODIFIER, assertNumDiff

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
