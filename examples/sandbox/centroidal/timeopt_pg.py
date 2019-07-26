from wpg import *


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
