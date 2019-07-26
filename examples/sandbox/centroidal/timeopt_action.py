import numpy as np


class IntegratedActionModelTimeOptEuler:
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True, timeScale = 1e-3):
        self.differential = diffModel
        self.State = self.differential.State
        self.nx = self.differential.nx
        self.ndx = self.differential.ndx
        self.nu = self.differential.nu + 1
        self.nq = self.differential.nq
        self.nv = self.differential.nv
        self.withCostResiduals = withCostResiduals
        self.weightTS = 10.
        self.timeStep = timeStep
        self.timeScale = timeScale

    @property
    def ncost(self):
        return self.differential.ncost

    def createData(self):
        return IntegratedActionDataTimeOptEuler(self)

    def calc(model, data, x, u=None):
        _u = None
        dt = 0.
        if u is not None:
            dt = u[-1]*model.timeScale
            _u = u[:-1]

        nq = model.nq
        data.acc, data.cost = model.differential.calc(data.differential, x, _u)
        if model.withCostResiduals:
            data.costResiduals[:] = data.differential.costResiduals[:]
        data.dx = np.concatenate([x[nq:] * dt + data.acc * dt**2, data.acc * dt])
        data.xnext[:] = model.differential.State.integrate(x, data.dx)

        if u is not None:
            data.cost += 0.5 * model.weightTS * (dt - model.timeStep)**2
            data.cost += -np.log(dt)/10
        
        return data.xnext, data.cost

    def calcDiff(model, data, x, u=None, recalc=True):
        _u = None
        dt = 0.
        if u is not None:
            dt = u[-1]*model.timeScale
            _u = u[:-1]
        nq, nv = model.nq, model.nv
        if recalc: model.calc(data, x, u)
        model.differential.calcDiff(data.differential, x, _u, recalc=False)
        dxnext_dx, dxnext_ddx = model.State.Jintegrate(x, data.dx)
        da_dx, da_du = data.differential.Fx, data.differential.Fu
        ddx_dx = np.vstack([da_dx * dt, da_dx])
        ddx_dx[range(nv), range(nv, 2 * nv)] += 1
        data.Fx[:, :] = dxnext_dx + dt * np.dot(dxnext_ddx, ddx_dx)
        ddx_du = np.vstack([da_du * dt, da_du])
        data.Fu[:, :-1] = dt * np.dot(dxnext_ddx, ddx_du)
        data.Lx[:] = data.differential.Lx
        data.Lu[:-1] = data.differential.Lu
        data.Lxx[:] = data.differential.Lxx
        data.Lxu[:, :-1] = data.differential.Lxu
        data.Luu[:-1, :-1] = data.differential.Luu

        if u is not None:
            acc, cost = data.acc,data.cost
            data.Fu[:, -1] = np.dot(dxnext_ddx, np.hstack([x[nq:] + 2. * acc * dt, acc]))*model.timeScale
            data.Lu[-1] = model.weightTS * (dt - model.timeStep)*model.timeScale
            data.Luu[-1, -1] = model.weightTS*model.timeScale**2

            data.Lxu[:, -1] = 0.
            
            data.Lu[-1] += model.timeScale**2/dt**2 /10
            data.Luu[-1,-1] -= model.timeScale/dt /10

class IntegratedActionDataTimeOptEuler:
    """ Implement the RK4 integration scheme and its derivatives.
    The effect on performance of the dense matrix multiplications in
    the calcDiff function needs to be taken into account when considering
    this integration scheme.
    """

    def __init__(self, model):
        nx, ndx, nu, ncost = model.nx, model.ndx, model.nu, model.ncost
        self.differential = model.differential.createData()
        self.xnext = np.zeros([nx])
        self.cost = np.nan

        # Dynamics data
        self.F = np.zeros([ndx, ndx + nu])
        self.Fx = self.F[:, :ndx]
        self.Fu = self.F[:, ndx:]

        # Cost data
        self.costResiduals = np.zeros([ncost])
        self.g = np.zeros([ndx + nu])
        self.L = np.zeros([ndx + nu, ndx + nu])
        self.R = np.zeros([ncost, ndx + nu])
        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        self.Lxx = self.L[:ndx, :ndx]
        self.Lxu = self.L[:ndx, ndx:]
        self.Luu = self.L[ndx:, ndx:]
        self.Rx = self.R[:, :ndx]
        self.Ru = self.R[:, ndx:]

