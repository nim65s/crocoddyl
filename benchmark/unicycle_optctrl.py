import sys
import time

import numpy as np

import crocoddyl

N = 200  # number of nodes
T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False


def createProblem(model):
    x0 = np.matrix([[1.0], [0.0], [0.0]])
    runningModels = [model()] * N
    terminalModel = model()
    xs = [x0] * (N + 1)
    us = [np.matrix([[0.0], [0.0]])] * N

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    ddp = crocoddyl.SolverDDP(problem)
    if CALLBACKS:
        ddp.setCallbacks([crocoddyl.CallbackVerbose()])
    duration = []
    for i in range(T):
        c_start = time.time()
        ddp.solve(xs, us, MAXITER)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calc(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcDiffBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


xs, us, problem = createProblem(crocoddyl.ActionModelUnicycle)
print("NQ:", problem.terminalModel.state.nq)
print("Number of nodes:", problem.T)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print("  DDP.solve [ms]: {:.4f} ({:.4f}-{:.4f})".format(avrg_dur, min_dur, max_dur))
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(
    "  ShootingProblem.calc [ms]: {:.4f} ({:.4f}-{:.4f})".format(
        avrg_dur, min_dur, max_dur
    )
)
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(
    "  ShootingProblem.calcDiff [ms]: {:.4f} ({:.4f}-{:.4f})".format(
        avrg_dur, min_dur, max_dur
    )
)
