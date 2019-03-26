'''
These classes create an object logger with a callback operator callback(solver)
that the solver can call at every iteration end to store some data and display the robot motion
in gepetto-gui.
In the solver, set up the logger with solver.callback = [CallbackName()], and add the robot-wrapper
object in argument if you want to use the display functionalities.
'''

from __future__ import print_function

import copy
import os
import time

from diagnostic import displayTrajectory

try:
    from StringIO import StringIO
except ModuleNotFoundError:
    from io import StringIO


class CallbackDDPLogger:
    def __init__(self):
        self.steps = []
        self.iters = []
        self.costs = []
        self.control_regs = []
        self.state_regs = []
        self.th_stops = []
        self.gm_stops = []
        self.xs = []
        self.us = []

    def __call__(self, solver):
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.steps.append(solver.stepLength)
        self.iters.append(solver.iter)
        self.costs.append(copy.copy([d.cost for d in solver.datas()]))
        self.control_regs.append(solver.u_reg)
        self.state_regs.append(solver.x_reg)
        self.th_stops.append(solver.stop)
        self.gm_stops.append(-solver.expectedImprovement()[1])


class CallbackDDPVerbose:
    def __init__(self, level=0, filename=False):
        self.level = level
        self.filename = filename
        if filename and os.path.isfile(filename):
            os.remove(filename)

    def __call__(self, solver):
        if solver.iter % 10 == 0:
            line = "iter \t cost \t      stop \t    grad \t  xreg \t      ureg \t step \t feas"
            if self.level == 1:
                line += " \tdV-exp \t      dV",
            self.write(line)
        line = "%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e   %0.4f     %1d" % (
            solver.iter, sum(copy.copy([d.cost for d in solver.datas()])), solver.stop,
            -solver.expectedImprovement()[1], solver.x_reg, solver.u_reg, solver.stepLength, solver.isFeasible)
        if self.level == 1:
            line += "  %0.5e  %0.5e" % (solver.dV_exp, solver.dV),
        self.write(line)

    def write(self, line):
        print(line)
        if self.filename:
            with open(self.filename, 'a') as f:
                print(line, file=f)


class CallbackSolverDisplay:
    def __init__(self, robotwrapper, rate=-1, freq=1, cameraTF=None):
        self.robotwrapper = robotwrapper
        self.rate = rate
        self.cameraTF = cameraTF
        self.freq = freq

    def __call__(self, solver):
        if (solver.iter + 1) % self.freq:
            return
        dt = solver.models()[0].timeStep
        displayTrajectory(self.robotwrapper, solver.xs, dt, self.rate, self.cameraTF)


class CallbackSolverTimer:
    def __init__(self):
        self.timings = [time.time()]

    def __call__(self, solver):
        self.timings.append(time.time() - self.timings[-1])
