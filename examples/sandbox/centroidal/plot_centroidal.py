import matplotlib.pylab as plt
import numpy as np

def plotf(us=None,models=None,ddp=None):
    if us is None and ddp is None: return
    if us is None: us = ddp.us
    if models is None: models = ddp.models()
    plt.subplot(2,1,1)
    contacts = [ m.differential.model0.contacts for m in models ]
    plt.plot([ u[contact['rfoot'].indexes] if 'rfoot' in contact else np.zeros(3)
               for u,contact in zip(us,contacts) ])
    plt.legend(['x','y','z'])
    plt.ylabel('rfoot')
    plt.subplot(2,1,2)
    plt.plot([ u[contact['lfoot'].indexes] if 'lfoot' in contact else np.zeros(3)
               for u,contact in zip(us,contacts) ])
    plt.legend(['x','y','z'])
    plt.ylabel('lfoot')

def plotx(xs):
    plt.subplot(3,1,1)
    plt.plot([ x[:3] for x in xs ])
    plt.legend(['x','y','z'])
    plt.ylabel('com')
    plt.subplot(3,1,2)
    plt.plot([ x[6:9] for x in xs ])
    plt.legend(['x','y','z'])
    plt.subplot(3,1,3)
    plt.ylabel('velocity')
    plt.plot([ x[9:12] for x in xs ])
    plt.legend(['x','y','z'])
    plt.ylabel('ang.mom')
