from gviewserver import GepettoViewerServer
import pinocchio as pio
import numpy as np
from crocoddyl import a2m,m2a

class DisplayCentroidal:
    def __init__(self,gv=None,frames = [ 'rfoot', 'lfoot' ],initDisplay=True,mass=100.):
        from gviewserver import GepettoViewerServer
        self.gv = gv if gv is not None else GepettoViewerServer()
        self.frames = frames
        self.mass = mass
        if initDisplay: self.initDisplay()
        self.normalSigma = 30.   # Ref value of angular momentum sigma
                                # (more than 1. will not be displayed).
        
    def initFrameByName(self,fname,append=True):
        #gv.addXYZaxis('world/M'+fname,[1.,1.,1.,1.],.1,.2)
        self.gv.addBox('world/M'+fname,.25,.1,.02,[1.,.6,.0,1.])
        self.gv.addCylinder('world/f'+fname,.01,1.,[ 0.,0.,1.,1. ])
        if append: self.frames.append(fname)
            
    def initDisplay(self):
        gv = self.gv
        gv.addFloor('world/floor')
        gv.addSphere('world/com',.1,[1.,1.,1.,1.])
        gv.addCylinder('world/vcom',.01,1.,[ 1.,0.,0.,1. ])
        for fname in self.frames: self.initFrameByName(fname,append=False)

    def dispVector(self,name,vec,start,startSize = 0,scale=1.):
        '''Display a vector vec from a starting point start as a cylinder in gepetto viewer.

        :param name: <name> in gepetto-viewer, ex. world/vec.
        :param vec: vector to be displayed (np.array of size 3)
        :param start: starting point to be displayed (np.array of size 3)
        :param StartSize: optional argument in the case the starting point is materialized 
        by a sphere of size StartSize.
        :param scale: optional argument to change the scale of vec (same as passing vec*scale).
        '''
        from numpy.linalg import norm
        nvec = norm(vec)
        if nvec<1e-4: self.gv.setVisibility(name,'OFF')
        else:
            q = pio.Quaternion.FromTwoVectors(a2m(vec/nvec),np.matrix([0,0.,1.]).T).conjugate()
            p = start + vec*(.5*scale+startSize/nvec)
            self.gv.applyConfiguration(name,p.tolist()+m2a(q.coeffs()).tolist())
            self.gv.setFloatProperty(name,'Height',nvec*scale)
            self.gv.setVisibility(name,'ON')
        
    def __call__(self,c,cdot,sigma,contacts):
        gv = self.gv
        # Display COM
        gv.applyConfiguration('world/com',c.tolist()+[1.,0.,0.,0.])

        # Display AngMom as COM color
        intensity = np.clip(np.linalg.norm(sigma)/self.normalSigma,0.,1.)
        gv.setColor('world/com',[ 1., 1-intensity, 1-intensity, 1. ])

        # Display vcom
        self.dispVector('world/vcom',cdot,c,startSize=0.1)
        
        # Display contact placements
        act = []
        for fname,[placement,force] in contacts.items():
            if fname not in self.frames: self.initFrameByName(fname)
            # Display placement
            gv.setVisibility('world/M'+fname,'ON')
            R,p = placement.rotation,placement.translation
            gv.applyConfiguration('world/M'+fname,
                                  m2a(p).tolist()+\
                                  m2a(pio.Quaternion(R).coeffs()).tolist())
            # Display force
            self.dispVector('world/f'+fname,force,m2a(p),scale=1/(self.mass*20))
            # Set as active (prevent hiding).
            act.append(fname)
        for fname in self.frames:
            if fname not in act:
                gv.setVisibility('world/M'+fname,'OFF')
                gv.setVisibility('world/f'+fname,'OFF')
            
            
        gv.refresh()


if __name__ == "__main__":
    from pinocchio.utils import eye,rand,zero
    disp = DisplayCentroidal()
    x = np.array([ .1, .2, .6, 0.,0.,0., .05, 0.,0., .1,.1,.2 ])
    c = x[:3]; cdot = x[6:9]; sigma = x[9:12]
    f = np.array([0.,10.,800.])
    p = np.array([.1,.2,0])
    disp(c,cdot,sigma,{ 'rfoot': [pio.SE3(eye(3),a2m(p)), f ]})

