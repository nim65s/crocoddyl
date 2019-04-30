import pickle
import sys
from os.path import dirname, join

from crocoddyl import loadTalos

# ---------------------------------------------------------
robot = loadTalos()
robot.model.armature[6:] = 1.
# ---------------------------------------------------------

DISPLAY = 'disp' in sys.argv
RUNTIME_DISPLAY = False
DT = 0.01

# -------------------------INITIAL TRAJECTORY--------------
TRAJ_DIR = join(dirname(__file__), "traj_1step")

X_init = pickle.load(open(join(TRAJ_DIR, "X_init.muscod_out"), "rb"))  # loads the state x
U_init = pickle.load(open(join(TRAJ_DIR, "U_init.muscod_out"), "rb"))  # loads the control u

# loads the forces ("lambda") in the local frame in order [LF, RF, LH, RH]
# where lf: 'leg_left_6_joint', 'rf' : 'leg_right_6_joint'
# Note: Hands are optional. Hands are not loaded for biped walking
f_init = pickle.load(open(join(TRAJ_DIR, "f_init.muscod_out"), "rb"))
ddq_init = pickle.load(open(join(TRAJ_DIR, "ddq_init.muscod_out"), "rb"))  # loads the acceleration

# -----------------------Contact Sequence-----------------

MUSCOD_CS_OUTPUT_FILENAME = join(TRAJ_DIR, "contact_sequence_trajectory.xml")
CONTACT_SEQUENCE_XML_TAG = "contact_sequence"

# Define the contact frames which are active in the robot.
# allowed keys are: (LF_patch,RF_patch, LH_patch, RH_patch)
contact_patches = {"LF_patch": "leg_left_6_joint", "RF_patch": "leg_right_6_joint"}
