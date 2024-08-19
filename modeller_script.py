# Comparative modeling by the AutoModel class
from modeller import *              # Load standard Modeller classes
from modeller.automodel import *    # Load the AutoModel class
import sys

log.verbose()
env = Environ()
env.io.atom_files_directory = ['input/your/atom/files']
a = AutoModel(env,alnfile='{}'.format(sys.argv[1]),knowns='{}'.format(sys.argv[2]),sequence='{}'.format(sys.argv[3]))
a.starting_model=1
a.ending_model=1
a.make()
