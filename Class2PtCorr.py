from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("Class2PtCorr")
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal




class Class2PtCorr():
    def __init__(self,NPix=11):
        
        self.Gamma=np.
