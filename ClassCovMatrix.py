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
log = logger.getLogger("ClassCovMatrix")
import scipy.sparse
import scipy.sparse.linalg

def test():
    CCM=ClassCovMatrix(NPix=11,SigmaPix=2.)
    CCM.buildGaussianCov()
    stop
    
class ClassCovMatrix():
    def __init__(self,NPix=11,SigmaPix=2.,Type="Normal"):
        self.NPix=NPix
        self.SigmaPix=SigmaPix
        self.Type=Type
        
    def toSparse(self,C,Th=1e-2):
        Ca=np.abs(C)
        indi,indj=np.where(Ca>Th*Ca.max())
        data=C[indi,indj]
        M=C.shape[0]
        Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
        S=Cs.size/M**2
        return Cs,S

    def buildGaussianCov(self):
        N=self.NPix
        x,y=np.mgrid[0:N,0:N]
        dx=x.reshape((-1,1))-x.reshape((1,-1))
        dy=y.reshape((-1,1))-y.reshape((1,-1))
        d=dx**2+dy**2
        Sig=self.SigmaPix

        C=np.exp(-(d)/(2.*Sig**2))
        Cs,Sparsity=self.toSparse(C)
        M=C.shape[0]
        log.print("  Non-zeros are %.1f%% of the matrix size [%ix%i]"%(Sparsity*100,M,M))
        self.Cs=Cs
        CellSize=1.
        A=(CellSize*N)**2
        k=np.max([A/Sig**2,1.])
        k=int(k*3)
        k=int(np.min([M-1,k]))
        log.print("Choosing k=%i [M=%i]"%(k,M))
        self.k=k
        Us,ss,Vs=scipy.sparse.linalg.svds(Cs,k=k)
        log.print("  log Singular value Max/Min: %5.2f"%(np.log10(ss.max()/ss.min())))
        ssqs=np.sqrt(ss)
        sqrtCs =Us*ssqs.reshape(1,ssqs.size)
        self.sqrtCs=sqrtCs
        
        # U,s,V=np.linalg.svd(C)
    
        # U,s,V=scipy.sparse.linalg.svds(Cs,k=k,return_singular_vectors="u")
        # T.timeit("SVDs")
        
        # u,s,v=np.linalg.svd(C)
        # T.timeit("SVD")
        
        # u,s,v=np.linalg.svd(C,hermitian=True)
        # T.timeit("SVDh")
        
        # pylab.clf()
        # pylab.plot(s)
        # pylab.draw()
        # pylab.show(block=False)
        
        #s[s<0.]=0.
        #ssq=np.sqrt(np.abs(s))
        # ssq=np.sqrt(s)
        

    #     Nr=10000
    #     X=np.random.randn(sqrtCs.shape[1],Nr)
    #     Y=sqrtCs.dot(X)
    #     Cm=np.dot(Y,Y.T)/Y.shape[1]
        
    #     v0,v1=C.min(),C.max()
    #     pylab.clf()
    #     ax=pylab.subplot(1,3,1)
    #     pylab.imshow(C,interpolation="nearest",vmin=v0,vmax=v1)
    #     pylab.subplot(1,3,2,sharex=ax,sharey=ax)
    #     pylab.imshow(Cm,interpolation="nearest",vmin=v0,vmax=v1)
    #     pylab.subplot(1,3,3,sharex=ax,sharey=ax)
    #     pylab.imshow(C-Cm,interpolation="nearest")
    #     pylab.draw()
    #     pylab.show(block=False)
    #     return

    # # pylab.clf()
    # # pylab.imshow(sqrtCs,interpolation="nearest")
    # # pylab.draw()
    # # pylab.show(False)
    
    # sqrtCss=toSparse(sqrtCs)
    # sC=sqrtCss.toarray()
    # T.timeit(0)

    # for i in range(10):
    #     v=np.random.randn(M).reshape((-1,1))
    #     Im=sqrtCss.dot(v).reshape((N,N))


    #     pylab.clf()
    #     pylab.imshow(Im,interpolation="nearest")
    #     pylab.draw()
    #     pylab.show(False)
    #     pylab.pause(0.5)
    # T.timeit("s")
    
    # # np.dot(sC,v)
    # # T.timeit("nos")
    

    # return sqrtCss
    
