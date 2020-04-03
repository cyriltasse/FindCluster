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
import scipy.signal

import scipy.stats
import pydtfe

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool

def test():
    CCM=ClassCovMatrix(NPix=11,SigmaPix=2.)
    CCM.buildFromCatalog()

def testDelaunayDensity():
    C=np.genfromtxt("/data/tasse/DataDeepFields/Millenium_z_0.5_0.63.txt",dtype=[("galaxyID",np.float32),("x",np.float32),("y",np.float32),("z",np.float32),("redshift",np.float32),("snapnum",np.float32),("stellarMass",np.float32)],delimiter=",")
    C=C.view(np.recarray)
    C=C[np.where(np.logical_not(np.isnan(C.x)))]
    
    X0,Y0=C.x, C.y
    X1,Y1=C.y, C.z
    X2,Y2=C.z, C.x
    X3,Y3=C.y, C.x 
    X4,Y4=C.z, C.y 
    X=np.concatenate([X0,X1,X2,X3,X4])
    Y=np.concatenate([Y0,Y1,Y2,Y3,Y4])

    

    
    Im=pydtfe.map_dtfe2d(X,Y,xsize=10000,ysize=10000)
    
    Dx=300
    Im=Im[Dx:-Dx,Dx:-Dx]
    Im/=np.mean(Im)
    
    pylab.clf()
    pylab.imshow(np.log10(Im))#,vmin=0,vmax=10000)
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)



def test2Pt():

    CC=ClassCovMatrix(NPix=31)
    CC.buildGaussianCov()
    n,m=CC.sqrtCs.shape
    NPix=CC.NPix
    np.random.seed(2)
    Gamma=np.dot(CC.sqrtCs,np.random.randn(m).reshape((-1,1))).reshape((NPix,NPix))+1
    Gamma-=np.min(Gamma)
    xg,yg=np.mgrid[0:NPix,0:NPix]
    
    
    Lx=[]
    Ly=[]
    for iPix in range(NPix):
        for jPix in range(NPix):
            n=Gamma[iPix,jPix]
            N=scipy.stats.poisson.rvs(n)
            if N>0:
                x=np.random.rand(N)-0.5+xg[iPix,jPix]
                y=np.random.rand(N)-0.5+yg[iPix,jPix]
                Lx+=x.tolist()
                Ly+=y.tolist()

    X=np.array(Lx)
    Y=np.array(Ly)
    
    def give_distMat(xx,yy,xx1=None,yy1=None):
        if xx1 is None:
            xx1=xx
            yy1=yy
        return np.sqrt( (xx.reshape((-1,1))-xx1.reshape((1,-1)))**2 + (yy.reshape((-1,1))-yy1.reshape((1,-1)))**2 )
    x0,x1=-0.5,NPix+0.5

    # X=np.random.rand(X.size)*(x1-x0)+x0
    # Y=np.random.rand(Y.size)*(x1-x0)+x

    Xr=np.random.rand(X.size)*(x1-x0)+x0
    Yr=np.random.rand(Y.size)*(x1-x0)+x0
    dDD=give_distMat(X,Y)
    dDR=give_distMat(X,Y,Xr,Yr)
    dRR=give_distMat(Xr,Yr)
    
    # pylab.clf()
    # pylab.imshow(Gamma.T)
    # pylab.scatter(X,Y)
    # pylab.draw()
    # pylab.show(False)
    # pylab.pause(0.1)
    
    rGrid=np.mgrid[0:int(NPix*np.sqrt(2))]
    rGrid_m=(rGrid[:-1]+rGrid[1:])/2.
    Psi=np.zeros((rGrid.size-1,),np.float32)
    Nd=X.size
    Nr=Xr.size
    for iR in range(rGrid.size-1):
        r0=rGrid[iR]
        r1=rGrid[iR+1]
        DD=np.where((dDD>=r0) & (dDD<r1))[0].size/(Nd*(Nd-1.)/2.)
        RR=np.where((dRR>=r0) & (dRR<r1))[0].size/(Nr*(Nr-1.)/2.)
        DR=np.where((dDR>=r0) & (dDR<r1))[0].size/(Nd*Nr)
        Psi[iR]=(DD+RR-2*DR)/RR
    
    pylab.clf()
    pylab.plot(rGrid_m,Psi)
    pylab.draw()
    pylab.show(False)
    pylab.pause(0.1)
        
class ClassCovMatrix():
    def __init__(self):
        pass
        # self.NPix=NPix
        # self.SigmaPix=SigmaPix
        # self.Type=Type
        
    def toSparse(self,C,Th=1e-2):
        Ca=np.abs(C)
        indi,indj=np.where(Ca>Th*Ca.max())
        data=C[indi,indj]
        M=C.shape[0]
        Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
        S=Cs.size/M**2
        return Cs,S

    def buildFromCatalog(self):
        C=np.genfromtxt("/data/tasse/DataDeepFields/Millenium_z_0.5_0.63.txt",dtype=[("galaxyID",np.float32),("x",np.float32),("y",np.float32),("z",np.float32),("redshift",np.float32),("snapnum",np.float32),("stellarMass",np.float32)],delimiter=",")
        C=C.view(np.recarray)
        h=0.7
        C.x*=h
        C.y*=h
        C.z*=h

        C=C[np.where(np.logical_not(np.isnan(C.x)))]
    
        X0,Y0=C.x, C.y
        X1,Y1=C.y, C.z
        X2,Y2=C.z, C.x

        D=C.x.max()-C.x.min()
        X3,Y3=np.mod(C.y+np.random.rand(1)[0]*D,D), np.mod(C.x +np.random.rand(1)[0]*D,D)
        X4,Y4=np.mod(C.z+np.random.rand(1)[0]*D,D), np.mod(C.y +np.random.rand(1)[0]*D,D)
        X=np.concatenate([X0,X1,X2,X3,X4])
        Y=np.concatenate([Y0,Y1,Y2,Y3,Y4])

    

        Nn=10000
        Im,dx,dy=pydtfe.map_dtfe2d(X,Y,xsize=Nn,ysize=Nn)

        Dx=300
        Im=Im[Dx:-Dx,Dx:-Dx]
        Gamma=Im/np.mean(Im)
        
        pylab.clf()
        pylab.imshow(np.log10(Im))#,vmin=0,vmax=10000)
        pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        
        Np=381
        Fact=3
        R=np.arange(0,5,dx*Fact)
        Rn=np.zeros((R.size,),np.float32)
        xg,yg=np.mgrid[0:Nn,0:Nn]
        xg=xg*dx
        yg=yg*dy
        G=Gamma-1.
        Cov=np.zeros((R.size,),np.float32)

        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        Dx=int(R.max()/dx)
        Dy=int(R.max()/dy)
        for iP in range(Np):
            print("%i/%i [%i]"%(iP,Np,R.size))
            i,j=int(np.random.rand(1)[0]*Im.shape[0]),int(np.random.rand(1)[0]*Im.shape[1])
            i0=np.max([0,i-Dx])
            i1=np.min([Nn,i+Dx])
            j0=np.max([0,j-Dy])
            j1=np.min([Nn,j+Dy])
            xgs=xg[i0:i1,j0:j1]
            ygs=yg[i0:i1,j0:j1]
            d=np.sqrt((i*dx-xgs)**2+(j*dy-ygs)**2)
            Gs=G[i0:i1,j0:j1]
            T.timeit("Compute d")
            for iR in range(R.size):
                # print(iP,iR)
                rm=R[iR]
                r0=rm-dx/2.*Fact
                r1=rm+dx/2.*Fact
                indx,indy=np.where((d>=r0) & (d<r1))
                Nsx,Nsy=Gs.shape
                Cx=((indx>=0)&(indx<Nsx))
                Cy=((indy>=0)&(indy<Nsy))
                Cxy=(Cx&Cy)
                indx=indx[Cxy]
                indy=indy[Cxy]
                T.timeit("  Where ")
                Cov[iR]+=np.sum(G[i,j]*Gs[indx,indy])
                Rn[iR]+=indx.size
                T.timeit("  Rest ")

        Cov/=Rn

        np.savez("CovMillenium",R=R,Cov=Cov)
        pylab.clf()
        pylab.plot(R,Cov)
        pylab.draw()
        pylab.show(False)
        stop
                
                                   
    def giveCovMat(self,
                   CellSizeRad=None,
                   NPix=None,
                   zm=None,
                   SigmaPix=None):
        N=NPix
        x,y=np.mgrid[0:N,0:N]
        dx=x.reshape((-1,1))-x.reshape((1,-1))
        dy=y.reshape((-1,1))-y.reshape((1,-1))
        d=dx**2+dy**2
        Sig=SigmaPix

        C=np.exp(-(d)/(2.*Sig**2))#*100
        return C
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
        sss=ss[ss>0]
        log.print("  log Singular value Max/Min: %5.2f"%(np.log10(sss.max()/sss.min())))
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
    
