import scipy.special
import numpy as np
import pylab
import scipy.stats
import GeneDist
import scipy.optimize
from ModIrregular import *
from DDFacet.Other import logger
log = logger.getLogger("ClassAndersonDarling")

erf=scipy.special.erf
G=scipy.special.gamma
logG=scipy.special.loggamma
Phi=lambda x: 0.5*(1+erf(x/np.sqrt(2)))
def Sigmoid(x,a=1):
    return 1./(1+np.exp(-x/a))

from scipy.special import gamma
from scipy.special import loggamma
def Chi2(x,k=1):
    A=1./( 2**(k/2.) * gamma(k/2.) )
    B=x**(k/2.-1)
    C=np.exp(-x/2.)
    return A*B*C

def logChi2(x,k=1):
    A=-( (k/2.)*np.log(2.) + loggamma(k/2.) )
    B=(k/2.-1)*np.log(x)
    C=(-x/2.)
    return A+B+C

def logChi2_parms(x,a,b,c,d):
    A=a
    B=b*np.log(x/np.abs(d))
    C=c*x
    return A+B+C

def logTwoSlopes(x,pars):
    a0,b0,a1,b1,s0,s1=pars
    y0=a0*x+b0
    y1=a1*x+b1
    S=Sigmoid(x-s0,a=s1)
    return y0*(1-S)+y1*S

    
def Fit_logPA2(x,y,GaussPars=(0,1),func=None):

    w=np.exp(y)
    ind=np.where(np.isnan(w) | np.isinf(w))[0]
    w[ind]=0
    w/=np.sum(w)
    #w.fill(0)
    med,sig=GaussPars
    x0=x[np.where(y==y.max())[0]][0]
    a0,b0=np.polyfit(x[x<x0], y[x<x0], 1)
    a1,b1=np.polyfit(x[x>x0], y[x>x0], 1)
    s0,s1=x0,sig/2.
    pars0=a0,b0,a1,b1,s0,s1

    def D(pars):
        return np.sum(w*(y-func(x,pars))**2)
        #return np.sum((y-func(x,pars))**2)
    Chi2a=D(pars0)/x.size
    pars=scipy.optimize.minimize(D,pars0)["x"]
    Chi2b=D(pars)/x.size
    return pars,Chi2a,Chi2b

    


class ClassAndersonDarlingMachine():
    def __init__(self,Scale="linear"):
        x=np.linspace(-5,5,1001)
        f_Phi=ClassF(x,Phi(x))
        f_Phi1=ClassF(x,1-Phi(x))
        r=f_Phi*f_Phi1
        ind=np.where(r.y<1e-10)
        r.y[ind]=1e-10
        #r.y=np.sqrt(r.y)
        #r.y[:]=1.
        r.update()

        #g=np.exp(-x**2/2.)
        #g/=np.sum(g)
        #r=ClassF(x,g)
        
        self.f_Phi=f_Phi
        self.r=r
        self.w=r.inv()
        self.logP=None
        # A2=np.linspace(0,5,10000)
        # self.logP=ClassF(A2,logTwoSlopes(A2,CoefsTwoSlope))
        # dx=0.01
        # self.dlogPdA2=self.logP.diff()

    def dlogPdA2(self,A2):
        if self.logP is None:
            log.print("Need a logP function")
            stop
        dx=0.01
        return (self.logP(A2+dx)-self.logP(A2))/dx#
            
    def generatePA2(self,n,NTry=100000):
        L_y=[]
        log.print("Number of generated %i-size samples: %i"%(n,NTry))
        for iTry in range(NTry):
            X=np.random.randn(n)
            L_y.append(self.giveA2(X))
        log.print("Compute cumulative distribution...")
        P=self.empirical_FA2=giveIrregularCumulDist(np.array(L_y),Type="Continuous")
        PT=P.T()
        x=np.linspace(PT(0.001),PT(0.95),100)
        P1=ClassF(x,P(x))
        
        self.empirical_PA2=P1.diff()
        m0,m1=PT(np.array([0.16,0.5]))
        med,sig=m1,m1-m0
        func=logTwoSlopes
        log.print("Fit cumulative distribution...")
        self.pars_fit_logPA2,Chi2a,Chi2b=Fit_logPA2(self.empirical_PA2.x,np.log(self.empirical_PA2.y),GaussPars=(med,sig),func=func)
        log.print("  reduced Chi-square of fit = ( %f -> %f )"%(Chi2a,Chi2b))
        self.logP=lambda x: func(x,self.pars_fit_logPA2)

    def dlogPdx(self,x):
        dA2dx=self.dA2_dx(x)
        A2=self.giveA2(x)
        return self.dlogPdA2(A2)*dA2dx
    
    def giveA2(self,X):
        n=X.size
        f_Phi=self.f_Phi
        F_X=giveIrregularCumulDist(X)
        #sw=np.sum(self.w(X))
        #A2=(n/sw)*n*((F_X-f_Phi)**2 * self.w).int()
        A2=n*((F_X-f_Phi)**2 * self.w).int()
        return A2

    # #########################
    # Jacob
    
    def dA2_dx(self,xi):

        nk=xi.size
        wk=1.
        Fn_k=giveIrregularCumulDist(xi)
        F_k=self.f_Phi
        dA2_dxi=np.zeros((nk,),np.float32)
        dA2_dxi1=np.zeros((nk,),np.float32)
        dDirac=0.001
        Diff=(Fn_k-F_k)
        
        dA2_dxi= 2*nk* (-1/nk) *self.w(xi)* (Diff(xi+1e-6)+Diff(xi-1e-6))/2.

        return dA2_dxi

    def meas_dA2_dx(self,X):
        LJ=[]
        for iParm in range(X.size):
            dx=np.linspace(-.0001,.0001,2)
            logL=[]
            Lx=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                logL.append(self.giveA2(Xix))
                Lx.append(Xix[iParm])
            J=(logL[1]-logL[0])/(dx[1]-dx[0])
            LJ.append(J)
        
        return np.array(LJ)

    # #########################
    # Hessian

    def d2A2_dx2(self,xi):

        nk=xi.size
        wk=1.
        Fn_k=giveIrregularCumulDist(xi)
        F_k=self.f_Phi
        dA2_dxi=np.zeros((nk,),np.float32)
        dA2_dxi1=np.zeros((nk,),np.float32)
        dDirac=0.001
        Diff=(Fn_k-F_k)
        
        dA2_dxi= 2*nk* (-1/nk) *self.w(xi)#* (Diff(xi+1e-6)+Diff(xi-1e-6))/2.

        return dA2_dxi
    
    def meas_d2A2_dx2(self,X):
        LH=[]
        for iParm in range(X.size):
            dx=np.linspace(-.001,.001,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.dA2_dx(Xix)[iParm])
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)
        
        return np.array(LH)
    
