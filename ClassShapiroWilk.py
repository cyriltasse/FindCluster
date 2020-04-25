import scipy.special
import numpy as np
import pylab
import scipy.stats
import GeneDist
import scipy.optimize
from ModIrregular import *
from DDFacet.Other import logger
log = logger.getLogger("ClassAndersonDarling")
from killMS.Array import ModLinAlg
erf=scipy.special.erf
G=scipy.special.gamma
logG=scipy.special.loggamma
#Phi=lambda x: 0.5*(1+erf(x/np.sqrt(2)))
def Phi(x,mu=0.,sig=1.):
    return 0.5*(1+erf((x-mu)/(sig*np.sqrt(2))))
def Sigmoid(x,a=1):
    return 1./(1+np.exp(-x/a))
mdot=np.linalg.multi_dot

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

def Gaussian1D(x,mu=0.,sig=1.):
    return 1./(sig*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2.*sig**2))

def BetaCDF(x,pars):
    a,b=pars
    #a=Sigmoid(a)
    #b=Sigmoid(b)
    #print(a,b)
    return scipy.stats.beta.logcdf(x,a=a,b=b)

def BetaPDF(x,pars):
    a,b=pars
    return scipy.stats.beta.logpdf(x,a=a,b=b)


def Fit_logPW(x,y,GaussPars=(0,1),func=None):
    def D(pars):
        return np.sum(w*(y-func(x,pars))**2)
    w=1.
    F=ClassF(x,np.exp(y))
    xx=np.linspace(x.min(),x.max(),100)
    Fi=ClassF(xx,F(xx)).diff()
    w=Fi(x)
    mu=np.mean(x)
    sig=np.std(x)
    al=10**(np.linspace(-1,7,1000))
    M=1./mu-1.
    be=al*M
    V_=al*be/( (al+be)**2 * (al+be+1))
    Al=ClassF(V_,al)(sig**2)
    Be=Al*M
    pars0=Al[()],Be[()]
    Chi2a=D(pars0)/x.size
    pars=scipy.optimize.minimize(D,pars0)["x"]
    Chi2b=D(pars)/x.size
    return pars,Chi2a,Chi2b,pars0

    


class ClassShapiroWilk():
    def __init__(self,Scale="linear"):
        self.xSampleFunc=x=np.linspace(-7,7,3001)
        self.f_Phi=ClassF(x,Phi(x))
        # f_Phi1=ClassF(x,1-Phi(x))
        self.f_Gauss1D=ClassF(x,Gaussian1D(x))
        self.AOrderCoef=None

    def generateOrderStats(self,n):
        NTry=10000
        log.print("Measure order statistics for %i-size using %i samples"%(n,NTry))
        x=self.xSampleFunc
        Lm=[]
        Phix=self.f_Phi(x)
        LX=[]
        for i in range(NTry):
            X0=np.sort(np.random.randn(n))
            X1=np.sort(-X0)
            LX.append(X0)
            LX.append(X1)
        X=np.array(LX)
        m=np.mean(X,axis=0)
        self.mOrderCoef=m
        m=m.reshape((-1,1))
        x=x.reshape((-1,1))
        V=(1./NTry)*np.dot(X.T,X)
        Vinv=ModLinAlg.invSVDHermitianCut(V,Th=1e-2)
        m0=np.dot(Vinv,m.reshape((-1,1)))
        C=np.sqrt(np.sum(m0.flatten()**2))
        self.AOrderCoef=(np.dot(m.reshape((1,-1)),Vinv)).flatten()/C
        self.AOrderCoef=((self.AOrderCoef-self.AOrderCoef[::-1])/2.)
        self.VinvOrderCoef=Vinv#np.diag(Vinv)
        dA=self.AOrderCoef[1:]-self.AOrderCoef[:-1]
        dA*=np.sign(dA[0])
        if np.count_nonzero(dA<0)>0:
            log.print("AOrderCoef is non monotonous")
            stop
        
        self.MaxW=self.giveW(m0)
        
        log.print("Maximum possible W value: %f"%self.MaxW)
        
        # pylab.figure("Order Stats")
        # pylab.subplot(1,2,1)
        # pylab.plot(self.AOrderCoef)
        # pylab.subplot(1,2,2)
        # pylab.imshow(Vinv,interpolation="nearest")
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)

        
    def Init(self,n,NTry=10000):
        L_y=[]
        if self.AOrderCoef is None: self.generateOrderStats(n)
        log.print("Number of generated %i-size samples: %i"%(n,NTry))
        L_y1=[]
        
        for iTry in range(NTry):
            X=np.random.randn(n)
            L_y.append(self.giveW(X))
            L_y1.append(scipy.stats.shapiro(X)[0])
         
        log.print("Compute cumulative distribution...")
        P=self.empirical_FW=giveIrregularCumulDist(np.array(L_y),Type="Continuous")
        PT=P.T()
        x=np.linspace(PT(0.001),PT(0.95),200)
        P1=ClassF(x,P(x))

        Pscipy=self.empirical_FW=giveIrregularCumulDist(np.array(L_y1),Type="Continuous")
        PscipyT=Pscipy.T()
        xscipy=np.linspace(PscipyT(0.001),PscipyT(0.95),200)
        Pscipy1=ClassF(xscipy,Pscipy(xscipy))
        
        self.empirical_PW=P1.diff()
        m0,m1=PT(np.array([0.16,0.5]))
        med,sig=m1,m1-m0
        log.print("W-distribution (Med, Sig)=(%f, %f)"%(med,sig))
        self.EmpWDistMed=med
        self.EmpWDistSig=sig

        
        log.print("Fit cumulative distribution...")
        func=BetaCDF
        self.pars_fit_logPW,Chi2a,Chi2b,parsinit=Fit_logPW(self.empirical_FW.x,np.log(self.empirical_FW.y),GaussPars=(med,sig),func=func)
        log.print("  reduced Chi-square of fit = ( %f -> %f )"%(Chi2a,Chi2b))
        self.logP_W=lambda x: BetaPDF(x,self.pars_fit_logPW)
        
        Fm=self.empirical_FW
        Pm=self.empirical_PW
        Pfit=self.logP_W

        mu=med#self.empirical_FW.T()(0.5)
        mu=np.sum(Pm.y*Pm.x)/np.sum(Pm.y)
        sig=np.sqrt(np.sum(Pm.y*(Pm.x-mu)**2)/np.sum(Pm.y))

        xx=np.linspace(0.,1.,1000)
        
        fig=pylab.figure("logP-W")
        
        pylab.subplot(1,2,1)
        pylab.scatter(Fm.x,np.log(Fm.y))
        xx=Fm.x#np.linspace(0.01,0.99,100)
        pylab.plot(Fm.x,func(Fm.x,parsinit))
        pylab.plot(Fm.x,BetaCDF(Fm.x,self.pars_fit_logPW),ls="--",color="black")
        pylab.xlim(Fm.x.min(),Fm.x.max())
        pylab.subplot(1,2,2)
        pylab.scatter(Pscipy1.diff().x,np.log(Pscipy1.diff().y),c="red")
        pylab.scatter(Pm.x,np.log(Pm.y))#,edgecolors="black")
        pylab.plot(Pm.x,self.logP_W(Pm.x),ls="--",color="black")
        pylab.xlim(Pm.x.min(),Pm.x.max())
        pylab.draw()
        pylab.show(block=False)


    # #########################
    def giveW(self,X0,DoPlot=False):
        X=np.sort(X0)
        n=X.size
        W= ( np.sum(self.AOrderCoef.flatten()*X.flatten()) )**2 / (np.sum( (X-np.mean(X))**2 ) )
        #W= ( np.sum(self.AOrderCoef.flatten()*X.flatten()) )**2
        #W= 1 / (np.sum( (X-np.mean(X))**2 ) )
        #W= np.sum((X-np.mean(X)))
        return W

    # #########################
    # Jacob
    
    def dW_dx(self,x):
        n=x.size
        ind=np.argsort(x.flatten())
        nn=np.arange(n)
        #indInv=np.arange(n)[ind]
        indInv=np.argsort(nn[ind])
        x=np.sort(x.flatten())
        a=self.AOrderCoef.flatten()
        xm=np.mean(x)
        A=np.sum((x-xm)**2)
        aTx=np.sum(a*x)
        # # ###########
        # # ok
        B=2*a*aTx/A
        D=(aTx)**2 * (x-xm)
        E=A**(-2)*np.ones((n,),np.float32)
        C= E * D
        dW_dx=B-2*C
        return dW_dx[indInv]
        # #############
        

    # #########################
    # Hessian

    def d2W_dx2(self,x):
        n=x.size
        ind=np.argsort(x.flatten())
        nn=np.arange(n)
        indInv=np.argsort(nn[ind])
        x=np.sort(x.flatten())
        a=self.AOrderCoef.flatten()
        xm=np.mean(x)
        A=np.sum((x-xm)**2)
        aTx=np.sum(a*x)
        # #################
        # ok
        dB_dx=-4*aTx* A**(-2) * a * (x-xm)  + 2 * A**(-1) * a**2
        #return dB_dx[indInv]
        dD_dx=(x-xm)*2*a*aTx+(aTx)**2*(1.-1/n)
        dE_dx=-4*A**(-3)*(x-xm)
        D=aTx**2*(x-xm)
        E=A**(-2)
        dC_dx= ( D*dE_dx+E*dD_dx)
        dJ_dx=dB_dx-2*dC_dx
        return dJ_dx[indInv]
        # #################
    
    # #########################
    # Measure
    
    def meas_dW_dx(self,X):
        #X=np.sort(X)
        LJ=[]
        for iParm in range(X.size):
            dx=np.linspace(-.001,.001,2)
            logL=[]
            Lx=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                logL.append(self.giveW(Xix))
                Lx.append(Xix[iParm])
            J=(logL[1]-logL[0])/(dx[1]-dx[0])
            LJ.append(J)
        
        return np.array(LJ)

    def meas_d2W_dx2(self,X):
        LH=[]
        for iParm in range(X.size):
            dx=np.linspace(-.001,.001,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.dW_dx(Xix)[iParm])
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)
        
        return np.array(LH)

    # ##################################################
    # ################################## logP
    
    def logP_x(self,X):
        return self.logP_W(self.giveW(X))
    
    def dlogPdx(self,x,Break=False):
        
        dWdx=self.dW_dx(x)
        W=self.giveW(x)
        # print(dWdx)
        # print(W)
        # print(self.dlogPdW(W))
        # if Break:
        #     stop
        return self.dlogPdW(W)*dWdx

    def dlogPdW(self,W):
        dx=self.EmpWDistSig/100.
        x0=np.max([0.,W-dx])
        x1=np.min([1.,W+dx])
        return (self.logP_W(x1)-self.logP_W(x0))/(x1-x0)#

    def d2logPdW(self,W):
        dx=self.EmpWDistSig/100.
        x0=np.max([0.,W-dx])
        x1=np.min([1.,W+dx])
        return (self.dlogPdW(x1)-self.dlogPdW(x0))/(x1-x0)#
            
    def d2logPdx2(self,X):
        W=self.giveW(X)
        A=self.d2logPdW(W)
        B=self.dW_dx(X)
        C=self.dlogPdW(W)
        D=self.d2W_dx2(X)
        return A*B**2+C*D
    
    

    

    

    def meas_dlogP_dx(self,X):
        LH=[]
        for iParm in range(X.size):
            dx=np.linspace(-.001,.001,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.logP_x(Xix))
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)

        return np.array(LH)

    def meas_d2logP_dx2(self,X):
        LH=[]
        for iParm in range(X.size):
            dx=np.linspace(-.01,.01,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.dlogPdx(Xix)[iParm])
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)
        
        return np.array(LH)
    
    

