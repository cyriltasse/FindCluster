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


def Fit_logPA2(x,y,GaussPars=(0,1),func=None):
    
    
    # w=np.exp(y/np.sum(y))
    # ind=np.where(np.isnan(w) | np.isinf(w))[0]
    # w[ind]=0
    # # #w=np.sqrt(w)
    # w/=np.sum(w)
    # # #w.fill(0)
    # # med,sig=GaussPars
    # # x0=x[np.where(y==y.max())[0]][0]
    # # print("x0",x0)
    # # a0,b0=np.polyfit(x[x<x0], y[x<x0], 1)
    # # a1,b1=np.polyfit(x[x>x0], y[x>x0], 1)
    # # s0,s1=x0,(sig/2.)
    # # pars0=a0,b0,a1,b1,s0,s1

    def D(pars):
        return np.sum(w*(y-func(x,pars))**2)
        #return np.sum((y-func(x,pars))**2)
    
    w=1.
#     nn=50
#     print("init")
#     A,B=np.mgrid[-1:5:nn*1j,-4:2:nn*1j]
#     A=10**(A)
#     B=10**(B)
#     Chi2=np.zeros((nn,nn),np.float32)
# #    for (a,b) in zip(A.flatten(),B.flatten()):
#     for i in range(nn):
#         for j in range(nn):
#             a=A[i,j]
#             b=B[i,j]
#             pars=a,b
#             Chi2[i,j]=D(pars)


#     ind=np.where(Chi2==Chi2.min())
#     i,j=ind[0][0],ind[1][0]
#     a,b=A[i,j],B[i,j]
#     pars0=a,b
#     print("done init")


    F=ClassF(x,np.exp(y))
    xx=np.linspace(x.min(),x.max(),100)
    Fi=ClassF(xx,F(xx)).diff()
    w=Fi(x)
    #mu=np.sum(F.y*F.x)/np.sum(F.y)
    #sig=np.sqrt(np.sum(F.y*(F.x-mu)**2)/np.sum(F.y))
    mu=np.mean(x)
    sig=np.std(x)
    
    al=10**(np.linspace(-1,6,1000))
    
    M=1./mu-1.
    be=al*M
    V_=al*be/( (al+be)**2 * (al+be+1))
    Al=ClassF(V_,al)(sig**2)
    Be=Al*M
    pars0=Al[()],Be[()]
    # stop
    # array([4.49846004e+03, 3.71495573e+00])
    #3.86731910e+03 3.01140042e+00
    
    Chi2a=D(pars0)/x.size
    pars=scipy.optimize.minimize(D,pars0)["x"]
    #print("parsout",pars)
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
        
        
        # V=np.var(X,axis=0)
        # m0=m/V
        # C=np.sqrt(np.sum(m0**2))
        # self.AOrderCoef=(m/V)/C
        
        V=(1./NTry)*np.dot(X.T,X)


        

        
        #Vinv=ModLinAlg.invSVDHermitianCut(V)#np.real(ModLinAlg.invSVD(V))
        Vinv=ModLinAlg.invSVDHermitianCut(V,Th=1e-2)
        #Vinv=(Vinv.T+Vinv)/2.
        #Vinv=np.diag(np.diag(Vinv))
        #print(np.diag(np.dot(V,Vinv)))
        #Vinv=np.linalg.inv(V)
        m0=np.dot(Vinv,m.reshape((-1,1)))
        C=np.sqrt(np.sum(m0.flatten()**2))
        self.AOrderCoef=(np.dot(m.reshape((1,-1)),Vinv)).flatten()/C
        #print("C2a",C**2)
        #self.AOrderCoef=(np.dot(m.reshape((1,-1)),Vinv)).flatten()
        self.AOrderCoef=((self.AOrderCoef-self.AOrderCoef[::-1])/2.)
        #print("AOrderCoef",self.AOrderCoef)
        #print("Sum AOrderCoef",np.sum(self.AOrderCoef**2))
        self.VinvOrderCoef=Vinv#np.diag(Vinv)
        dA=self.AOrderCoef[1:]-self.AOrderCoef[:-1]
        dA*=np.sign(dA[0])
        if np.count_nonzero(dA<0)>0:
            log.print("AOrderCoef is non monotonous")
            stop
        # m=m.reshape((-1,1))
        # x=m0.reshape((-1,1))
        # C2=mdot([m.T,self.VinvOrderCoef,self.VinvOrderCoef,m]).flat[0]
        # print("C2b",C2)
        # AOrderCoef=mdot([m.T,self.VinvOrderCoef])/np.sqrt(C2)
        # res=mdot([AOrderCoef,np.sort(x.flatten()).reshape((-1,1))])**2/((np.sum( (x-np.mean(x))**2 ) ))
        # res=mdot([AOrderCoef,x.reshape((-1,1))])**2/((np.sum( (x-np.mean(x))**2 ) ))
        # print("res",res)

        # #C=mdot([m.T,Vinv,Vinv,m])
        # def giveW0(X0,DoPlot=False):
        #     XX=np.sort(X0.flatten())
        #     n=XX.size
        #     W= ( np.sum(self.AOrderCoef.flatten()*XX.flatten()) )**2 / (np.sum( (XX-np.mean(XX))**2 ) )
        #     #W1=mdot([self.AOrderCoef.reshape((1,-1)),XX])**2
        #     return W
        
        self.MaxW=self.giveW(m0)
        
        log.print("Maximum possible W value: %f"%self.MaxW)


        # # ################### Test coefs 17  #######################
        
        # Ath=np.array([-0.4968,-.3273,-.2540,-.1988,-.1524,-.1109,-.0725,-.0359])
        # Ath=np.array(Ath.tolist()+[0.]+(-Ath[::-1]).tolist())

        
        # u,s,v=np.linalg.svd(np.complex128(V))
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.plot(np.log(s.flatten()))
        # #pylab.imshow(Vinv,interpolation="nearest")
        # #pylab.colorbar()
        # #pylab.plot(np.log(s.flatten()))
        # pylab.subplot(1,2,2)
        # #pylab.imshow(Vinv1,interpolation="nearest")
        # #pylab.colorbar()
        # pylab.plot(self.AOrderCoef.flatten())
        # #pylab.plot(Ath)
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)
        # stop
        
        # # pylab.figure("Order Stats")
        # # pylab.subplot(1,2,1)
        # # pylab.plot(self.AOrderCoef)
        # # #pylab.plot(m,self.AOrderCoef)
        # # pylab.subplot(1,2,2)
        # # pylab.imshow(Vinv,interpolation="nearest")
        # # pylab.draw()
        # # pylab.show(block=False)
        # # pylab.pause(0.1)

        
    def generatePW(self,n,NTry=10000,UseScipy=False):
        L_y=[]
        if self.AOrderCoef is None: self.generateOrderStats(n)
        log.print("Number of generated %i-size samples: %i"%(n,NTry))
        #L_y1=[]
        for iTry in range(NTry):
            X=np.random.randn(n)
            if UseScipy:
                L_y.append(scipy.stats.shapiro(X)[0])
            else:
                DoPlot=0
                if iTry==0: DoPlot=1
                #L_y.append(self.giveW(X,DoPlot=DoPlot)/scipy.stats.shapiro(X)[0])
                L_y.append(self.giveW(X,DoPlot=DoPlot))
                #print(scipy.stats.shapiro(self.mOrderCoef)[0])
                #stop

        # print(scipy.stats.shapiro(self.mOrderCoef)[0])
        # print(self.giveW(self.AOrderCoef))#self.mOrderCoef))
        # X=np.random.randn(n)
        # X=np.random.randn(n)
        # print("one",self.giveW(X))
        # X[0]=-4
        # print("two",self.giveW(X))
         
        log.print("Compute cumulative distribution...")
        P=self.empirical_FA2=giveIrregularCumulDist(np.array(L_y),Type="Continuous")
        PT=P.T()
        x=np.linspace(PT(0.001),PT(0.95),200)
        P1=ClassF(x,P(x))
        
        self.empirical_PA2=P1.diff()
        m0,m1=PT(np.array([0.16,0.5]))
        med,sig=m1,m1-m0
        # log.print("Fit cumulative distribution...")
        # func=logTwoSlopes
        # self.pars_fit_logPA2,Chi2a,Chi2b,parsinit=Fit_logPA2(self.empirical_PA2.x,np.log(self.empirical_PA2.y),GaussPars=(med,sig),func=func)
        log.print("Fit cumulative distribution...")
        func=BetaCDF
        self.pars_fit_logPA2,Chi2a,Chi2b,parsinit=Fit_logPA2(self.empirical_FA2.x,np.log(self.empirical_FA2.y),GaussPars=(med,sig),func=func)
        log.print("  reduced Chi-square of fit = ( %f -> %f )"%(Chi2a,Chi2b))
        self.logP_A2=lambda x: func(x,self.pars_fit_logPA2)

        
        Fm=self.empirical_FA2
        Pm=self.empirical_PA2
        Pfit=self.logP_A2

        mu=med#self.empirical_FA2.T()(0.5)
        mu=np.sum(Pm.y*Pm.x)/np.sum(Pm.y)
        sig=np.sqrt(np.sum(Pm.y*(Pm.x-mu)**2)/np.sum(Pm.y))

        xx=np.linspace(0.,1.,1000)
        self.fP_A2=ClassF(x,np.exp(self.logP_A2(x))).diff()
        
        fig=pylab.figure("logP-W")
        
        pylab.subplot(1,2,1)
        pylab.scatter(Fm.x,np.log(Fm.y))
        xx=Fm.x#np.linspace(0.01,0.99,100)
        pylab.plot(Fm.x,func(Fm.x,parsinit))
        pylab.plot(Fm.x,self.logP_A2(Fm.x),ls="--",color="black")
        pylab.xlim(Fm.x.min(),Fm.x.max())
        # y=Phi(Fm.x,mu=mu,sig=sig)
        # y/=y.max()
        # pylab.plot(Fm.x,y)
        #pylab.xlim(0,100)
        pylab.subplot(1,2,2)
        #pylab.scatter(Pm.x,np.log(Pm.y))#,edgecolors="black")
        #pylab.scatter(Pm.x,Pm.y)#,edgecolors="black")
        pylab.scatter(Pm.x,np.log(Pm.y))#,edgecolors="black")
        #pylab.plot(Pm.x,Pfit(Pm.x))
        pylab.plot(self.fP_A2.x,np.log(self.fP_A2.y),ls="--",color="black")
        #pylab.plot(Pm.x,np.log(Gaussian1D(Pm.x,mu=mu,sig=sig)),ls=":")
        pylab.xlim(Pm.x.min(),Pm.x.max())
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

    # #########################
    def giveW(self,X0,DoPlot=False):
        X=np.sort(X0)
        n=X.size
        #W= ( np.sum(self.AOrderCoef*X) )**2 / (np.sum( (X-np.mean(X))**2 ) )
        W= ( np.sum(self.AOrderCoef.flatten()*X.flatten()) )**2 / (np.sum( (X-np.mean(X))**2 ) )
        return W
        m=self.mOrderCoef

        x=X
        return np.sum(m*self.VinvOrderCoef*x)/np.sum(m**2*self.VinvOrderCoef)


        m=m.reshape((-1,1))
        x=x.reshape((-1,1))
        res=mdot([m.T,self.VinvOrderCoef,x])**2/(mdot([m.T,self.VinvOrderCoef,self.VinvOrderCoef,m])*(np.sum( (X-np.mean(X))**2 ) ))
        return res.flat[0] 
        
        #W= ( np.sum(self.AOrderCoef-X) )**2 / (np.sum( (X-np.mean(X))**2 ) )
        
        # if DoPlot:
        #     pylab.figure("WW")
        #     pylab.scatter(X,self.AOrderCoef*X)
        #     pylab.xlabel("x")
        #     pylab.ylabel("a")
        #     pylab.draw()
        #     pylab.show(block=False)
        #     pylab.pause(0.1)
        
        return W

        
    def logP_x(self,X):
        if self.logP_A2 is None:
            self.generatePA2(X.size)
        return self.logP_A2(self.giveA2(X))
    
    def dlogPdx(self,x):
        
        dA2dx=self.dA2_dx(x)
        A2=self.giveA2(x)
        return self.dlogPdA2(A2)*dA2dx

    def dlogPdA2(self,A2):
        if self.logP_A2 is None:
            log.print("Need a logP function")
            stop
        dx=0.001
        return (self.logP_A2(A2+dx)-self.logP_A2(A2-dx))/(2*dx)#

    def d2logPdA2(self,A2):
        if self.logP_A2 is None:
            log.print("Need a logP function")
            stop
        dx=0.001
        return (self.dlogPdA2(A2+dx)-self.dlogPdA2(A2-dx))/(2*dx)#
            
    def d2logPdx2(self,X):
        A2=self.giveA2(X)
        A=self.d2logPdA2(A2)
        B=self.dA2_dx(X)
        C=self.dlogPdA2(A2)
        D=self.d2A2_dx2(X)
        return A*B**2+C*D
    
    

    

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
        
        #dA2_dxi= -2 *self.w(xi)* self.f_Gauss1D(xi) * (Diff(xi+1e-6)+Diff(xi-1e-6))/2.
        dA2_dxi= -2 *self.w(xi) * (Diff(xi+1e-6)+Diff(xi-1e-6))/2.
        #dA2_dxi= -2* (Diff(xi+1e-5)+Diff(xi-1e-5))/2.

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
        
        dA2_dxi= -2 *self.dwdx(xi)#* (Diff(xi+1e-6)+Diff(xi-1e-6))/2.
        dA2_dxi= -2*(-self.dwdx(xi)*(Diff(xi+1e-3)+Diff(xi-1e-3))/2.+self.w(xi)*self.f_Phi.diff()(xi))#((Diff(xi+1e-3)+Diff(xi-1e-3))/2.)**2
        #dA2_dxi= -2*(self.w(xi)*self.f_Phi.diff()(xi))#((Diff(xi+1e-3)+Diff(xi-1e-3))/2.)**2
        #dA2_dxi= -2*(self.dwdx(xi)*self.f_Phi.diff()(xi))#((Diff(xi+1e-3)+Diff(xi-1e-3))/2.)**2

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
    
    

