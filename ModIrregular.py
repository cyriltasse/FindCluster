from __future__ import division

import numpy as np
import scipy.special
import numpy as np
import pylab
import scipy.stats
import GeneDist
import scipy.optimize
import pylab
from scipy.interpolate import interp1d

erf=scipy.special.erf
G=scipy.special.gamma
logG=scipy.special.loggamma
Phi=lambda x: 0.5*(1+erf(x/np.sqrt(2)))

def test():
    x0=np.linspace(-4,4,10)
    y0=Phi(x0)
    f0=ClassF(x0,y0)

    np.random.seed(42)
    X=np.random.randn(5)
    f1=giveIrregularCumulDist(X)

    f2=((f1-f0)**2).inv()

    pylab.ion()
    pylab.clf()
    pylab.plot(x0,y0)
    pylab.plot(f1.x,f1.y)
    pylab.plot(f2.x,f2.y)
    pylab.draw()

    x=np.random.randn(10)
    x/=(x.max()-x.min())
    f2=ClassF(x*2,np.ones_like(x))
    print(f2.int())



class ClassF():
    def __init__(self,x,y):
        ind=np.argsort(x)
        self.x=x[ind]
        self.y=y[ind]
        self.update()
        
    def update(self,x=None,y=None):
        if x is not None:
            self.x=x
        if y is not None:
            self.y=y
        self.f=interp1d(self.x,self.y,bounds_error=False,fill_value=(self.y[0],self.y[-1]))
        
    def T(self):
        return ClassF(self.y,self.x)

    def __call__(self,x):
        return self.f(x)
        
    def give_y0y1(self,f1):
        x=np.sort(np.unique(np.concatenate([self.x,f1.x])))
        y0=self.f(x)
        y1=f1(x)
        return x,y0,y1
    
    def __sub__(self,f1):
        x,y0,y1=self.give_y0y1(f1)
        y2=y0-y1
        fout=ClassF(x,y2)
        return fout

    def __add__(self,f1):
        x,y0,y1=self.give_y0y1(f1)
        y2=y0+y1
        fout=ClassF(x,y2)
        return fout
    
    def __mul__(self,f1):
        x,y0,y1=self.give_y0y1(f1)
        y2=y0*y1
        fout=ClassF(x,y2)
        return fout
    
    def __pow__(self,n):
        return ClassF(self.x,self.y**n)
    
    def __truediv__(self,f1):
        x,y0,y1=self.give_y0y1(f1)
        y2=y0/y1
        fout=ClassF(x,y2)
        return fout
    
    def int(self,xmm=None,DoPlot=False):
        if xmm is None:
            x=self.x
        else:
            x=np.sort(np.concatenate([self.x,np.array(xmm)]))
        xc=(x[0:-1]+x[1:])/2.
        dx=(x[1:]-x[0:-1])
        yc=self(xc)

        if DoPlot:
            import pylab
            pylab.clf()
            pylab.plot(xc,yc)
            pylab.draw()
            stop
        
        return np.sum(yc*dx)

    def diff(self):
        f=self._diff3()
        C0=np.logical_not(np.isnan(f.y))
        C1=np.logical_not(np.isinf(f.y))
        ind=np.where(C0&C1)[0]
        f.x=f.x[ind]
        f.y=f.y[ind]
        f.update()
        return f
    
    def _diffOverGrid(self,xEdge):
        x,y=self.x,self.y
        xc=(xEdge[0:-1]+xEdge[1:])/2.
        dx=(xEdge[1:]-xEdge[0:-1])
        y=self(xEdge)
        dy=(y[1:]-y[0:-1])
        dydx=dy/dx
        return ClassF(xc,dydx)
    
    def _diff2(self):
        x,y=self.x,self.y
        #print(x)
        xc=(x[0:-1]+x[1:])/2.
        dx=(x[1:]-x[0:-1])
        dy=(y[1:]-y[0:-1])
        dydx=dy/dx
        print(xc)
        
        return ClassF(xc,dydx)
    
    def _diff3(self):
        x,y=self.x,self.y
        xc=x[1:-1]
        dx=(x[2:]-x[0:-2])
        dy=(y[2:]-y[0:-2])
        dydx=dy/dx
        return ClassF(xc,dydx)

    # def _diff5(self):
    #     x,y=self.x,self.y
    #     i0,i1=2,-2
    #     xc=x[i0:i1]
    #     F0=(y[i0-2:i1-2])
    #     F1=(y[i0-1:i1-1])
    #     F2=(y[i0+1:i1+1])
    #     F3=(y[i0+2:i1+2])
    #     dydx=dy/dx
        
    #     (-F3+8*F2-8*F1+F0)/(12*dx)


    def inv(self):
        return ClassF(self.x,1./self.y)
        
    
def giveIrregularCumulDist(X0,xmm=None,Type="Discrete"):
    if xmm is None:
        x0=X0.min()-1.
        x1=X0.max()+1.
    else:
        x0,x1=xmm
    X=np.float64(np.sort(X0))

    if Type=="Continuous":
        xx=X
        yy=np.linspace(0,1,xx.size+1)[1::]
        
    elif Type=="Discrete":
        NMin=1+X.size
        NMax=1+X.size
        y=np.linspace(0,1,X.size+1)
        
        xx=np.zeros((NMin*2,),np.float64)
        xx[0]=x0
        xx[1:-1][0::2]=X[:]-1e-10
        xx[1:-1][1::2]=X[:]+1e-10
        xx[-1]=x1
        yy=np.zeros_like(xx)
        yy[0]=0
        yy[1:-1][0::2]=y[0:-1]
        yy[1:-1][1::2]=y[1:]
        yy[-1]=1
        # import pylab
        # pylab.ion()
        # pylab.clf()
        # pylab.plot(xx,yy)
        # pylab.draw()
        # pylab.show(False)

    
    return ClassF(xx,yy)

