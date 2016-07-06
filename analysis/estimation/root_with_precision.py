
from sys import float_info

def middle(f,x,fp):
    if len(x)==2:
        return (x[0]+x[-1])/2.
    else:
        xp=f(x[1])/fp(x[1])
        if x[0]<xp<x[-1]:
            return xp
        else:
            return (x[0]+x[-1])/2.

from numpy import sign, Inf, isnan, any

def root_interval(f,x,middle,fp=None):
    """
    f is a monotonic function defined over an interval (x[0],x[-1]).
    x is a pair or a triple such that the root of f is within that same interval.
    middle is a function mapping a function and an interval into a point within the interval.
    """ 
    if len(x)==2: # If only endpoints provided supply an interior guess
        x=[x[0],middle(f,x,fp),x[1]]
        yield x

    y=[f(z) for z in x]
    while abs(x[0]-x[-1])>0:
        if y[0]==0.: # Found the root!
            x=[x[0]]*3
            yield x
        elif y[1]==0.:
            x=[x[1]]*3
            yield x
        elif y[-1]==0.:
            x=[x[-1]]*3
            yield x
        elif sign(y[0])==sign(y[-1]):
            raise ValueError, "No root in interval (%f,%f) or function not monotone." % (a,b)
        elif sign(y[0])==sign(y[1]): # Root must be in upper sub-interval
            x=[x[1],middle(f,[x[1],x[-1]],fp),x[-1]]
            y=[y[1],f(x[1]),y[-1]]
            yield x
        elif sign(y[1])==sign(y[2]): # Root must be in lower sub-interval
            x=[x[0],middle(f,[x[0],x[1]],fp),x[1]]
            y=[y[0],f(x[1]),y[1]]
            yield x
        elif any(isnan(y)):
            print x,y
            raise FloatingPointError, "Function not a number at some point in x."
        else:
            print x,y
            raise AssertionError, "Something impossible happened."

def root_in_open_interval(f,x,middle=middle):
    (a,x0,b)=x
    if b==Inf: b=float_info.max
    radius=[min(x0-a,b-x0)]*2
    xc=[x0-radius[0]/2.,x0,x0+radius[1]/2.] # Closed interval
    while sign(f(xc[0]))==sign(f(xc[-1])): # No root in closed interval; expand toward bounds
        if xc[0]-a < radius[0]:
            xc[0]=(a+xc[0])/2.
        else:
            radius[0]+=radius[0]
            xc[0]=x0-radius[0]/2.
        if b-xc[1] < radius[1]:
            xc[-1]=(b+xc[1])/2.
        else:
            radius[1]+=radius[1]
            xc[-1]=x0+radius[1]/2.

    return root_interval(f,xc,middle)

def root_with_precision(f,axb,tol,open_interval=False,middle=middle):
    if open_interval:
        seq=root_in_open_interval(f,axb,middle)
    else:
        seq=root_interval(f,axb,middle)
    x=seq.next() 
    while abs(x[0]-x[-1])>tol:
        x=seq.next()

    return x[1]
