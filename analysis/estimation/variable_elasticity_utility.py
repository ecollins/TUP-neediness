#!/usr/bin/env python

from scipy import optimize 
from numpy import array, ones, zeros, sum, log, Inf, dot, nan
from root_with_precision import root_with_precision
import warnings

def check_args(p,alpha,gamma,phi):
    """
    Perform sanity check on inputs.  Supply default values if these are missing.
    """

    # Make sure all args are of type array:
    p=array(p)

    try: 
        len(alpha) # If len() not defined, then must be a singleton
        alpha=array(alpha)
    except TypeError: alpha=array([alpha])

    try:
        len(gamma) # If len() not defined, then must be a singleton
        gamma=array(gamma)
    except TypeError: gamma=array([gamma])

    try:
        len(phi) # If len() not defined, then must be a singleton
        phi=array(phi)
    except TypeError: phi=array([phi])

    n=len(p)

    if len(alpha)==1<n:
        alpha=ones(n)*alpha
    else:
        if not alpha.all():
            raise ValueError

    if len(gamma)==1<n:
        gamma=ones(n)*gamma
    else:
        if not gamma.all():
            raise ValueError
    
    if len(phi)==1<n:
        phi=ones(n)*phi

    return (n,alpha,gamma,phi)

def dfdx(f,h=2e-5,tol=2e-8,maxsteps=5e3, maxslope=1e10):
    """
    computes numerical derivative of f(x) as lim (f(x+h/2)-f(x-h/2))/h as h->0
    Fixes "Bug" in derivative function.
    """

    def df(x,h=h,tol=tol,maxsteps=maxsteps,maxslope=maxslope):
        stepsize=h/maxsteps
        D0 = ( f(x+h/2) - f(x-h/2) )/h
        diff = 1.
        for steps in xrange(int(maxsteps)):
            h -= stepsize
            D1 = ( f(x+h/2) - f(x-h/2) )/h
            if D1>maxslope:
                print("Derivative seems to diverge?\nIncrease maxslope>{} to keep going.".format(maxslope))
                break
            diff = abs(D1-D0)
            if diff<tol:
                #~ Check if convergence is stable
                h2 = h-stepsize
                D2 = ( f(x+h2/2) - f(x-h2/2) )/h2
                if abs(D2-D0)<tol: return D2
            else: D0 = D1 
        if steps+1==maxsteps: print("Derivative failed to converge. Maybe tweak parameters of dfdx?")
    return df

def derivative(f,h=2e-5):
    """
    Computes the numerical derivative of a function with a single scalar argument.

    BUGS: Would be better to actually take a limit, instead of assuming that h 
    is infinitesimal.
    """
    def df(x, h=h):
        return ( f(x+h/2) - f(x-h/2) )/h
    return df

def frischdemands(lbda,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Given marginal utility of income lbda and prices, 
    returns a list of $n$ quantities demanded, conditional on 
    preference parameters (alpha,gamma,phi).
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=[((alpha[i]/(p[i]*lbda))**(1/gamma[i]) - phi[i]) for i in range(n)]

    if not NegativeDemands:
        x=[max(x[i],0.) for i in range(n)]        

    return x

def frischV(lbda,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Returns value of Frisch Indirect Utility function
    evaluated at (lbda,p) given preference parameters (alpha,gamma,phi).
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=frischdemands(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    U=0
    for i in range(n):
        if gamma[i]==1:
            U += alpha[i]*log(x[i]+phi[i])
        else:
            U += alpha[i]*((x[i]+phi[i])**(1-gamma[i])-1)/(1-gamma[i])

    return U

def excess_expenditures(y,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Return a function which will tell excess expenditures associated with a lambda.
    Elliott: now using dot for d instead of a for loop
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    n = len(p)


    def f(lbda):
        lbda=abs(lbda)

        x=frischdemands(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

        d = dot(x,p)

        return d - y

    return f

def excess_expenditures_derivative(p,alpha,gamma,phi):
    """
    Return derivative of excess expenditures function with respect to lambda
    eed(p,a,g,p)(lbda)*dlbda ~= dy
    Have p0 (pull out prices of other goods? No way around some price measure? Why not call them all 1 in year0 and move on?)
    Need lbda. This is the crux, isn't it? Need Ethan's code? Does it make sense in this context?
    OR: Maybe we can assume that we have y in year0, like if we have small surveys between larger LSMS-type surveys...
        Then plug into lambdavalue function.
    Solve for each HH & take difference in means? Or adjust dloglambda (multiplicatively or additively)
    and solve for control group's counterfactual dy?
    ALSO: What the hell am I supposed to do about alpha, gamma, and phi?
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    n = len(p)
    d=1/gamma

    def df(lbda):

        lbda=abs(lbda)
        y=0.0
        for i in range(n):
            y += -d[i]*p[i]*(alpha[i]/(p[i]))**(d[i])*lbda**-(1+d[i])

        return y 

    return df

def excess_utility(U,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Return a function which will tell excess utility associated with a lambda.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    n = len(p)
    def f(lbda):

        return U - frischV(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return f

def lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=True,ub=10,method='bisect'):
    """
    Given income y, prices p and preference parameters
    (alpha,gamma,phi), find the marginal utility of income lbda.
    Elliott: Changed default method from 'root_with_precision' to 'bisect'.
    I think Rw/P was hanging up somehow because the fake_data simulation freezes when direct==False
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    if NegativeDemands:
        subsistence=sum([p[i]*phi[i] for i in range(n)])
    else:
        subsistence=sum([p[i]*phi[i] for i in range(n) if phi[i]<0])
        
    if y+subsistence<0: # Income too low to satisfy subsistence demands
        warnings.warn('Income too small to cover subsistence phis (%f < %f)' % (y,subsistence))
        return nan

    f = excess_expenditures(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    if method=='bisect':
        try:
            return optimize.bisect(f,1e-20,ub)
        except ValueError:
            return lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands,ub=ub*2.0)
    elif method=='newton':
        df = excess_expenditures_derivative(p,alpha,gamma,phi)
        return optimize.newton(f,ub/2.,fprime=df)
    elif method=='root_with_precision':
        return root_with_precision(f,[0,ub,Inf],1e-12,open_interval=True)
    else:
        raise ValueError, "Method not defined."

def marshalliandemands(y,p,alpha,gamma,phi,NegativeDemands=True):
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    lbda=lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return frischdemands(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

def indirectutility(y,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Returns utils associated with income y and prices p.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    lbda=lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return frischV(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

def lambdaforU(U,p,alpha,gamma,phi,NegativeDemands=True,ub=10):
    """
    Given level of utility U, prices p, and preference parameters
    (alpha,gamma,phi), find the marginal utility of income lbda.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    f = excess_utility(U,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    # Our root-finder looks within an interval [1e-20,ub].  If root
    # isn't in this interval, optimize.bisect will raise a ValueError;
    # in this case, try again, but with a larger upper bound.
    try:
        #return optimize.bisect(f,1e-20,ub)
        return root_with_precision(f,[0,ub,Inf],1e-12,open_interval=True)
    except ValueError:
        return lambdaforU(U,p,alpha,gamma,phi,NegativeDemands=True,ub=ub*2.0)

def expenditurefunction(U,p,alpha,gamma,phi,NegativeDemands=True):
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=hicksiandemands(U,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return sum(array([p[i]*x[i] for i in range(n)]))

def hicksiandemands(U,p,alpha,gamma,phi,NegativeDemands=True):
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    lbda=lambdaforU(U,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return frischdemands(lbda,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

def hicksianbudgetshares(U,p,alpha,gamma,phi,NegativeDemands=True):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    
    h=hicksiandemands(U,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    y=expenditurefunction(U,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return array([p[i]*h[i]/y for i in range(n)])
    
def expenditures(y,p,alpha,gamma,phi,NegativeDemands=True):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    
    x=marshalliandemands(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return array([p[i]*x[i] for i in range(n)])

def budgetshares(y,p,alpha,gamma,phi,NegativeDemands=True):
    
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    
    x=expenditures(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    return array([x[i]/y for i in range(n)])

def share_income_elasticity(y,p,alpha,gamma,phi,NegativeDemands=True):
    """
    Expenditure-share elasticity with respect to total expenditures.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    def w(xbar):
        return budgetshares(xbar,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    dw=derivative(w)

    return [dw(y)[i]*(y/w(y)[i]) for i in range(n)]

def income_elasticity(y,p,alpha,gamma,phi,NegativeDemands=True):

    return array(share_income_elasticity(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands))+1.0
    

def main(y,p,alpha,gamma,phi,NegativeDemands=True):
    n=len(p)
    print 'lambda=%f' % lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    print 'budget shares '+'%6.5f\t'*n % tuple(budgetshares(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands))
    print 'share income elasticities '+'%6.5f\t'*n % tuple(share_income_elasticity(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands))
    print 'indirect utility=%f' % indirectutility(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    
    # Here's a test of the connections between different demand representations:
    print "Testing identity relating expenditures and indirect utility...",
    V=indirectutility(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    X=expenditurefunction(V,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    assert abs(y-X)<1e-6
    print "passed."
    
    def V(xbar):
        return indirectutility(xbar,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    dV=derivative(V)

    tol=1e-6

    try:
        print "Evaluating lambda-V'''...",
        lbda=lambdavalue(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
        assert abs(dV(y)-lbda)<tol
        print "within tolerance %f" % tol
    except AssertionError:
        print "dV=%f; lambda=%f" % (dV(y),lbda)

if __name__=="__main__":
    print "Single good; negative phi"
    main(1.,[1],[1],[1],[-2.],NegativeDemands=False)

    print "Passed."
    print

    print "Two goods; phis of different signs; no negative demands"
    main(3,[1]*2,[1]*2,[1]*2,[2,-2.],NegativeDemands=False)

    print "Passed."
    print

    print "Two goods; phis of different signs; negative demands allowed"
    main(3,[1]*2,[1]*2,[1]*2,[2,-2.],NegativeDemands=True)

    print "Passed."
    print

    y=6
    p=array([10.0,15.0])
    alpha=array([0.25,0.75])
    gamma=array([2.0,0.5])
    phi=array([-.1,0.0])

    main(y,p,alpha,gamma,phi)
