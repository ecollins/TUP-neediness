#!/usr/bin/env python
"""
This is a module with functions which implement various aspects of estimation of VES demand systems.
"""

from warnings import warn
import pandas as pd
from numpy import linalg
from numpy.linalg import norm
import numpy as np
import pylab as pl
#~ import sys
#~ sys.path.append('../Computation')
#~ sys.path.append('../Data/Uganda')
#~ import uganda
import variable_elasticity_utility as ves

def series_outer(x,y):
    """
    Outer product for pd.Series; returns pd.DataFrame.
    """
    z=x.as_matrix().reshape((-1,1))*y.as_matrix()
    return pd.DataFrame(z,index=x.index,columns=y.index)

def pandas2vic(*dfs):
    """
    Transform a pandas series or dataframe to a (values,index,columns) tuple; drop any rows with NaNs.
    """
    all=pd.concat(dfs,join='inner',axis=1).dropna()

    assert len(all)>0

    return tuple([(df.loc[all.index].values,all.index,df.columns) for df in dfs])

def vic2pandas(vic):
    """Transform (value,columns,index) to pandas.DataFrame."""
    return pd.DataFrame(**vic)

def use_indices(df,idxnames):
    return df.reset_index()[idxnames].set_index(df.index)

def proj(y,x,returnb=False):
    """
    Linear projection of a matrix y on a matrix x.

    If y and x are pandas.DataFrames or Series, then return similar objects.
    """

    vicx,vicy=pandas2vic(x,y)

    b=linalg.lstsq(vicx[0],vicy[0])[0]
    b=pd.DataFrame(b.T,index=y.columns,columns=x.columns)

    if returnb:
        return pd.DataFrame(np.dot(vicx[0],b.T),index=vicx[1],columns=vicy[2]),b.T
    else:
        return pd.DataFrame(np.dot(vicx[0],b.T),index=vicx[1],columns=vicy[2])

from scipy import sparse

def ols(x,y,return_se=True,return_v=False):

    x=pd.DataFrame(x) # Deal with possibility that x & y are series.
    y=pd.DataFrame(y)
    N,n=y.shape
    k=x.shape[1]

    # Drop any observations that have missing data in *either* x or y.
    x,y = drop_missing([x,y]) 
    
    b=linalg.lstsq(x,y)[0]

    b=pd.DataFrame(b,index=x.columns,columns=y.columns)

    if return_se or return_v:

        u=y-x.dot(b)

        # Use SUR structure if multiple equations; otherwise OLS.
        # Only using diagonal of this, for reasons related to memory.  
        S=sparse.dia_matrix((sparse.kron(u.T.dot(u),sparse.eye(N)).diagonal(),[0]),shape=(N*n,)*2) 

        # This will be a very large matrix!  Use sparse types
        V=sparse.kron(sparse.eye(n),(x.T.dot(x)).as_matrix().view(type=np.matrix).I.dot(x.T))
        V=V.dot(S).dot(V.T)/N

        if return_se:
            se=np.sqrt(V.diagonal()).reshape((x.shape[1],y.shape[1]))
            se=pd.DataFrame(se,index=x.columns,columns=y.columns)
            
            return b.T,se
        elif return_v:
            # Extract blocks along diagonal; return an Nxkxn array
            V={y.columns[i]:pd.DataFrame(V[i*k:(i+1)*k,i*k:(i+1)*k],index=x.columns,columns=x.columns) for i in range(n)}
            return b.T,V
    else:
        return b.T

def drop_missing(X):
    """
    Return tuple of pd.DataFrames in X with any 
    missing observations dropped.  Assumes common index.
    """
    nonmissing=X[0].copy()
    nonmissing['Nonmissing']=True
    nonmissing=nonmissing['Nonmissing']
    for x in X:
        nonmissing.where(pd.notnull(x).all(axis=1),False,inplace=True)

    for i in range(len(X)):
        X[i] = X[i].loc[nonmissing,:]

    return tuple(X)

def group_expenditures(df,groups):
    myX=pd.DataFrame(index=df.index)
    for k,v in groups.iteritems():
        myX[k]=df[['$x_{%d}$' % i for i in v]].sum(axis=1)
            
    return myX

def difference_over_time(df):
    mydf=df.copy()
    mydf.sortlevel(inplace=True)
    
    Rounds=df.index.levels[0]
    T=len(Rounds)
    N=len(df.index.levels[1])

    mydf.reset_index(level='Year',inplace=True)
    #~ ELLIOTT: until 2015-02-14, seems like this just provides differences between T-1 and T. Need to append ddf_t to ddf, not replace ddf
    ddf= mydf[mydf['Year']==Rounds[1]] - mydf[mydf['Year']==Rounds[0]]
    ddf['Year']=Rounds[1]
    for t in range(2,T):
        ddf_t= mydf[mydf['Year']==Rounds[t]] - mydf[mydf['Year']==Rounds[t-1]]
        ddf_t['Year']=Rounds[t]
        ddf=ddf.append(ddf_t)

    ddf.reset_index(inplace=True)
    ddf.set_index(['Year','HH'],inplace=True)

    return ddf

def balance_panel(df):
    """Drop households that aren't observed in all rounds."""
    pnl=df.to_panel()
    keep=pnl.loc[list(pnl.items)[0],:,:].dropna(how='any',axis=1).iloc[0,:]
    df=pnl.loc[:,:,keep.index].to_frame(filter_observations=False)
    df.index.names=pd.core.base.FrozenList(['Year','HH'])
    
    return df


def svd_approximation(x,rnk=1,returnSigma=False):
    """Compute best rank rnk approximation to the matrix x."""

    x=x.dropna()

    u,s,v=linalg.svd(x)

    print "SVD fit: %6.3f" % ((s[0]**2)/np.sum(s**2))
    
    S=np.zeros((u.shape[1],v.shape[0]))

    for i in range(rnk): S[i,i]=s[i]
    xhat = np.dot(np.dot(u,S),v).T

    xhat=pd.DataFrame(xhat.T,index=x.index,columns=x.columns)

    if returnSigma:
        return xhat,s
    else:
        return xhat

def estimate_with_time_effects(x,z,phi=1e-4,tol=1e+1):
    """
    Given a pd.Dataframe x of expenditure data, and a pd.DataFrame z
    of household characteristics, estimate parameters 1/gamma, alphabar,
    delta, and lambdas.

    This uses a specification of log expenditures as the dependent variable.

    Dataframes are indexed by (Year,HH), with columns in x
    corresponding to expenditure types (no other variables should
    appear), and columns in z corresponding to characteristics.
    """

    # Monkey around to eliminate observations with missing values, (or
    # with negative values for expenditures)
    logx = np.log(x+phi)

    logx,z = drop_missing([logx,z])
    
    barlogxit = logx.groupby(level='Year').mean()
    barzt = z.groupby(level='Year').mean()

    z_demeaned = z.to_panel().subtract(barzt,axis='minor').to_frame()
    logx_demeaned = logx.to_panel().subtract(barlogxit,axis='minor').to_frame()

    # Possible issues with missing data?
    assert(logx.shape==logx_demeaned.shape) 

    dhat,se_d=ols(z_demeaned,logx_demeaned)

    a_it = barlogxit

    cwT = logx_demeaned - z_demeaned.dot(dhat.T)

    cwThat = svd_approximation(cwT,rnk=1)

    R2 = cwThat.var()/cwT.var()
    e = cwT - cwThat # Residuals

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    if cwThat.iloc[0,:].mean()>0:
        b=cwThat.iloc[0,:]
    else:
        b=-cwThat.iloc[0,:]

    # Normalize so that mean value of b elasticity is one, like an income elasticity
    b=b/b.mean()
    b=pd.Series(b,index=x.columns)

    loglambdas=-cwThat.iloc[:,0]/b.iloc[0]

    deltas=dhat.T.divide(b)

    try:
        assert((abs(deltas.as_matrix())<np.inf).all())
    except AssertionError:
        warn('delta has some non-finite elements.')

    goodsdf=deltas.T.copy()

    logalphabars=a_it.mean(axis=0)/b
    
    #logalphabars=logalphabars-logalphabars.mean()  # Normalization

    # Now prices & average log lambdas per period

    atilde = a_it - a_it.mean(axis=0)

    barloglambda_t = -atilde.mean(axis=1)/b.mean()

    hhdf={z_l:z_demeaned[z_l] for z_l in deltas.index} # Grab household characteristics
    hhdf.update({'lambdas':loglambdas})
    hhdf = pd.DataFrame(hhdf)
    hhdf['lambdas'] = hhdf.to_panel()['lambdas'].add(barloglambda_t,axis=0).stack()

    atildetilde = atilde.subtract(series_outer(b,atilde.mean(axis=1)).T,axis=0)

    if np.any(b==1):
        warn("For goods with beta_i==1 prices aren't identified.")
    logrealprices = atildetilde.divide(1-b)

    goodsdf['beta']=b
    goodsdf['R2']=R2
    goodsdf['alphabar']=logalphabars
    goodsdf['phi']=phi

    # Now as a check put the Humpty-Dumpty of log expenditures back together
    ahat = (1-b)*logrealprices + b*logalphabars - series_outer(barloglambda_t,b) + e.mean()
    assert(np.linalg.norm(a_it - ahat)<1e-1)

    nonwealth = z_demeaned.dot(b*deltas).to_panel().add(ahat,axis='minor').to_frame()
    wealth = - series_outer((loglambdas.unstack('Year') - barloglambda_t).stack(),b).reorder_levels(['Year','HH']).sort()
    logxhat = nonwealth + wealth

    print (logx-logxhat).var()/logx.var()
    assert(np.linalg.norm((logx-logxhat).var()/logx.var()) < tol)

    return hhdf,pd.DataFrame(goodsdf,index=deltas.columns),logrealprices,logx,logxhat

def estimate_with_time_effects_differenced(x,z,phi=1e-4,tol=1e+1,return_a=False):
    """
    Given a pd.Dataframe x of expenditure data, and a pd.DataFrame z
    of household characteristics, estimate parameters 1/gamma, alphabar,
    delta, and lambdas.

    Dataframes are indexed by (Year,HH), with columns in x
    corresponding to expenditure types (no other variables should
    appear), and columns in z corresponding to characteristics.
    """

    # Monkey around to eliminate observations with missing values, (or
    # with negative values for expenditures)
    Rounds=x.index.levels[0]
    dlogx = difference_over_time(np.log((x+phi)))
    dz = difference_over_time(z)

    dlogx,dz = drop_missing([dlogx,dz])
    
    bardlogxit = dlogx.groupby(level='Year').mean()
    bardzt = dz.groupby(level='Year').mean()

    dz_demeaned = dz.to_panel().subtract(bardzt,axis='minor').to_frame()
    dlogx_demeaned = dlogx.to_panel().subtract(bardlogxit,axis='minor').to_frame()

    # Possible issues with missing data?
    assert(dlogx.shape==dlogx_demeaned.shape) 

    dhat,se_d=ols(dz_demeaned,dlogx_demeaned)

    a_it = bardlogxit

    cwT = dlogx_demeaned - dz_demeaned.dot(dhat.T)

    cwThat = svd_approximation(cwT,rnk=1)

    R2 = cwThat.var()/cwT.var()
    e = cwT - cwThat # Residuals

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    if cwThat.iloc[0,:].mean()>0:
        b=cwThat.iloc[0,:]
    else:
        b=-cwThat.iloc[0,:]

    # Normalize so that mean value of b elasticity is one, like an income elasticity
    b=b/b.mean()
    b=pd.Series(b,index=x.columns)

    dloglambdas=-cwThat.iloc[:,0]/b.iloc[0]

    deltas=dhat.T.divide(b)

    try:
        assert((abs(deltas.as_matrix())<np.inf).all())
    except AssertionError:
        warn('delta has some non-finite elements.')

    goodsdf=deltas.T.copy()

    dlogalphabars=a_it.mean(axis=0)/b
    
    # Now prices & average log lambdas per period

    atilde = a_it - a_it.mean(axis=0)

    dbarloglambda_t = -atilde.mean(axis=1)/b.mean()

    hhdf={z_l:dz_demeaned[z_l] for z_l in deltas.index} # Grab household characteristics
    hhdf.update({'lambdas':dloglambdas})
    hhdf = pd.DataFrame(hhdf)
    hhdf['lambdas'] = hhdf.to_panel()['lambdas'].add(dbarloglambda_t,axis=0).stack()

    atildetilde = atilde.subtract(series_outer(b,atilde.mean(axis=1)).T,axis=0)

    if np.any(b==1):
        warn("For goods with beta_i==1 prices aren't identified.")
    dlogrealprices = atildetilde.divide(1-b)

    goodsdf['beta']=b
    goodsdf['R2']=R2
    goodsdf['alphabar']=dlogalphabars
    goodsdf['phi']=phi

    # Now as a check put the Humpty-Dumpty of log expenditures back together
    ahat = ((1-b)*dlogrealprices + b*dlogalphabars - series_outer(dbarloglambda_t,b) + e.mean()).T.dropna()
    a_it=a_it.T
    assert(np.linalg.norm((a_it - ahat).dropna())<1e-1)

    nonwealth = dz_demeaned.dot(b*deltas).add(ahat[Rounds[-1]])
    wealth = - series_outer((dloglambdas.unstack('Year') - dbarloglambda_t).stack(),b).reorder_levels(['Year','HH']).sort()
    dlogxhat = nonwealth + wealth

    dlogx,dlogxhat = drop_missing([dlogx.T,dlogxhat.T])
    dlogx=dlogx.T
    dlogxhat=dlogxhat.T
    
    errvar=((dlogx-dlogxhat).var()/dlogx.var())
    assert(np.linalg.norm(errvar[np.isfinite(errvar)]) < tol)

    if return_a:
        return hhdf,pd.DataFrame(goodsdf,index=deltas.columns),dlogrealprices,dlogx,dlogxhat,a_it
    else:
        return hhdf,pd.DataFrame(goodsdf,index=deltas.columns),dlogrealprices,dlogx,dlogxhat


def predicted_expenditures(goodsdf,hhdf,prices):
    """Yields predicted expenditures.  The pd.DataFrame goodsdf should
    have columns 'gamma', 'delta', and 'alphabar'.  The hhdf should
    have a column 'lambdas', and other columns corresponding to
    columns in goodsdf['deltas'].  The price series should be in
    /levels/, and have prices by good, period.
    """
    try: # See if hhdf is stacked using a MultiIndex
        Rounds,HHs = tuple(hhdf.index.levels)
    except AttributeError: # Guess it's unstacked, with different columns for different years?
        HHs,Rounds=(hhdf.index,hhdf['lambdas'].columns)
        
    alpha=goodsdf['alphabar']

    lambdas=hhdf['lambdas'].unstack('Year')
    
    for v in set(goodsdf.columns).intersection(set(hhdf.columns)): # Names of household characteristics in z
        alpha += series_outer(hhdf[v],goodsdf[v])

    X=[]
    idx=[]
    for t in Rounds:
        for j in HHs:
            idx+=[(t,j)]
            try:
                gamma=goodsdf['gamma']
            except KeyError:
                gamma=1./goodsdf['beta']
            p=prices.loc[t,:]
            try:
                c=np.array(ves.frischdemands(np.exp(lambdas.loc[j,t]),p,np.exp(alpha.loc[(t,j)]),
                                             gamma.as_matrix(),goodsdf['phi'].as_matrix())).reshape(-1)
                X.append(c*p)
            except KeyError:
                X.append(np.nan*p)

    return pd.DataFrame(X,index=pd.MultiIndex.from_tuples(idx),columns=goodsdf.index).sort()

def diffs2levels(x,z,da,goodsdf,phi=1e-4):
    """Given levels of expenditures x, characteristics z, differenced a_it, and
       parameters from goodsdf return levels of log prices a_it (assuming
       initial prices all one) and levels lf log lambdas.
    """
    x=np.log(x+phi)
    Years=x.index.levels[0]
    da[Years[0]]=0
    a=da[Years].cumsum(axis=1)

    d=goodsdf[z.columns].copy()
    for c in d.columns:
        d[c]=goodsdf['beta']*d[c]

    loglambda=(x.reset_index(level='HH')[x.columns] - a.T)
    loglambda.index=x.index
    loglambda = (loglambda - z.dot(d.T)).mean(axis=1)

    return a,loglambda


def CRRA_adjustment(X,g):
    """
    Given an NTxn dataframe of expenditures X on n goods and a vector g=[1/gamma_i]_i,
    calculate the factor by which relative risk aversion differs from
    what it would if gamma_i=gamma for each household-period.
    """
    RRA=(X*g).dropna(axis=1,how='all').dropna(axis=0,how='any').sum(axis=1)
    ell=g/g
    RRA=g.mean()*(X*ell).dropna(axis=1,how='all').dropna(axis=0,how='any').sum(axis=1)/RRA

    return RRA

def bootstrap(df,lhsvar,rhsvar,reps=100,phi=1e-4):
    """
    Bootstrap reps draws of estimate using df.
    """
    Theta=[]
    Rounds=list(set(df.index.levels[0]))
    T=len(Rounds)
    hhs=list(set(df.index.levels[1]))
    N=len(hhs) # Number of households
    for i in range(reps):
        print "Draw %d" % i
        use=[]
        bootdf=pd.DataFrame(columns=df.columns,dtype='float64')
        k=0
        while k<N:
            hh=hhs[np.random.random_integers(0,N-1,1)] # Draw single household
            try:  
                obs=pd.DataFrame([df.loc[(t,hh)] for t in Rounds],dtype='float64')
                obs.rename(index={(t,hh):(t,k) for t in Rounds},inplace=True)
                k+=1
            except KeyError: # Not a full set of rounds for household?
                pass
            use.append(obs)
        bootdf=bootdf.append(use)
        bootdf.index = pd.MultiIndex.from_tuples(bootdf.index,names=['Year','HH'])
        t=estimate_with_time_effects_differenced(bootdf.loc[:,lhsvar],bootdf.loc[:,rhsvar],phi=1e-14)[1]['beta']

        Theta.append(t)

    return pd.DataFrame(Theta,index=range(reps))

def fake_hhsize(N,T,p0=1./3,p1=.9):
    """
    Generate time-varying household sizes for N households over T periods.

    The parameter p0 governs the initial (geometric) size distribution (smaller p0 means bigger sizes),
    while the parameter p1 governs the rate at which these household sizes evolve over time.
    """
    hhsize = np.random.geometric(p0,size=N).reshape((N,1)) # Initial household size
    X=np.c_[np.zeros((N,1)),np.arange(N).reshape((-1,1)),hhsize]
    for t in range(1,T):
        xt=hhsize + np.random.geometric(p1,size=(N,1))-np.random.geometric(p1,size=(N,1))
        hhsize=xt # xt
        xt=np.choose(xt>0,[1,xt]) # Minimum hhsize should be 1
        xt=np.c_[t*np.ones((N,1)),np.arange(N).reshape((-1,1)),xt]
        X=np.r_[X,xt]

    return X.astype(np.int32)

def fake_prices(K,T,sigma=1./4):
    """
    Generate time-varying log prices for K goods over T periods.  

    In levels this is a Martingale process with multiplicative log-normal innovations.

    First column is an indicator of the round.
    """
    p=np.zeros(K).reshape((1,-1)) # Initial prices; normalize to 1 (zero in logs)
    X=np.c_[np.zeros((1,1)),p]
    for t in range(1,T):
        pt=p + np.random.normal(-(sigma**2/2),sigma,size=(1,K))
        p=pt
        pt=np.c_[t*np.ones((1,1)),pt]
        X=np.r_[X,pt]

    return X

def fake_data(size=(2,100,4),delta=1.,alphasigma=0.1,direct=False):
    """Generate a fake dataset for for K goods for N households over T periods. 
 
    If direct, draw lambdas directly from a log normal distribution;
    otherwise draw /total expenditures/ from such a distribution, and
    compute corresponding lambda.
    """
    T,N,n=size

    d={}
    x = fake_hhsize(N,T)
    d['HHSize']=np.reshape(x[:,-1],(N,T)).reshape(-1)
    #d['HHSize']=(d['HHSize']-d['HHSize'].mean(axis=0)).reshape(-1)  # Take out per-period means
    d['Year']=x[:,0]
    d['HH']=x[:,1]

    # Let alphas be a function of hhsize; note that we keep this all in logs
    alphabar=np.random.random(size=n)  # Draws from uniform distribution
    alphabar=alphabar - np.mean(alphabar) # Normalization
    alphabar=alphabar.reshape((1,n)) 
    alpha=alphabar + delta*(d['HHSize']).reshape((-1,1)) + np.random.normal(-(alphasigma**2)/2.,alphasigma,size=(alphabar.shape))
    
    # Generate phis from a double geometric; make proportional to household size
    #phi=0.25*(np.random.geometric(2./3,size=(1,K))-np.random.geometric(2./K,size=(1,K)))*x[:,-1].reshape((-1,1))

    # Eliminate dependence of phi on household size.
    phi=1e-14*np.ones((n,))

    beta=1./np.arange(1.,n+1)/n*3
    beta=beta/np.mean(beta)  # Normalization
    gamma=1./beta

    logprices=fake_prices(n,T,sigma=1e-15)

    ystar=np.random.normal(size=(N,T))  # In logs
    #ystar=1+np.arange(N*T).reshape(N,T) # Use this to eliminate randomness in total expenditures
    X=[]
    Y=[]
    L=[]
    if not direct:
        for t in range(T):
            for j in range(N):
                y=np.exp(ystar[j,t])-sum(np.exp(logprices[t,1:])*phi)
                Y.append(y)
                L.append(ves.lambdavalue(y,np.exp(logprices[t,1:]),np.exp(alpha[j,:]),gamma,phi))
            print "Period %d" % t
        L=np.log(np.array(L).reshape((N,T),order='F'))
        d['y']=Y
    else:
        L=ystar
    
    L = L - L.mean(axis=0)
    
    for t in range(T):
        for j in range(N):
            X.append(ves.frischdemands(np.exp(L[j,t]),np.exp(logprices[t,1:]),np.exp(alpha[j,:]),gamma,phi))

    X=np.array(X)

    for k in range(n):
        d['x%d' % k]=X[:,k]

    df=pd.DataFrame(d)
    df.set_index(['Year','HH'],inplace=True,drop=True)
    
    truegoodsdf=pd.DataFrame({'gamma':gamma,'phi':phi,'alphabar':alphabar[0]},index=['x%d' %k for k in range(n)])
    truegoodsdf['delta_0']=delta
    
    hhdf={'lambda':L.reshape(-1,order='F')}
    hhdf.update({'alpha_%d' % i:alpha[:,i] for i in range(n)})
    truehhdf=pd.DataFrame(hhdf,index=df.index).unstack('Year')
    logprices=pd.DataFrame(logprices[:,1:],columns=truegoodsdf.index,index=range(T))

    return df,{'goods':truegoodsdf,'hh':truehhdf,'prices':logprices}

def test_data():
    T,N,n=(2,3,2)
    delta=0.

    d={}
    x = fake_hhsize(N,T)
    d['HHSize']=np.ones((N,T))
    d['HHSize']=(d['HHSize']-d['HHSize'].mean(axis=0)).reshape(-1)  # Take out per-period means
    d['Year']=x[:,0]
    d['HH']=x[:,1]

    # Let alphas be a function of hhsize; note that we keep this all in logs
    alphabar=np.zeros(n)
    alphabar=alphabar - np.mean(alphabar) # Normalization
    alphabar=alphabar.reshape((1,n)) 
    alpha=alphabar + delta*(d['HHSize']).reshape((-1,1))
    
    # Eliminate dependence of phi on household size.
    phi=1e-14*np.ones((n,))

    beta=np.array([1,2])
    beta=beta/np.mean(beta)  # Normalization
    gamma=1./beta

    logprices=np.zeros((T,n))

    L = np.array([[-1,0,1],[-1,0,1]]).T
    L = L - L.mean(axis=0)

    X=[]
    for t in range(T):
        for j in range(N):
            X.append(ves.frischdemands(np.exp(L[j,t]),np.exp(logprices[t,:]),np.exp(alpha[j,:]),gamma,phi))

    X=np.array(X)

    for k in range(n):
        d['x%d' % k]=X[:,k]

    df=pd.DataFrame(d)
    df.set_index(['Year','HH'],inplace=True,drop=True)
    
    truegoodsdf=pd.DataFrame({'gamma':gamma,'phi':phi,'alphabar':alphabar[0]},index=['x%d' %k for k in range(n)])
    truegoodsdf['delta_0']=delta
    
    hhdf={'lambda':L.reshape(-1,order='F')}
    hhdf.update({'alpha_%d' % i:alpha[:,i] for i in range(n)})
    truehhdf=pd.DataFrame(hhdf,index=df.index).unstack('Year')
    logprices=pd.DataFrame(logprices,columns=truegoodsdf.index,index=range(T))

    return df,{'goods':truegoodsdf,'hh':truehhdf,'prices':logprices}

def assert_approximate_equality(msg,thing1,thing2,tol=1e-2):
    print msg,
    diff=thing1-thing2
    err=norm(diff,np.inf)
    assert(err<tol)
    print "Yes.  Error is %6.4f." % err
    
def test2(n=2,N=3,T=2,delta=1.,alphasigma=1e-16):
    """
    Small, (almost) deterministic test of simplified estimation scheme using time effects.  Add household characteristics.
    """
    df,truevalues=fake_data(size=(T,N,n),delta=delta,alphasigma=alphasigma,direct=True)
    #df,truevalues=test_data()

    #hhdf,goodsdf,prices,logxhat = estimate_with_time_effects_2step(df.loc[:,["x%d" % i for i in range(n)]],df.loc[:,['HHSize']],phi=1e-14)
    hhdf,goodsdf,prices,logx,logxhat = estimate_with_time_effects(df[["x%d" % i for i in range(n)]],df[['HHSize']],phi=1e-14)

    exphat = predicted_expenditures(goodsdf,hhdf,np.exp(prices))
    mse = ((df-exphat)**2).mean().dropna()

    print "MSE:"
    print mse

    try:
        assert(mse.max()<1e-4) # bound on mse
    except AssertionError:
    
        assert_approximate_equality("betas ok?",1./truevalues['goods']['gamma'].as_matrix(),goodsdf['beta'].as_matrix())

        #assert_approximate_equality("deltas ok?",truevalues['goods']['delta_0'],goodsdf['HHSize'],tol=1./np.sqrt(N))

        #assert_approximate_equality("alphahat ok?",truevalues['goods']['alphabar'],goodsdf['alphabar'])

        l0=truevalues['hh']['lambda']
        lhat=hhdf.unstack('Year')['lambdas']
        assert_approximate_equality("Change in demeaned lambdas ok?",
                                    (l0-l0.mean()).T.diff().as_matrix()[1,:],
                                    (lhat-lhat.mean()).T.diff().as_matrix()[1,:])

        # These may not be guaranteed--lhat depends on aggregate prices in a way that l0 doesn't?
        assert_approximate_equality("Change in lambdas ok?",l0.T.diff().as_matrix()[1,:],
                                    lhat.T.diff().as_matrix()[1,:])


        assert_approximate_equality("lambdas ok?",l0.as_matrix(),lhat.as_matrix())

        alphahat = series_outer(hhdf['HHSize'],goodsdf['HHSize']).add(goodsdf['alphabar']).unstack('Year')
        alpha0 = truevalues['hh'][['alpha_%d' % i for i in range(n)]]
        assert_approximate_equality("alphas ok?",alpha0.as_matrix(),alphahat.as_matrix(),tol=1./np.sqrt(N))




    return mse,goodsdf,prices

def test_differenced(n=2,N=3,T=2,delta=1.,alphasigma=1e-16):
    """
    Small, (almost) deterministic test of simplified estimation scheme using time effects.  Add household characteristics.
    """
    df,truevalues=fake_data(size=(T,N,n),delta=delta,alphasigma=alphasigma,direct=True)
    #df,truevalues=test_data()

    #hhdf,goodsdf,prices,logxhat = estimate_with_time_effects_2step(df.loc[:,["x%d" % i for i in range(n)]],df.loc[:,['HHSize']],phi=1e-14)
    hhdf,goodsdf,prices,logx,logxhat = estimate_with_time_effects_differenced(df[["x%d" % i for i in range(n)]],df[['HHSize']],phi=1e-14)

    exphat = predicted_expenditures(goodsdf,hhdf,np.exp(prices))
    mse = ((df-exphat)**2).mean().dropna()

    print "MSE:"
    print mse

    try:
        assert(mse.max()<1e-4) # bound on mse
    except AssertionError:
    
        assert_approximate_equality("betas ok?",1./truevalues['goods']['gamma'].as_matrix(),goodsdf['beta'].as_matrix())

        #assert_approximate_equality("deltas ok?",truevalues['goods']['delta_0'],goodsdf['HHSize'],tol=1./np.sqrt(N))

        #assert_approximate_equality("alphahat ok?",truevalues['goods']['alphabar'],goodsdf['alphabar'])

        l0=truevalues['hh']['lambda']
        lhat=hhdf.unstack('Year')['lambdas']
        assert_approximate_equality("Change in demeaned lambdas ok?",
                                    (l0-l0.mean()).T.diff().as_matrix()[1,:],
                                    (lhat-lhat.mean()).as_matrix())

        # These may not be guaranteed--lhat depends on aggregate prices in a way that l0 doesn't?
        assert_approximate_equality("Change in lambdas ok?",l0.T.diff().as_matrix()[1,:],
                                    lhat.T.diff().as_matrix()[1,:])


        assert_approximate_equality("lambdas ok?",l0.as_matrix(),lhat.as_matrix())

        alphahat = series_outer(hhdf['HHSize'],goodsdf['HHSize']).add(goodsdf['alphabar']).unstack('Year')
        alpha0 = truevalues['hh'][['alpha_%d' % i for i in range(n)]]
        assert_approximate_equality("alphas ok?",alpha0.as_matrix(),alphahat.as_matrix(),tol=1./np.sqrt(N))

    return mse,goodsdf,prices
                  
        
if __name__=='__main__':
    #test0() # Passes
    #test0(N=1500) # Passes (reasonable # of households)
    #test0(N=100,n=25) # Passes
    #test0(N=1800,n=25) # Passes
    #test1(N=1000) # Fails (test of adding hh characteristics)
    mse,goodsdf,prices=test2(n=29,N=300,delta=1.,alphasigma=1e-16)
    #mse,goodsdf,prices=test_differenced(n=29,N=300,delta=1.,alphasigma=1e-16)

    
    #uganda.main(datadir='../Data/Uganda/')
