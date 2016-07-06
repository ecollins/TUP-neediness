import numpy as np
from oct2py import octave
octave.addpath('../utils/IncPACK')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

def svd_rank1_approximation_with_missing_data(x,return_usv=False,VERBOSE=True): 
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    m,n=x.shape

    if n<m: 
        x=x.dropna(how='all')
        x=x.T
        TRANSPOSE=True
    else:
        x=x.dropna(how='all',axis=1)
        TRANSPOSE=False

    u,s,v=svd_missing(x.as_matrix())
    if VERBOSE:
        print "Estimated singular values: ",
        print s

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns)

    if not TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

def get_loglambdas(e,tol=1e-5,TEST=False):
    """
    Use singular-value decomposition to compute loglambdas and price elasticities.  
    
    Input e is the residual from a regression of log expenditures purged
    of the effects of prices and household characteristics.   The residuals
    should be arranged as a matrix, with columns corresponding to goods. 
    """ 

    assert(e.shape[0]>e.shape[1]) # Fewer goods than observations

    chat = svd_rank1_approximation_with_missing_data(e,VERBOSE=False) #~ New version doesn't have a tolerance,tol=tol)

    R2 = chat.var()/e.var()

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    if chat.iloc[0,:].mean()>0:
        b=chat.iloc[0,:]
    else:
        b=-chat.iloc[0,:]

    # If no numeraire, then normalize so var(log lambda)=1.
    loglambdas=-chat.iloc[:,0]

    bphi=b*loglambdas.std()

    bphi=pd.Series(bphi,index=e.columns)

    loglambdas=loglambdas/loglambdas.std()

    if TEST:
        foo=-np.outer(bphi,loglambdas).T
        print "blogL norm: %f" % np.linalg.norm(foo-chat)
        print "R2=", R2

    return bphi,loglambdas

def bootstrap_bphi_se(e,svdtol=1e-5,setol=1e-3):
    """
    Bootstrap svd estimation of bphi in get_loglambdas.
    """
    B=[]
    for i in range(10): # Minimum number of bootstraps
      print i
      b,l=get_loglambdas(e.iloc[np.random.random_integers(0,e.shape[0]-1,size=e.shape[0]),:],tol=svdtol,TEST=False)
      B.append(b.as_matrix().tolist())

    selast=0
    B=np.array(B)
    se=B.std(axis=0)
    while np.linalg.norm(se-selast)>setol:
      print i, np.linalg.norm(se-selast), se
      selast=se
      b,l=get_loglambdas(e.iloc[np.random.random_integers(0,e.shape[0]-1,size=e.shape[0]),:],tol=svdtol,TEST=False)
      B=np.r_[B,b.as_matrix().reshape((1,-1))]
      se=B.std(axis=0)
      i+=1

    return se

import numpy as np
from oct2py import octave
octave.addpath('../utils/IncPACK')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

import numpy as np
from oct2py import octave
octave.addpath('../utils/IncPACK')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

def svd_rank1_approximation_with_missing_data(x,return_usv=False,VERBOSE=True): 
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    m,n=x.shape

    if n<m: 
        x=x.dropna(how='all')
        x=x.T
        TRANSPOSE=True
    else:
        x=x.dropna(how='all',axis=1)
        TRANSPOSE=False

    u,s,v=svd_missing(x.as_matrix())
    if VERBOSE:
        print "Estimated singular values: ",
        print s

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns)

    if not TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

percent_missing=0.2
import numpy as np
import pandas as pd
import numpy as np
from oct2py import octave
octave.addpath('../utils/IncPACK')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

def svd_rank1_approximation_with_missing_data(x,return_usv=False,VERBOSE=True): 
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    m,n=x.shape

    if n<m: 
        x=x.dropna(how='all')
        x=x.T
        TRANSPOSE=True
    else:
        x=x.dropna(how='all',axis=1)
        TRANSPOSE=False

    u,s,v=svd_missing(x.as_matrix())
    if VERBOSE:
        print "Estimated singular values: ",
        print s

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns)

    if not TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

(n,m)=(50,5000)
a=np.random.normal(size=(n,1))
b=np.random.normal(size=(1,m))
e=np.random.normal(size=(n,m))*1e-1

X0=np.outer(a,b)+e

X=X0.copy()
X[np.random.random_sample(X.shape)<percent_missing]=np.nan

X0=pd.DataFrame(X0).T
X=pd.DataFrame(X).T

Xhat=svd_rank1_approximation_with_missing_data(X,VERBOSE=False)

print "Proportion missing %g and correlation %5.4f" % (percent_missing, pd.concat([X.stack(dropna=False),Xhat.stack()],axis=1).corr().iloc[0,1])

import numpy as np



import numpy as np
from oct2py import octave
octave.addpath('../utils/IncPACK')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

def svd_rank1_approximation_with_missing_data(x,return_usv=False,VERBOSE=True): 
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    m,n=x.shape

    if n<m: 
        x=x.dropna(how='all')
        x=x.T
        TRANSPOSE=True
    else:
        x=x.dropna(how='all',axis=1)
        TRANSPOSE=False

    u,s,v=svd_missing(x.as_matrix())
    if VERBOSE:
        print "Estimated singular values: ",
        print s

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns)

    if not TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

y,truth=artificial_data(T=1,N=1000,n=12,sigma_e=1e-1)
#y,truth=artificial_data(T=2,N=20,n=6,sigma_e=1e-8)
beta,L,dz,p=truth

numeraire='x0'

b0,ce0,d0=estimate_bdce_with_missing_values(y,np.log(dz),return_v=False)
myce0=ce0.copy()
cehat=svd_rank1_approximation_with_missing_data(myce0)

rho=pd.concat([ce0.stack(dropna=False),cehat.stack()],axis=1).corr().iloc[0,1]

print "Norm of error in approximation of CE: %f; Correlation %f." % (df_norm(cehat,ce0)/df_norm(ce0),rho)
