def df_to_orgtbl(df,tdf=None,sedf=None,float_fmt='%5.3f'):
    """
    Print pd.DataFrame in format which forms an org-table.
    Note that headers for code block should include ":results table raw".
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)

    if (tdf is None) and (sedf is None):
        return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')
    elif not (tdf is None) and (sedf is None):
        s = '|  |'+'|   '.join(df.columns)+'\t|\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns:
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n'

        return s
    elif not sedf is None: # Print standard errors on alternate rows
        s = '|  |'+'|   '.join(df.columns)+'  |\n|-\n'
        tdf = df.div(sedf)
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now standard errors
                s+='  '
                try:
                    se='$(' + float_fmt % sedf.loc[i,j] + ')$' 
                except KeyError: se=''
                entry='| '+se+'  '
                s+=entry 
            s+='|\n'
        return s
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
import pandas as pd
from scipy import sparse

def ols(x,y,return_se=True,return_v=False):

    x=pd.DataFrame(x) # Deal with possibility that x & y are series.
    y=pd.DataFrame(y)
    k=x.shape[1]

    # Drop any observations that have missing data in *either* x or y.
    x,y = drop_missing([x,y]) 
    N,n=y.shape
    
    b=np.linalg.lstsq(x,y)[0]

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

import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
import pylab as pl
from numpy import linalg
import TUP

import sys
sys.path.append('./estimation')

year = ("_b", "_m")

#~ This uses the disaggregated carbohydrate and meat expenditure categories collected at midline rather than aggregating over them.
#~ Previously we've added them up as "cereals", since this sum has a very similar distribution to cereals at baseline
#~ and we're relying on baseline controls.
#~ Instead, this treats cereals as the baseline control for each category separately


try: df=pd.read_pickle('../data/modified/ss-goods_b_m.df') # Created by TUP.process_data
except IOError: 
    D  = TUP.full_data(balance = ["Base", "Mid"])

    carbohydrates = ["c_maize", "c_sorghum", "c_millet", "c_potato", "c_sweetpotato", "c_rice", "c_bread"]
    for carb in carbohydrates: D[carb+"_b"] = D["c_cereals_b"]
    #~ Same thing for animals
    #~ D["c_poultry_b"] = D["c_meat_b"]
    #~ D["c_meat_m"] = D["c_livestock_m"]

    C, HH, T = TUP.consumption_data(D, how="wide", goods_from_years=["Base", "Mid"])

    Midline_Goods = ['c_beans', 'c_cosmetics', 'c_egg', 'c_fish', 'c_fruit', 'c_fuel', \
                     'c_meat',  'c_oil', 'c_salt', 'c_soap', 'c_sugar', 'c_transport', \
                     'c_vegetables'] + carbohydrates  #~ 'c_poultry','c_cereals', (See comments about carbohydrates for why this is commented out)
    use_goods = [item+"_b" for item in Midline_Goods] + [item+"_m" for item in Midline_Goods]

    C = C[use_goods]
    df = TUP.process_data(C, HH, T, year = year, save=True) #~ Process_data() builds consumption data if not given as an argument

df['Constant']=1
df["CTL"] = 1-df["TUP"] #~ Code the cash group as controls since they're not in the midline analysis

explist=[s[2:] for s in Midline_Goods if (s+year[0] in df) and (s+year[1] in df)]
#explist=[s[2:-2] for s in df.columns[[s.startswith('c_') and s.endswith(year[1]) for s in df.columns]]]
#explist = ['beans','cereals','cosmetics','egg','fish','fruit','fuel','meat','oil','salt','soap','sugar','transport','vegetables']
#explist = df.columns[[s.startswith('c_') and s.endswith(year[1]) for s in df.columns]]

#~ df[explist]=~np.isnan(df[explist]) # Do this  just to  see if non-zero expenditure reported

df = df.rename(columns= lambda x: x[:-2] if x.endswith(year[1]) else x)

bothdf=[]
xvars=['hh_size','child_total','Loc']
for x in explist:
    if 'c_'+x+r'_b' not in df:
        #~ When you take out the baseline controls in favor of repeated cross-sections, this is where to start...
        print(x+" has no baseline data or had too few non-zero responses at baseline. Skipping.")
        continue
    ydf=pd.DataFrame(df[['c_'+x]].rename(columns={'c_'+x:x.capitalize()}).stack())
    rdict=dict(zip(xvars+['c_'+x+r'_b'],["%s_%s" % (s,x.capitalize()) for s in xvars]+['Baseline_%s' % x.capitalize()]))

    xdf=pd.DataFrame(df[xvars+['c_'+x+r'_b']]) # Add baseline values of expenditure variables.
    xdf.index=pd.MultiIndex.from_tuples([(i,x.capitalize()) for i in xdf.index])
    locations=pd.get_dummies(xdf['Loc'],prefix='Loc_%s' % x.capitalize())
    del xdf['Loc']
    xdf.rename(columns=rdict,inplace=True)
    xdf=xdf.join(locations)
    xdf.replace(to_replace=np.NaN,value=0,inplace=True)

    # Add row to restrict location dummies to sum to zero
    ydf=pd.concat([ydf,pd.DataFrame([0],index=[(0,x.capitalize())])])
    xdf=pd.concat([xdf,pd.DataFrame([s.startswith('Loc_')+0. for s in xdf.columns],index=xdf.columns,columns=[(0,x.capitalize())]).T]) 

    xdf[0]=ydf
    xdf.dropna(how='any',inplace=True) # Drops any obs. with missing expenditures
    bothdf.append(xdf)

#~ Are this fillna() call and the xdf.replace call above a problem? It seems necessary for the block-diagonal ols function
#~ we're using, but aren't we coding zeros as missing and calculating residuals for only those positive consumption? Wouldn't
#~ replacing them to zero insert some non-zero residual for households that never consume a given good?
#~ And isn't this the motivation behind svd_missing?
mydf=pd.concat(bothdf).fillna(value=0)

X=mydf.iloc[:,1:]

y=mydf[[0]]

x=np.exp(y.unstack().iloc[1:,:]) # Expenditures (in levels)
xshares=x.divide(x.sum(axis=1),axis=0).fillna(value=0).mean() # Expenditure shares (taking missing as zero)
xshares.index=xshares.index.droplevel(0)

b,se=ols(X,y)

## betahat=b[['Constant_%s' % s.capitalize() for s in explist]]
## betahat.rename(columns=dict(zip(betahat.columns,[s.capitalize() for s in explist])),inplace=True)

e=y-X.dot(b.T)

e.rename(columns={0:'Resid'},inplace=True)
e.index.names=['HH','Good']

testdf=pd.merge(df[['TUP','CTL']].reset_index(),e.reset_index(),how='outer',on=['HH'])
testdf.set_index(['HH','Good'],inplace=True)

TUP=testdf['TUP'].mul(testdf['Resid']).dropna().unstack()
CTL=testdf['CTL'].mul(testdf['Resid']).dropna().unstack()

e=(e-e.mean()).unstack()

# Test of significant differences between treatment and control:
# Weighting matrix:
A=np.matrix((TUP-CTL).cov().as_matrix()).I
g=np.matrix((TUP-CTL).mean())
J=e.shape[0]*g*A*g.T # Chi2 statistic

p=1-chi2.cdf(J,e.shape[1])

chi2test="Chi2 test: %f (%f)" % (J,p)

N=pd.Series([d.shape[0]-1 for d in bothdf],index=[d.index.levels[1][0] for d in bothdf])

resultdf=pd.DataFrame({'TUP':TUP.mean(),'CTL':CTL.mean(),'$N$':N})
sedf=pd.DataFrame({'TUP':TUP.std()/np.sqrt(resultdf['$N$']),'CTL':CTL.std()/np.sqrt(resultdf['$N$'])})
resultdf['Diff.']=resultdf['TUP']-resultdf['CTL']
sedf['Diff.']=np.sqrt((sedf['TUP']**2) + (sedf['CTL']**2))

# Use svd (with missing data) to construct beta & log lambda

myb,myl = get_loglambdas(e,TEST=True)

myb.index=myb.index.droplevel(0)

# Normalize log lambdas
l=myl/myl.std()

# Normalize so weighted Frisch elasticities sum to one
# resultdf['beta_i']=myb/(myb.dot(xshares.T)) # Actually, prefer ols results for beta, see below...

# Imputed elements of e means that we get incorrect values for betas.  
# Go back and respect missing values; also use this opportunity to calculate standard errors.
B=[]
SEb=[]
for i in e:
    olsb,seb = ols(-l,e[i])
    SEb.append(seb.iloc[0,0])
    B.append(olsb.iloc[0,0])

B=np.array(B)

sedf['beta_i']=np.array(SEb) #/(B.dot(xshares.T))
resultdf['beta_i']=B #/(B.dot(xshares.T))

lresult=pd.DataFrame({'TUP':(df['TUP']*l).mean(),'CTL':(df['CTL']*l).mean(),'Diff.':(-(df['CTL']-df['TUP'])*l).mean()},index=['lambda'])
lse=pd.DataFrame({'TUP':(df['TUP']*l.std())/np.sqrt(l.shape[0]),'CTL':(df['CTL']*l.std())/np.sqrt(l.shape[0])})

resultdf=pd.concat([resultdf,lresult])

tstats=pd.DataFrame({'TUP':resultdf['TUP']/sedf['TUP'],
                     'CTL':resultdf['CTL']/sedf['CTL'],
                     'Diff.':resultdf['Diff.']/sedf['Diff.'],
                     'beta_i':resultdf['beta_i']/sedf['beta_i'],
})

tab=df_to_orgtbl(resultdf,sedf=sedf,tdf=tstats)

# Prepare plot of lambdas by treatment
pl.clf()
lambdas=pd.DataFrame({'lambda':l,'TUP':df['TUP'],'CTL':df['CTL']})
lambdas.to_pickle('/tmp/ss-loglambdas.df')

lambdas.dropna().query('TUP==1')['lambda'].plot(kind='kde')
lambdas.dropna().query('TUP==0')['lambda'].plot(kind='kde',linestyle='--')

pl.legend(('TUP','CTL'))
pl.xlabel('$\log\lambda^j_t$',fontsize=18)
pl.savefig('./figures/loglambda_distribution_by_treatment.png')

# Variance
lambdas.sort('lambda',inplace=True)
l0=lambdas.query('TUP==0')['lambda']
l1=lambdas.query('TUP==1')['lambda']

v0=l0.var()
v1=l1.var()

# Bootstrap distribution of variances
V=[]
M=[]
P=[]
for i in range(5000): #00):
    s0=np.random.randint(len(l0),size=len(l0)) # Draw sample size equal to TUP group
    s0.sort()
    s1=np.random.randint(len(l1),size=len(l0))
    s1.sort()

    V.append(l0.iloc[s0].var() > l1.iloc[s1].var())
    M.append(l0.iloc[s0].mean() > l1.iloc[s1].mean())
    P.append((l0.iloc[s0].as_matrix() > l1.iloc[s1].as_matrix()).mean())

print "Probability TUP variance smaller: %f" % np.array(V).mean()
print "Probability TUP mean smaller: %f" % np.array(M).mean()
print "Probability of complete stochastic dominance: %f" % (np.array(P)==1.).mean() # Not sure how to interpret this statistic--must be something better?

# Now back out prices:
ploc=b.filter(regex = '^Loc_.+')
ploc.rename(columns = lambda c: tuple(c[4:].split('_')),inplace=True)
p=ploc.T.unstack()

pse=se.T.filter(regex = '^Loc_.+')
pse.rename(columns = lambda c: tuple(c[4:].split('_')),inplace=True)
pse=pse.T.unstack()

# Drop "inferior" goods (inconsistent with theory)
beta=resultdf.query('beta_i>0')['beta_i']
A=np.matrix(np.eye(len(beta)) - np.diag(beta)).I

# Prices across locations
pbar=np.exp(A.dot(p.loc[beta.index]))
pse=pse.loc[beta.index] # Drop inferior goods

pdf=pd.DataFrame(pbar.A,index=beta.index,columns=p.columns.droplevel())
pdf.rename(columns=lambda c: " ".join([x.capitalize() for x in c.split(" ")]),inplace=True)
pdf.to_pickle('/tmp/prices.df')

prices=df_to_orgtbl(pdf,sedf=pse,tdf=pdf/sedf,float_fmt="%4.2f")

#~ As of right now, this is *NOT* producing standard errors for
#~ $\overline{\log\lambda}^g$, so I don't know what the stars should look like on that column.
#~ Note that the central effect size of the paper is now different when we disaggregate the cereals
return tab

def df_to_orgtbl(df,tdf=None,sedf=None,float_fmt='%5.3f'):
    """
    Print pd.DataFrame in format which forms an org-table.
    Note that headers for code block should include ":results table raw".
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)

    if (tdf is None) and (sedf is None):
        return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')
    elif not (tdf is None) and (sedf is None):
        s = '|  |'+'|   '.join(df.columns)+'\t|\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns:
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n'

        return s
    elif not sedf is None: # Print standard errors on alternate rows
        s = '|  |'+'|   '.join(df.columns)+'  |\n|-\n'
        tdf = df.div(sedf)
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now standard errors
                s+='  '
                try:
                    se='$(' + float_fmt % sedf.loc[i,j] + ')$' 
                except KeyError: se=''
                entry='| '+se+'  '
                s+=entry 
            s+='|\n'
        return s
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
import pandas as pd
from scipy import sparse

def ols(x,y,return_se=True,return_v=False):

    x=pd.DataFrame(x) # Deal with possibility that x & y are series.
    y=pd.DataFrame(y)
    k=x.shape[1]

    # Drop any observations that have missing data in *either* x or y.
    x,y = drop_missing([x,y]) 
    N,n=y.shape
    
    b=np.linalg.lstsq(x,y)[0]

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

import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
import pylab as pl
from numpy import linalg
import TUP

import sys
sys.path.append('./estimation')

year = ("_b", "_m")

#~ This uses the disaggregated carbohydrate and meat expenditure categories collected at midline rather than aggregating over them.
#~ Previously we've added them up as "cereals", since this sum has a very similar distribution to cereals at baseline
#~ and we're relying on baseline controls.
#~ Instead, this treats cereals as the baseline control for each category separately


try: df=pd.read_pickle('../data/modified/ss-goods_b_m.df') # Created by TUP.process_data
except IOError: 
    D  = TUP.full_data(balance = ["Base", "Mid"])

    carbohydrates = ["c_maize", "c_sorghum", "c_millet", "c_potato", "c_sweetpotato", "c_rice", "c_bread"]
    for carb in carbohydrates: D[carb+"_b"] = D["c_cereals_b"]
    #~ Same thing for animals
    #~ D["c_poultry_b"] = D["c_meat_b"]
    #~ D["c_meat_m"] = D["c_livestock_m"]

    C, HH, T = TUP.consumption_data(D, how="wide", goods_from_years=["Base", "Mid"])

    Midline_Goods = ['c_beans', 'c_cosmetics', 'c_egg', 'c_fish', 'c_fruit', 'c_fuel', \
                     'c_meat',  'c_oil', 'c_salt', 'c_soap', 'c_sugar', 'c_transport', \
                     'c_vegetables'] + carbohydrates  #~ 'c_poultry','c_cereals', (See comments about carbohydrates for why this is commented out)
    use_goods = [item+"_b" for item in Midline_Goods] + [item+"_m" for item in Midline_Goods]

    C = C[use_goods]
    df = TUP.process_data(C, HH, T, year = year, save=True) #~ Process_data() builds consumption data if not given as an argument

df['Constant']=1
df["CTL"] = 1-df["TUP"] #~ Code the cash group as controls since they're not in the midline analysis

explist=[s[2:] for s in Midline_Goods if (s+year[0] in df) and (s+year[1] in df)]
#explist=[s[2:-2] for s in df.columns[[s.startswith('c_') and s.endswith(year[1]) for s in df.columns]]]
#explist = ['beans','cereals','cosmetics','egg','fish','fruit','fuel','meat','oil','salt','soap','sugar','transport','vegetables']
#explist = df.columns[[s.startswith('c_') and s.endswith(year[1]) for s in df.columns]]

#~ df[explist]=~np.isnan(df[explist]) # Do this  just to  see if non-zero expenditure reported

df = df.rename(columns= lambda x: x[:-2] if x.endswith(year[1]) else x)

bothdf=[]
xvars=['hh_size','child_total','Loc']
for x in explist:
    if 'c_'+x+r'_b' not in df:
        #~ When you take out the baseline controls in favor of repeated cross-sections, this is where to start...
        print(x+" has no baseline data or had too few non-zero responses at baseline. Skipping.")
        continue
    ydf=pd.DataFrame(df[['c_'+x]].rename(columns={'c_'+x:x.capitalize()}).stack())
    rdict=dict(zip(xvars+['c_'+x+r'_b'],["%s_%s" % (s,x.capitalize()) for s in xvars]+['Baseline_%s' % x.capitalize()]))

    xdf=pd.DataFrame(df[xvars+['c_'+x+r'_b']]) # Add baseline values of expenditure variables.
    xdf.index=pd.MultiIndex.from_tuples([(i,x.capitalize()) for i in xdf.index])
    locations=pd.get_dummies(xdf['Loc'],prefix='Loc_%s' % x.capitalize())
    del xdf['Loc']
    xdf.rename(columns=rdict,inplace=True)
    xdf=xdf.join(locations)
    xdf.replace(to_replace=np.NaN,value=0,inplace=True)

    # Add row to restrict location dummies to sum to zero
    ydf=pd.concat([ydf,pd.DataFrame([0],index=[(0,x.capitalize())])])
    xdf=pd.concat([xdf,pd.DataFrame([s.startswith('Loc_')+0. for s in xdf.columns],index=xdf.columns,columns=[(0,x.capitalize())]).T]) 

    xdf[0]=ydf
    xdf.dropna(how='any',inplace=True) # Drops any obs. with missing expenditures
    bothdf.append(xdf)

#~ Are this fillna() call and the xdf.replace call above a problem? It seems necessary for the block-diagonal ols function
#~ we're using, but aren't we coding zeros as missing and calculating residuals for only those positive consumption? Wouldn't
#~ replacing them to zero insert some non-zero residual for households that never consume a given good?
#~ And isn't this the motivation behind svd_missing?
mydf=pd.concat(bothdf).fillna(value=0)

X=mydf.iloc[:,1:]

y=mydf[[0]]

x=np.exp(y.unstack().iloc[1:,:]) # Expenditures (in levels)
xshares=x.divide(x.sum(axis=1),axis=0).fillna(value=0).mean() # Expenditure shares (taking missing as zero)
xshares.index=xshares.index.droplevel(0)

b,se=ols(X,y)

## betahat=b[['Constant_%s' % s.capitalize() for s in explist]]
## betahat.rename(columns=dict(zip(betahat.columns,[s.capitalize() for s in explist])),inplace=True)

e=y-X.dot(b.T)

e.rename(columns={0:'Resid'},inplace=True)
e.index.names=['HH','Good']

testdf=pd.merge(df[['TUP','CTL']].reset_index(),e.reset_index(),how='outer',on=['HH'])
testdf.set_index(['HH','Good'],inplace=True)

TUP=testdf['TUP'].mul(testdf['Resid']).dropna().unstack()
CTL=testdf['CTL'].mul(testdf['Resid']).dropna().unstack()

e=(e-e.mean()).unstack()

# Test of significant differences between treatment and control:
# Weighting matrix:
A=np.matrix((TUP-CTL).cov().as_matrix()).I
g=np.matrix((TUP-CTL).mean())
J=e.shape[0]*g*A*g.T # Chi2 statistic

p=1-chi2.cdf(J,e.shape[1])

chi2test="Chi2 test: %f (%f)" % (J,p)

N=pd.Series([d.shape[0]-1 for d in bothdf],index=[d.index.levels[1][0] for d in bothdf])

resultdf=pd.DataFrame({'TUP':TUP.mean(),'CTL':CTL.mean(),'$N$':N})
sedf=pd.DataFrame({'TUP':TUP.std()/np.sqrt(resultdf['$N$']),'CTL':CTL.std()/np.sqrt(resultdf['$N$'])})
resultdf['Diff.']=resultdf['TUP']-resultdf['CTL']
sedf['Diff.']=np.sqrt((sedf['TUP']**2) + (sedf['CTL']**2))

# Use svd (with missing data) to construct beta & log lambda

myb,myl = get_loglambdas(e,TEST=True)

myb.index=myb.index.droplevel(0)

# Normalize log lambdas
l=myl/myl.std()

# Normalize so weighted Frisch elasticities sum to one
# resultdf['beta_i']=myb/(myb.dot(xshares.T)) # Actually, prefer ols results for beta, see below...

# Imputed elements of e means that we get incorrect values for betas.  
# Go back and respect missing values; also use this opportunity to calculate standard errors.
B=[]
SEb=[]
for i in e:
    olsb,seb = ols(-l,e[i])
    SEb.append(seb.iloc[0,0])
    B.append(olsb.iloc[0,0])

B=np.array(B)

sedf['beta_i']=np.array(SEb) #/(B.dot(xshares.T))
resultdf['beta_i']=B #/(B.dot(xshares.T))

lresult=pd.DataFrame({'TUP':(df['TUP']*l).mean(),'CTL':(df['CTL']*l).mean(),'Diff.':(-(df['CTL']-df['TUP'])*l).mean()},index=['lambda'])
lse=pd.DataFrame({'TUP':(df['TUP']*l.std())/np.sqrt(l.shape[0]),'CTL':(df['CTL']*l.std())/np.sqrt(l.shape[0])})

resultdf=pd.concat([resultdf,lresult])

tstats=pd.DataFrame({'TUP':resultdf['TUP']/sedf['TUP'],
                     'CTL':resultdf['CTL']/sedf['CTL'],
                     'Diff.':resultdf['Diff.']/sedf['Diff.'],
                     'beta_i':resultdf['beta_i']/sedf['beta_i'],
})

tab=df_to_orgtbl(resultdf,sedf=sedf,tdf=tstats)

# Prepare plot of lambdas by treatment
pl.clf()
lambdas=pd.DataFrame({'lambda':l,'TUP':df['TUP'],'CTL':df['CTL']})
lambdas.to_pickle('/tmp/ss-loglambdas.df')

lambdas.dropna().query('TUP==1')['lambda'].plot(kind='kde')
lambdas.dropna().query('TUP==0')['lambda'].plot(kind='kde',linestyle='--')

pl.legend(('TUP','CTL'))
pl.xlabel('$\log\lambda^j_t$',fontsize=18)
pl.savefig('./figures/loglambda_distribution_by_treatment.png')

# Variance
lambdas.sort('lambda',inplace=True)
l0=lambdas.query('TUP==0')['lambda']
l1=lambdas.query('TUP==1')['lambda']

v0=l0.var()
v1=l1.var()

# Bootstrap distribution of variances
V=[]
M=[]
P=[]
for i in range(5000): #00):
    s0=np.random.randint(len(l0),size=len(l0)) # Draw sample size equal to TUP group
    s0.sort()
    s1=np.random.randint(len(l1),size=len(l0))
    s1.sort()

    V.append(l0.iloc[s0].var() > l1.iloc[s1].var())
    M.append(l0.iloc[s0].mean() > l1.iloc[s1].mean())
    P.append((l0.iloc[s0].as_matrix() > l1.iloc[s1].as_matrix()).mean())

print "Probability TUP variance smaller: %f" % np.array(V).mean()
print "Probability TUP mean smaller: %f" % np.array(M).mean()
print "Probability of complete stochastic dominance: %f" % (np.array(P)==1.).mean() # Not sure how to interpret this statistic--must be something better?

# Now back out prices:
ploc=b.filter(regex = '^Loc_.+')
ploc.rename(columns = lambda c: tuple(c[4:].split('_')),inplace=True)
p=ploc.T.unstack()

pse=se.T.filter(regex = '^Loc_.+')
pse.rename(columns = lambda c: tuple(c[4:].split('_')),inplace=True)
pse=pse.T.unstack()

# Drop "inferior" goods (inconsistent with theory)
beta=resultdf.query('beta_i>0')['beta_i']
A=np.matrix(np.eye(len(beta)) - np.diag(beta)).I

# Prices across locations
pbar=np.exp(A.dot(p.loc[beta.index]))
pse=pse.loc[beta.index] # Drop inferior goods

pdf=pd.DataFrame(pbar.A,index=beta.index,columns=p.columns.droplevel())
pdf.rename(columns=lambda c: " ".join([x.capitalize() for x in c.split(" ")]),inplace=True)
pdf.to_pickle('/tmp/prices.df')

prices=df_to_orgtbl(pdf,sedf=pse,tdf=pdf/sedf,float_fmt="%4.2f")
