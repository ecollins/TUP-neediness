#loglambdas=None 
loglambdas=True
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
from scipy.stats.distributions import chi2

def my_ancova(outcomelist,maindf,df14,df13=None,loglambdas=None,xvars=['hh_size','child_total','Loc'],outcomekind='Outcome',missing_y=0):

    xvars=list(set(xvars))
    
    maindf=maindf.copy()
    if not loglambdas is None:
        maindf['loglambda']=loglambdas['lambda']
        xvars+=['loglambda']
        print xvars
        print "loglambdas shape ", loglambdas.shape

    bothdf=[]
    for x in outcomelist:
        ydf=df14[[x]]
        ydf.rename(columns={x:("%s" % x).capitalize()},inplace=True)
        ydf=ydf.stack()
        if not df13 is None:
            bdf=df13[[x]].rename(columns={x:("%s" % x).capitalize()}).stack()

        rdict=dict(zip(xvars,["%s_%s" % (s,("%s" % x).capitalize()) for s in xvars]))

        xdf=maindf[xvars]
        xdf.index=pd.MultiIndex.from_tuples([(i,("%s" % x).capitalize()) for i in xdf.index])

        if not df13 is None:
            xdf=pd.concat([xdf,pd.DataFrame({'Baseline_%s' % ("%s" % x).capitalize():bdf})],axis=1)

        locations=pd.get_dummies(xdf['Loc'],prefix='Loc_%s' % ("%s" % x).capitalize())
        del xdf['Loc']
        xdf.rename(columns=rdict,inplace=True)
        xdf=xdf.join(locations)
        xdf.replace(to_replace=np.NaN,value=0,inplace=True)

        # Add row to restrict location dummies to sum to one
        ydf=pd.concat([ydf,pd.DataFrame([0],index=[(0,("%s" % x).capitalize())])])
        #xdf=pd.concat([xdf,pd.DataFrame([s.startswith('Loc_')+0. for s in xdf.columns],index=xdf.columns,columns=[(0,("%s" % x).capitalize())]).T])
        R=[s.startswith('Loc_')+0. for s in xdf.columns]
        R=pd.DataFrame(R,index=xdf.columns,columns=[(0,("%s" % x).capitalize())]).T
               
        xdf=pd.concat([xdf,R]) 

        xdf[0]=ydf
        if missing_y==0:
            xdf[0].fillna(value=0,inplace=True)

        xdf.dropna(how='any',inplace=True)

        bothdf.append(xdf)

    mydf=pd.concat(bothdf).fillna(value=0)

    X=mydf.iloc[:,1:]
    X.to_pickle('/tmp/myX.df')

    y=mydf[[0]]

    b,se=ols(X,y)

    e=y-X.dot(b.T)

    e.rename(columns={0:'Resid'},inplace=True)
    e.index.names=['HH',outcomekind]

    testdf=pd.merge(df[['TUP','CTL']].reset_index(),e.reset_index(),how='outer',on=['HH'])
    testdf.set_index(['HH',outcomekind],inplace=True)
    testdf.dropna(inplace=True)

    TUP=testdf['TUP'].mul(testdf['Resid'])
    TUP=TUP.unstack()
    Control=testdf['CTL'].mul(testdf['Resid']).unstack()

    e=(e-e.mean()).unstack()

    # Test of significant differences between treatment and control:
    # Weighting matrix:
    A=np.matrix((TUP-Control).cov().as_matrix()).I
    g=np.matrix((TUP-Control).mean())
    J=e.shape[0]*g*A*g.T # Chi2 statistic

    p=1-chi2.cdf(J,e.shape[1])

    chi2test="Chi2 test: %f (%f)" % (J,p)

    N=pd.Series([d.shape[0]-1
                  for d in bothdf],index=[d.index.levels[1][0] for d in bothdf])

    resultdf=pd.DataFrame({'TUP':TUP.mean(),'CTL':Control.mean(),'$N$':N})
    sedf=pd.DataFrame({'TUP':TUP.std()/np.sqrt(resultdf['$N$']),'CTL':Control.std()/np.sqrt(resultdf['$N$'])})
    
    resultdf['Diff.']=resultdf['TUP']-resultdf['CTL']
    sedf['Diff.']=np.sqrt(sedf['TUP']**2 + sedf['CTL']**2)

    sedf[r'$\log\lambda$']=e.std().as_matrix()/np.sqrt(resultdf['$N$'])

    tstats=pd.DataFrame({'TUP':resultdf['TUP']/sedf['TUP'],
                         'CTL':resultdf['CTL']/sedf['CTL'],
                         'Diff.':resultdf['Diff.']/sedf['Diff.']})

    if not loglambdas is None:
        llb=b.filter(like='loglambda_')
        resultdf[r'$\log\lambda$']=llb.iloc[0,:].as_matrix()
        tstats[r'$\log\lambda$']=resultdf['$\log\lambda$'].as_matrix()/sedf[r'$\log\lambda$']

    return resultdf,sedf,tstats,chi2test

df=pd.read_pickle('/tmp/ss-goods.df')

roster13=pd.read_stata('../data/Baseline/HHroster.dta')
roster14=pd.read_stata('../data/Midline/household_roster.dta')

occ14=roster14[['rid','s16a','s16b','s16c']]
#occ14=roster14.query('s14>16')[['rid','s16a','s16b','s16c']]
#occ14=roster14.query('s14<=16')[['rid','s16a','s16b','s16c']]

occ14.set_index('rid',inplace=True)
occ14=occ14.stack()
occ14.index=occ14.index.droplevel(1)

occs=pd.get_dummies(occ14)
occcount=occs.groupby(level=0).sum()

occdict={1:'Household chores/housewife',
         2:'Household land cultivation',
         3:"Laborer on some else's land",
         4:'Non-ag day laborer',
         5:'Factory worker',
         6:'Livestock husbandry',
         7:'Poultry husbandry',
         8:'Vegetable farming for sale',
         9:'Fisherman/Fish-farmer',
         10:'Maid/Servant',
         11:'Food processing for sale',
         12:'Stiching/Handicrafts/Tailoring for sale',
         13:'Driver',
         14:'Street vendor',
         15:'Homestead-based business',
         16:'Owner of shop/hotel/restaurant',
         17:'Worker in shop/hotel/restaurant',
         18:'Small trade/business',
         19:'Large-scale business',
         20:'Skilled labor',
         21:'Community health worker',
         22:'Teacher',
         23:'Professionals',
         24:'Manager/Administrator in private company',
         25:'Manager/Administrator in government',
         26:'Clerk/Employee in private company',
         27:'Clerk/Employee in government',
         28:'Politician',
         29:'Priest',
         30:'Student',
         31:'Beggar',
         32:'Unemployed, looking for work',
         33:'Unemployed, not looking for work',
         35:'Renting out land',
         36:'Renting out assets'}

# Drop occupations with very few observations
occcount=occcount.loc[:,occcount.sum()>30]
occcount.index.name='idno'
occcount.rename(columns=occdict,inplace=True)

if not loglambdas is None:
   loglambdas=pd.read_pickle('/tmp/ss-loglambdas.df')

resultdf,sedf,tstats,chi2test=my_ancova(occcount.columns,df,occcount,loglambdas=loglambdas,missing_y=0,outcomekind='Occupation')

tab=df_to_orgtbl(resultdf,sedf=sedf,tdf=tstats,float_fmt='%4.2f')
return tab
