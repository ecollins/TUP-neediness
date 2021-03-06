#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from TUP import full_data, consumption_data, asset_vars

def df_to_orgtbl(df,float_fmt='%5.3f'):
    """
    Print pd.DataFrame in format which forms an org-table.
    Note that headers for code block should include ":results table raw".
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)
       
    return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')

def ttest_table(D,vardict):
    T = D[vardict.keys()]
    T.rename(columns=vardict,inplace=True)
    T['group'] = map(lambda x: "TUP" if x else "CTL" , T['TUP'])
    treat = T[T['group']=='TUP']
    control = T[T['group']=='CTL']
    Table = T.groupby('group').mean().T
    
    for var in Table: Table[var] = Table[var].apply(lambda x: round(x,3))

    Table['Diff.'] = map(str,Table['TUP']-Table['CTL'])
    Table['$p$-val'] = 0
    Table['$N$']=(T>0).sum()
    Table.drop('TUP', inplace=True)

    for var in T:
        if var in ('$N$','group','TUP'): continue
        pval = ttest_ind(treat[var].dropna(), control[var].dropna())[1]
        pval = round(pval,3)
        Table.ix[var,'$p$-val']+= pval
        for threshold in (.1, .05, .01):
            if pval < threshold: Table.ix[var,'Diff.']+="*"
    return Table

if True: #~ Make DataFrame
    D = full_data(balance=[])

    D['livestock_val_m'] = D.filter(regex='^asset_val_(cows|smallanimals|chickens|ducks|poultry)_m').sum(axis=1)
    D['livestock_val'] = D.filter(regex='^asset_val_(cows|smallanimals|chickens|ducks|poultry)').sum(axis=1) - D['livestock_val_m']
    A = asset_vars(D,year=2013)[0]
    D['Asset Tot'] = A['Total']
    D['Asset Prod'] = A['Productive']
    D["Cash Savings"] = D.filter(regex="^savings_.*_b$").sum(axis=1)
    D["Land Access (fedan)"] = D.filter(regex="^land_.*_b$").sum(axis=1)
    C = consumption_data(D)[0].ix[2013]
    food = ['c_cereals', 'c_maize', 'c_sorghum', 'c_millet', 'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread', 'c_beans', 'c_oil', 'c_salt', 'c_sugar', 'c_meat', 'c_livestock', 'c_poultry', 'c_fish', 'c_egg', 'c_nuts', 'c_milk', 'c_vegetables', 'c_fruit', 'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']
    month = ['c_fuel', 'c_medicine', 'c_airtime', 'c_cosmetics', 'c_soap', 'c_transport', 'c_entertainment', 'c_childcare', 'c_tobacco', 'c_batteries', 'c_church', 'c_othermonth']    
    year = ['c_clothesfootwear', 'c_womensclothes', 'c_childrensclothes', 'c_shoes', 'c_homeimprovement', 'c_utensils', 'c_furniture', 'c_textiles', 'c_ceremonies', 'c_funerals', 'c_charities', 'c_dowry', 'c_other']    
    C["Food"]  = C[[item for item in food  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Month"] = C[[item for item in month if item in C]].sum(axis=1).replace(0,np.nan)
    C["Year"]  = C[[item for item in year  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Tot"]   = C[["Food","Month","Year"]].sum(axis=1)
    D["Daily Exp"] = C["Tot"]
    D["Daily Food"] = C["Food"]

    drop_vars = ['c_milk', 'c_alcohol', 'c_spices', 'c_entertainment', 'c_otherfood', 'asset_val_house', 'asset_val_plough']
    D.drop([item for item in D if any(var in item for var in drop_vars)], 1, inplace=True)

consumption = dict([(c,c[2:-2].capitalize()) for c in D.filter(regex='^c_.*_b$').columns])
assets = dict([(a,a[10:].capitalize()) for a in D.filter(regex='^asset_val.*_b$').columns])
for lst in (consumption, assets): lst['TUP']='TUP'
other_outcomes = {"hh_size_b":"HH size",
        "child_total_b": "# Child",
        'asset_n_house_b':'No. Houses',
        'in_business_b':'In Business',
        'c_cereals_b':'Cereals',
        'c_cosmetics_b':'Cosmetics',
        'Land Access (fedan)':'Land Access (fedan)',
        'Cash Savings': 'Cash Savings',
        'Asset Tot': 'Asset Tot.',
        'Asset Prod': 'Asset Prod.',
        'Daily Exp':'Daily Exp',
        'Daily Food':'Daily Food',
        'TUP':'TUP'}

c_table     = ttest_table(D,consumption)
a_table     = ttest_table(D,assets)
other_table = ttest_table(D,other_outcomes)
    
c_orgtable     = df_to_orgtbl(c_table,float_fmt='%5.2f')
a_orgtable     = df_to_orgtbl(a_table,float_fmt='%5.2f')
other_orgtable = df_to_orgtbl(other_table,float_fmt='%5.2f')
    
tables="\n".join((c_orgtable,a_orgtable,other_orgtable))
