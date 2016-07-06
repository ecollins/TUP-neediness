#!/usr/bin/env python

"""
A collection of functions pertaining to graphing Engel Curves
"""

from itertools import cycle
import variable_elasticity_utility as ves
import pylab as pl
import matplotlib.transforms as transforms
from warnings import warn

def line_to_axis(ax,x,y,xlabel=None,ylabel=None,fontsize=12):
    """
    Draw a line from a point x,y on to x and y axes ax.
    """
    v=ax.axis()

    if not (xlabel is None):
        ax.arrow(x,y,0,-(y-v[2])) # To x axis
        trans_x = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(x, -0.03, xlabel, transform=trans_x, fontsize=fontsize, va='center',ha='center')

    if not (ylabel is None):
        ax.arrow(x,y,-(x-v[0]),0.) # To y axis
        trans_y = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.01, y, ylabel, transform=trans_y, fontsize=fontsize, va='center',ha='right')

def plot(p,alpha,gamma,phi,labels=[],ybounds=[0,10],fname=None,NegativeDemands=False,use_linestyles=False,shares=False,use_figure=1):
    if not shares:
        f=lambda y: ves.marshalliandemands(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)
    else:
        f=lambda y: ves.budgetshares(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    if NegativeDemands:
        ymin=-sum([p[i]*phi[i] for i in range(len(p))])
    else:
        ymin=0
        
    if ybounds[0]<=ymin: ybounds[0]=ymin+1e-12
        
    Y=pl.linspace(ybounds[0],ybounds[1],100)

    pl.figure(use_figure)
    pl.clf()
    p=pl.plot(Y,[f(y) for y in Y])

    if use_linestyles:
        ls=cycle(['-',':','--','-.'])  # See pl.Line2D.lineStyles.keys()
        for line in p:
            line.set_ls(next(ls))

    if len(labels)>0:
        pl.legend(labels,loc=2)

    pl.xlabel('Total Expenditures',x=1.,fontsize=16,ha='right')
    pl.ylabel('Particular Expenditures',y=1.,fontsize=16,va='top')

    if fname:
        pl.savefig(fname)
    else:
        pl.show()

    return p

def plot_shares(p,alpha,gamma,phi,labels=[],ybounds=[0,10],fname=None,NegativeDemands=False):
    warn("Deprecated!  Use plot(...,shares=True) instead.")
         
    y=linspace(sum(-phi*p)+0.00001,ybounds[1],100)

    pl.figure(1)
    x=array([budgetshares(xbar,p,alpha,gamma,phi,NegativeDemands=NegativeDemands) for xbar in y])

    p=pl.plot(y,x)

    pl.xlabel('Total Expenditures')
    pl.ylabel('Expenditure Share')

    if len(labels)>0:
        pl.legend(labels,loc=2)

    if fname:
        pl.savefig(fname)
    else:
        pl.show()

    return p

    
if __name__=='__main__':
    p=plot([1.,1.,1.],[1.,2.,3.],[1.,1.,1.],[.5,0.,0.])
    
