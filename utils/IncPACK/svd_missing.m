function [u,s,v]=svd_missing(X);
% function [u,s,v,O]=svd_missing(X);
%  
% SVD of X when X may have missing values.
%
% X should be an m x n matrix or array, with n>m.
%
% A python wrapper around the matlab package IncPACK
% (https://www.math.fsu.edu/~cbaker/IncPACK/), which must 
% be supplied separately.
%
%  Ethan Ligon                                         December 2015

  [m,n]=size(X);

  % Need an initial value to initialize u that has no missing
  x0=nanmean(X')';
  X=[x0 X];

  opts.kstart=1;
  opts.numpasses=1;
  opts.lmax=4; % Updates one vector at a time
  opts.disp=2;
  opts.debug=0;
  [u,s,v,O]=seqkl(X,1,opts);
  v=v(2:end); # Discard first v row corresponding to initial x0
endfunction
