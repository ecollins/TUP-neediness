function a=fixmiss(a,S,U)
% function ahat = fixmiss(a);
%
% Predict missing values in a vector a drawn from a matrix A.
% Return vector ahat with missing values (NaNs) replaced with
% predicted values.
%
% Suppose a is a column vector from a matrix A, with singular value
% decomposition A=USV'. The vector a can be partitioned in two parts y
% and x; the left-singular vectors U can similarly be partitioned into
% U = [U_y; U_x].
%
% Suppose now that y and the corresponding right-singular vector v are
% unknown or missing (taking the value of NaN).  Information from
% other vectors can be used to predict yhat = U_y*S*pinv(U_x*S)*x;
% with these predicted values then substituted in place of NaNs in a,
% producing a new "predicted" vector ahat.
%
% This function is designed to work with the IncPACK software of
% Christopher G. Baker, Kyle A. Gallivan, Paul Van Dooren
% (https://www.math.fsu.edu/~cbaker/IncPACK/).  If the argument U is
% not supplied then the routine relies on the existence of global
% variable SEQKL_U created by the IncPACK code.
%
% Ethan Ligon                                        December 2015
  global SEQKL_U;

  y=find(isnan(a));

  if y;
    if nargin<3;
      US=SEQKL_U*diag(S);
    else;
      US=U*diag(S);
    end;
    

    x=find(~isnan(a));

    a(y)=US(y,:)*pinv(US(x,:))*a(x);
  end

  return;
         
