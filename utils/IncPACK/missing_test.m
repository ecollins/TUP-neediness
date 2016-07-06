% m-file testing code to deal with missing data in  IncPACK.

m=50; n=5000;
sigma_e=0.2; % Increase this to add non-rank 1 variation to matrix.
missproportion=0.6;
baserank=10; 

a=rand(m,baserank);
b=rand(baserank,n);
e=rand(m,n)*sigma_e;  
X0=a*b + e;

M=(rand(m,n)<missproportion);

[u0,s0,v0]=svd(X0,1);
Xhat0=u0(:,1)*s0(1)*v0(:,1)';

X=X0;             
X(M)=NaN;
tic;

[u,s,v]=svd_missing(X);

toc

Xhat=u*s*v';

printf("Matrix size: (%d,%d); Base rank: %d, sigma_e: %5.4f; Proportion missing: %5.4f\n",m,n,baserank,sigma_e,missproportion)
printf("Percent error due to missing values: %5.4f\n",100*norm(Xhat-Xhat0,'fro')/norm(Xhat0,'fro'))
printf("Percent error due to rank 1 approximation: %5.4f\n",100*norm(X0-Xhat0,'fro')/norm(X0,'fro'))
printf("Correlation between non-missing U and missing U: %5.4f\n",corr([u0(:,1),u])(1,2))
printf("Correlation between non-missing V and missing V: %5.4f\n",
       corr([v0(:,1),v])(1,2))
disp("Singular values of matrix without missing divided by trace \
(greater than 0.05):"),
r2=diag(s0)'/trace(s0);
disp(r2(r2>.05))
