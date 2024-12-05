%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEN020 - Computer Vision 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: rq
%
% RQ factorization. Factorizes a such a=rq where r upper tri. and q unit matrix
% If a is not square, then q is equal q=[q1 q2] where q1 is unit matrix
% 
%   inputs:    a: (M, N) array, M<=N
%
%   outputs:   r: (M, M) array
%                 upper triangular matrix
%              q: (M, N) array
%                 unit matrix; if a is not square, then q=[q1 q2]
%                 where q1 is unit matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r,q]=rq(a)
[m,n]=size(a);
e=eye(m);
p=e(:,[m:-1:1]);
[q0,r0]=qr(p*a(:,1:m)'*p);

r=p*r0'*p;
q=p*q0'*p;

fix=diag(sign(diag(r)));
r=r*fix;
q=fix*q;

if n>m
  q=[q, inv(r)*a(:,m+1:n)];
end
