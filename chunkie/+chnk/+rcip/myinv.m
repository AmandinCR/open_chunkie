  function Mi=myinv(M)
% compute the inverse of 2x2 block matrix
%
%  | A B |
%  | C D |
%
% when BC \approx -I, and A is close to singular

  np=length(M);
  A=M(1:2:np,1:2:np);
  B=M(1:2:np,2:2:np);
  C=M(2:2:np,1:2:np);
  D=M(2:2:np,2:2:np);

  T=inv(A-B/D*C);

  M11=T;
  M12=-T*B/D;
  M21=-D\C*T;
  M22=inv(D)+D\C*T*B/D;

  Mi=zeros(np);
  Mi(1:2:np,1:2:np) = M11;
  Mi(1:2:np,2:2:np) = M12;
  Mi(2:2:np,1:2:np) = M21;
  Mi(2:2:np,2:2:np) = M22;

      
  end


