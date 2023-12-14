while t <= max_Iter
  a = 2 - 2 * (t / max_Iter); 
  for i = 1:N
    for d = 1:dim
      C1 = 2 * rand(); 
      C2 = 2 * rand();
      C3 = 2 * rand();
      Dalpha = abs(C1 * Xalpha(d) - X(i,d));
      Dbeta  = abs(C2 * Xbeta(d) - X(i,d));
      Ddelta = abs(C3 * Xdelta(d) - X(i,d));
      A1 = 2 * a * rand() - a; 
      Bstep1 = jBstepBGWO(A1 * Dalpha);
      Bstep2 = jBstepBGWO(A1 * Dbeta); 
      Bstep3 = jBstepBGWO(A1 * Ddelta);
      X1 = jBGWOupdate(Xalpha(d),Bstep1);
      X2 = jBGWOupdate(Xbeta(d),Bstep2);
      X3 = jBGWOupdate(Xdelta(d),Bstep3);
      r  = rand();
      if r < 1/3
        X(i,d) = X1;
      elseif r < 2/3  &&  r >=1/3
        X(i,d) = X2;
      else
        X(i,d) = X3;
      end
    end
  end
  for i = 1:N
    fit(i) = fun(feat,label,X(i,:),HO);
    if fit(i) < Falpha
      Falpha = fit(i);
      Xalpha = X(i,:);
    end
    if fit(i) < Fbeta  &&  fit(i) > Falpha
      Fbeta = fit(i); 
      Xbeta = X(i,:);
    end
    if fit(i) < Fdelta  &&  fit(i) > Falpha  &&  fit(i) > Fbeta
      Fdelta = fit(i);
      Xdelta = X(i,:);
    end
  end
  curve(t) = Falpha; 
  fprintf('\nIteration %d Best (BGWO1)= %f',t,curve(t))
  t = t + 1;
end
Pos   = 1:dim;
Sf    = Pos(Xalpha == 1);
Nf    = length(Sf); 
sFeat = feat(:,Sf); 
end
