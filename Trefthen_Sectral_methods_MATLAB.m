% CHEB  compute D = differentiation matrix, x = Chebyshev grid

  function [D,x] = cheb(N)
  if N==0, D=0; x=1; return, end
  x = cos(pi*(0:N)/N)'; 
  c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
  X = repmat(x,1,N+1);
  dX = X-X';                  
  D  = (c*(1./c)')./(dX+(eye(N+1)));      % off-diagonal entries
  D  = D - diag(sum(D'));                 % diagonal entries

% CHEBFFT  Chebyshev differentiation via FFT. Simple, not optimal.  
%          If v is complex, delete "real" commands.

  function w = chebfft(v)
  N = length(v)-1; if N==0, w=0; return, end
  x = cos((0:N)'*pi/N);
  ii = 0:N-1;
  v = v(:); V = [v; flipud(v(2:N))];      % transform x -> theta          
  U = real(fft(V));
  W = real(ifft(1i*[ii 0 1-N:-1]'.*U));
  w = zeros(N+1,1);
  w(2:N) = -W(2:N)./sqrt(1-x(2:N).^2);    % transform theta -> x     
  w(1) = sum(ii'.^2.*U(ii+1))/N + .5*N*U(N+1);     
  w(N+1) = sum((-1).^(ii+1)'.*ii'.^2.*U(ii+1))/N + ...
              .5*(-1)^(N+1)*N*U(N+1);
% CLENCURT  nodes x (Chebyshev points) and weights w
%           for Clenshaw-Curtis quadrature

  function [x,w] = clencurt(N)
  theta = pi*(0:N)'/N; x = cos(theta);
  w = zeros(1,N+1); ii = 2:N; v = ones(N-1,1);
  if mod(N,2)==0 
    w(1) = 1/(N^2-1); w(N+1) = w(1);
    for k=1:N/2-1, v = v - 2*cos(2*k*theta(ii))/(4*k^2-1); end
    v = v - cos(N*theta(ii))/(N^2-1);;
  else
    w(1) = 1/N^2; w(N+1) = w(1);
    for k=1:(N-1)/2, v = v - 2*cos(2*k*theta(ii))/(4*k^2-1); end
  end
  w(ii) = 2*v/N;
% GAUSS  nodes x (Legendre points) and weights w
%        for Gauss quadrature

  function [x,w] = gauss(N)
  beta = .5./sqrt(1-(2*(1:N-1)).^(-2));
  T = diag(beta,1) + diag(beta,-1);
  [V,D] = eig(T);
  x = diag(D); [x,i] = sort(x);
  w = 2*V(1,i).^2;

  x = x(:); M = length(x); a = zeros(M,1); D = zeros(M,M);
  for k = 1:M; a(k) = prod(x(k)-x([1:k-1 k+1:M])); end
  for j = 1:M; t = x(:)-x(j); t(j) = 1; D(:,j) = 1./t; end
  A = diag(a); D = A*D/A;
  for j = 1:M
    sum = 0;
    for k = [1:j-1 j+1:M];
      sum = sum + 1/(x(j)-x(k));
    end
    D(j,j) = sum;
  end
% p1.m - convergence of fourth-order finite differences

% For various N, set up grid in [-pi,pi] and function u(x):
  Nvec = 2.^(3:12);
  clf, subplot('position',[.1 .4 .8 .5])
  for N = Nvec
    h = 2*pi/N; x = -pi + (1:N)'*h;
    u = exp(sin(x)); uprime = cos(x).*u;

    % Construct sparse fourth-order differentiation matrix:
    e = ones(N,1);
    D =   sparse(1:N,[2:N 1],2*e/3,N,N)...
        - sparse(1:N,[3:N 1 2],e/12,N,N);
    D = (D-D')/h;

    % Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,inf);
    loglog(N,error,'.','markersize',15), hold on
  end
  grid on, xlabel N, ylabel error
  title('Convergence of fourth-order finite differences')
  semilogy(Nvec,Nvec.^(-4),'--') 
  text(105,5e-8,'N^{-4}','fontsize',18)
% p10.m - polynomials and corresponding equipotential curves

  N = 16; clf
  for i = 1:2
    if i==1, s = 'equispaced points'; x = -1 + 2*(0:N)/N; end
    if i==2, s = 'Chebyshev points';  x = cos(pi*(0:N)/N); end
    p = poly(x);

    % Plot p(x) over [-1,1]:
    xx = -1:.005:1; pp = polyval(p,xx);
    subplot(2,2,2*i-1)
    plot(x,0*x,'.','markersize',13), hold on
    plot(xx,pp), grid on
    set(gca,'xtick',-1:.5:1), title(s)

    % Plot equipotential curves:
    subplot(2,2,2*i)
    plot(real(x),imag(x),'.','markersize',13), hold on
    axis([-1.4 1.4 -1.12 1.12])
    xgrid = -1.4:.02:1.4; ygrid = -1.12:.02:1.12;
    [xx,yy] = meshgrid(xgrid,ygrid); zz = xx+1i*yy;
    pp = polyval(p,zz); levels = 10.^(-4:0);
    contour(xx,yy,abs(pp),levels), title(s), colormap(1e-6*[1 1 1]);
  end
% p11.m - Chebyshev differentation of a smooth function

  xx = -1:.01:1; uu = exp(xx).*sin(5*xx); clf
  for N = [10 20]
    [D,x] = cheb(N); u = exp(x).*sin(5*x);
      subplot('position',[.15 .66-.4*(N==20) .31 .28])
      plot(x,u,'.','markersize',14), grid on
      line(xx,uu)
      title(['u(x),  N=' int2str(N)])
    error = D*u - exp(x).*(sin(5*x)+5*cos(5*x));
      subplot('position',[.55 .66-.4*(N==20) .31 .28])
      plot(x,error,'.','markersize',14), grid on
      line(x,error)
      title(['  error in u''(x),  N=' int2str(N)])
  end
% p12.m - accuracy of Chebyshev spectral differentiation
%         (compare p7.m)

% Compute derivatives for various values of N:
  Nmax = 50; E = zeros(3,Nmax);
  for N = 1:Nmax;
    [D,x] = cheb(N);
    v = abs(x).^3; vprime = 3*x.*abs(x);     % 3rd deriv in BV
    E(1,N) = norm(D*v-vprime,inf);
    v = exp(-x.^(-2)); vprime = 2.*v./x.^3;  % C-infinity
    E(2,N) = norm(D*v-vprime,inf);
    v = 1./(1+x.^2); vprime = -2*x.*v.^2;    % analytic in [-1,1]
    E(3,N) = norm(D*v-vprime,inf);
    v = x.^10; vprime = 10*x.^9;             % polynomial
    E(4,N) = norm(D*v-vprime,inf);
  end

% Plot results:
  titles = {'|x^3|','exp(-x^{-2})','1/(1+x^2)','x^{10}'}; clf
  for iplot = 1:4
    subplot(2,2,iplot)
    semilogy(1:Nmax,E(iplot,:),'.','markersize',12)
    line(1:Nmax,E(iplot,:))
    axis([0 Nmax 1e-16 1e3]), grid on
    set(gca,'xtick',0:10:Nmax,'ytick',(10).^(-15:5:0))
    xlabel N, ylabel error, title(titles(iplot))
  end
% p13.m - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0

  N = 16;
  [D,x] = cheb(N);        
  D2 = D^2;              
  D2 = D2(2:N,2:N);                   % boundary conditions
  f = exp(4*x(2:N));           
  u = D2\f;                           % Poisson eq. solved here
  u = [0;u;0];              
  clf, subplot('position',[.1 .4 .8 .5])
  plot(x,u,'.','markersize',16)
  xx = -1:.01:1;
  uu = polyval(polyfit(x,u,N),xx);    % interpolate grid data
  line(xx,uu)
  grid on
  exact = ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16; 
  title(['max err = ' num2str(norm(uu-exact,inf))],'fontsize',12)
% p14.m - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
%         (compare p13.m)

  N = 16;
  [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
  u = zeros(N-1,1);
  change = 1; it = 0;
  while change > 1e-15                   % fixed-point iteration
    unew = D2\exp(u); 
    change = norm(unew-u,inf);
    u = unew; it = it+1;
  end
  u = [0;u;0];
  clf, subplot('position',[.1 .4 .8 .5])
  plot(x,u,'.','markersize',16)
  xx = -1:.01:1;
  uu = polyval(polyfit(x,u,N),xx);
  line(xx,uu), grid on
  title(sprintf('no. steps = %d      u(0) =%18.14f',it,u(N/2+1)))
% p15.m - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0

  N = 36; [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
  [V,Lam] = eig(D2); lam = diag(Lam);
  [foo,ii] = sort(-lam);          % sort eigenvalues and -vectors
  lam = lam(ii); V = V(:,ii); clf
  for j = 5:5:30                  % plot 6 eigenvectors
    u = [0;V(:,j);0]; subplot(7,1,j/5)
    plot(x,u,'.','markersize',12), grid on
    xx = -1:.01:1; uu = polyval(polyfit(x,u,N),xx);
    line(xx,uu), axis off
    text(-.4,.5,sprintf('eig %d =%20.13f*4/pi^2',j,lam(j)*4/pi^2))
    text(.7,.5,sprintf('%4.1f  ppw', 4*N/(pi*j)))
  end  
% p16.m - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary

% Set up grids and tensor product Laplacian and solve for u:
  N = 24; [D,x] = cheb(N); y = x;
  [xx,yy] = meshgrid(x(2:N),y(2:N));
  xx = xx(:); yy = yy(:);       % stretch 2D grids to 1D vectors
  f = 10*sin(8*xx.*(yy-1));
  D2 = D^2; D2 = D2(2:N,2:N); I = eye(N-1);
  L = kron(I,D2) + kron(D2,I);                       % Laplacian 
  figure(1), clf, spy(L), drawnow
  tic, u = L\f; toc          % solve problem and watch the clock

% Reshape long 1D results onto 2D grid:
  uu = zeros(N+1,N+1); uu(2:N,2:N) = reshape(u,N-1,N-1);
  [xx,yy] = meshgrid(x,y);
  value = uu(N/4+1,N/4+1);

% Interpolate to finer grid and plot:
  [xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
  uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
  figure(2), clf, mesh(xxx,yyy,uuu), colormap(1e-6*[1 1 1]);
  xlabel x, ylabel y, zlabel u
  text(.4,-.3,-.3,sprintf('u(2^{-1/2},2^{-1/2}) = %14.11f',value))
% p17.m - Helmholtz eq. u_xx + u_yy + (k^2)u = f
%         on [-1,1]x[-1,1]    (compare p16.m)

% Set up spectral grid and tensor product Helmholtz operator:
  N = 24; [D,x] = cheb(N); y = x;
  [xx,yy] = meshgrid(x(2:N),y(2:N));
  xx = xx(:); yy = yy(:);
  f = exp(-10*((yy-1).^2+(xx-.5).^2));
  D2 = D^2; D2 = D2(2:N,2:N); I = eye(N-1);
  k = 9;
  L = kron(I,D2) + kron(D2,I) + k^2*eye((N-1)^2);

% Solve for u, reshape to 2D grid, and plot:
  u = L\f;
  uu = zeros(N+1,N+1); uu(2:N,2:N) = reshape(u,N-1,N-1);
  [xx,yy] = meshgrid(x,y);
  [xxx,yyy] = meshgrid(-1:.0333:1,-1:.0333:1);
  uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
  figure(1), clf, mesh(xxx,yyy,uuu), colormap(1e-6*[1 1 1]);
  xlabel x, ylabel y, zlabel u
  text(.2,1,.022,sprintf('u(0,0) = %13.11f',uu(N/2+1,N/2+1)))
  figure(2), clf, contour(xxx,yyy,uuu)
  colormap(1e-6*[1 1 1]); axis square
% p18.m - Chebyshev differentiation via FFT (compare p11.m)

  xx = -1:.01:1; ff = exp(xx).*sin(5*xx); clf
  for N = [10 20]
    x = cos(pi*(0:N)'/N); f = exp(x).*sin(5*x);
      subplot('position',[.15 .66-.4*(N==20) .31 .28])
      plot(x,f,'.','markersize',14), grid on
      line(xx,ff)
      title(['f(x), N=' int2str(N)])
    error = chebfft(f) - exp(x).*(sin(5*x)+5*cos(5*x));
      subplot('position',[.55 .66-.4*(N==20) .31 .28])
      plot(x,error,'.','markersize',14), grid on
      line(x,error)
      title(['error in f''(x), N=' int2str(N)])
  end

  
% p19.m - 2nd-order wave eq. on Chebyshev grid (compare p6.m)

% Time-stepping by leap frog formula:
  N = 80; x = cos(pi*(0:N)/N); dt = 8/N^2;
  v = exp(-200*x.^2); vold = exp(-200*(x-dt).^2);
  tmax = 4; tplot = .075; 
  plotgap = round(tplot/dt); dt = tplot/plotgap;
  nplots = round(tmax/tplot);
  plotdata = [v; zeros(nplots,N+1)]; tdata = 0;
  clf, drawnow, h = waitbar(0,'please wait...');
  set(gcf,'renderer','zbuffer')
  for i = 1:nplots, waitbar(i/nplots)
    for n = 1:plotgap
      w = chebfft(chebfft(v))'; w(1) = 0; w(N+1) = 0;
      vnew = 2*v - vold + dt^2*w; vold = v; v = vnew;
    end
    plotdata(i+1,:) = v; tdata = [tdata; dt*i*plotgap];
  end

% Plot results:
  clf, drawnow, waterfall(x,tdata,plotdata)
  axis([-1 1 0 tmax -2 2]), view(10,70), grid off
  colormap(1e-6*[1 1 1]); ylabel t, zlabel u, close(h)
% p2.m - convergence of periodic spectral method (compare p1.m)

% For various N (even), set up grid as before:
  clf, subplot('position',[.1 .4 .8 .5])
  for N = 2:2:100;
    h = 2*pi/N;
    x = -pi + (1:N)'*h;
    u = exp(sin(x)); uprime = cos(x).*u;

    % Construct spectral differentiation matrix:
    column = [0 .5*(-1).^(1:N-1).*cot((1:N-1)*h/2)];
    D = toeplitz(column,column([1 N:-1:2]));

    % Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,inf);
    loglog(N,error,'.','markersize',15), hold on
  end
  grid on, xlabel N, ylabel error
  title('Convergence of spectral differentiation')
% p20.m - 2nd-order wave eq. in 2D via FFT (compare p19.m)

% Grid and initial data:
  N = 24; x = cos(pi*(0:N)/N); y = x';
  dt = 6/N^2;
  [xx,yy] = meshgrid(x,y);
  plotgap = round((1/3)/dt); dt = (1/3)/plotgap;
  vv = exp(-40*((xx-.4).^2 + yy.^2));
  vvold = vv; 

% Time-stepping by leap frog formula:
  [ay,ax] = meshgrid([.56 .06],[.1 .55]); clf
  for n = 0:3*plotgap
    t = n*dt;
    if rem(n+.5,plotgap)<1     % plots at multiples of t=1/3
      i = n/plotgap+1;
      subplot('position',[ax(i) ay(i) .36 .36])
      [xxx,yyy] = meshgrid(-1:1/16:1,-1:1/16:1);
      vvv = interp2(xx,yy,vv,xxx,yyy,'cubic');
      mesh(xxx,yyy,vvv), axis([-1 1 -1 1 -0.15 1])
      colormap(1e-6*[1 1 1]); title(['t = ' num2str(t)]), drawnow
    end
    uxx = zeros(N+1,N+1); uyy = zeros(N+1,N+1);
    ii = 2:N;
    for i = 2:N                % 2nd derivs wrt x in each row
      v = vv(i,:); V = [v fliplr(v(ii))];
      U = real(fft(V));
      W1 = real(ifft(1i*[0:N-1 0 1-N:-1].*U)); % diff wrt theta
      W2 = real(ifft(-[0:N 1-N:-1].^2.*U));    % diff^2 wrt theta
      uxx(i,ii) = W2(ii)./(1-x(ii).^2) - x(ii).* ... 
                     W1(ii)./(1-x(ii).^2).^(3/2);
    end
    for j = 2:N                % 2nd derivs wrt y in each column
      v = vv(:,j); V = [v; flipud(v(ii))];
      U = real(fft(V));
      W1 = real(ifft(1i*[0:N-1 0 1-N:-1]'.*U));% diff wrt theta   
      W2 = real(ifft(-[0:N 1-N:-1]'.^2.*U));   % diff^2 wrt theta
      uyy(ii,j) = W2(ii)./(1-y(ii).^2) - y(ii).* ...
                     W1(ii)./(1-y(ii).^2).^(3/2);
    end
    vvnew = 2*vv - vvold + dt^2*(uxx+uyy); 
    vvold = vv; vv = vvnew;
  end
% p20u.m - 2nd-order wave eq. in 2D via FFT (compare p19.m) UNSTABLE

% Grid and initial data:
  N = 24; x = cos(pi*(0:N)/N); y = x'; dt = 6.6/N^2;
  [xx,yy] = meshgrid(x,y);
  plotgap = round((1/3)/dt);
  vv = exp(-40*((xx-.4).^2 + yy.^2));
  vvold = vv; vvnew = 0*vv;

% Time-stepping by leap frog formula:
  [ay,ax] = meshgrid([.56 .06],[.1 .55]);
  for n = 0:3*plotgap
    t = n*dt;
    if rem(n+.5,plotgap)<1       % Plots at multiples of t=1/3
      i = n/plotgap+1; subplot('position',[ax(i) ay(i) .36 .36])
      [xxx,yyy] = meshgrid(-1:.05:1,-1:.05:1);
      vvv = interp2(xx,yy,vv,xxx,yyy,'cubic');
      mesh(xxx,yyy,vvv), axis([-1 1 -1 1 -0.15 1])
      colormap(1e-6*[1 1 1]); title(['t = ' num2str(t)]), drawnow
    end
    uxx = zeros(N+1,N+1); uyy = zeros(N+1,N+1);
    ii = 2:N;
    for i = ii                 % 2nd derivs wrt x in each row
      v = vv(i,:); V = [v fliplr(v(ii))]; U = real(fft(V));
      W1 = real(ifft(1i*[0:N-1 0 1-N:-1].*U)); % diff wrt theta
      W2 = real(ifft(-[0:N 1-N:-1].^2.*U));    % diff^2 wrt theta
      uxx(i,ii) = W2(ii)./(1-x(ii).^2) - x(ii).* ... 
                     W1(ii)./(1-x(ii).^2).^(3/2);
    end
    for j = ii                 % 2nd derivs wrt y in each column
      v = vv(:,j); V = [v; flipud(v(ii))]; U = real(fft(V));
      W1 = real(ifft(1i*[0:N-1 0 1-N:-1]'.*U));% diff wrt theta   
      W2 = real(ifft(-[0:N 1-N:-1]'.^2.*U));   % diff^2 wrt theta
      uyy(ii,j) = W2(ii)./(1-y(ii).^2) - y(ii).* ...
                     W1(ii)./(1-y(ii).^2).^(3/2);
    end
    vvnew = 2*vv - vvold + dt^2*(uxx+uyy); 
    vvold = vv; vv = vvnew;
    if max(max(abs(vv)))>2, break, end
  end

close
      [xxx,yyy] = meshgrid(-1:.05:1,-1:.05:1);
      vvv = interp2(xx,yy,vv,xxx,yyy,'cubic');
      mesh(xxx,yyy,vvv), colormap(1e-6*[1 1 1]);
      axis([-1 1 -1 1 -1 1]), title(['t = ' num2str(t)]), drawnow

% p21.m - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
%         (compare p8.m and p. 724 of Abramowitz & Stegun)

  N = 42; h = 2*pi/N; x = h*(1:N);
  D2 = toeplitz([-pi^2/(3*h^2)-1/6 ...
                 -.5*(-1).^(1:N-1)./sin(h*(1:N-1)/2).^2]);
  qq = 0:.2:15; data = [];
  for q = qq;
    e = sort(eig(-D2 + 2*q*diag(cos(2*x))))';
    data = [data; e(1:11)];
  end
  clf, subplot(1,2,1)
  set(gca,'colororder',[0 0 1],'linestyleorder','-|--'), hold on
  plot(qq,data), xlabel q, ylabel \lambda
  axis([0 15 -24 32]), set(gca,'ytick',-24:4:32)
% p22.m - 5th eigenvector of Airy equation u_xx = lambda*x*u

  clf
  for N = 12:12:48
    [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
    [V,Lam] = eig(D2,diag(x(2:N)));      % generalized ev problem
    Lam = diag(Lam); ii = find(Lam>0);
    V = V(:,ii); Lam = Lam(ii);
    [foo,ii] = sort(Lam); ii = ii(5); lambda = Lam(ii);
    v = [0;V(:,ii);0]; v = v/v(N/2+1)*airy(0);
    xx = -1:.01:1; vv = polyval(polyfit(x,v,N),xx);
    subplot(2,2,N/12), plot(xx,vv), grid on
    title(sprintf('N = %d     eig = %15.10f',N,lambda))
  end
% p23.m - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
%         (compare p16.m)

% Set up tensor product Laplacian and compute 4 eigenmodes:
  N = 16; [D,x] = cheb(N); y = x;
  [xx,yy] = meshgrid(x(2:N),y(2:N)); xx = xx(:); yy = yy(:);
  D2 = D^2; D2 = D2(2:N,2:N); I = eye(N-1);
  L = -kron(I,D2) - kron(D2,I);                %  Laplacian
  L = L + diag(exp(20*(yy-xx-1)));             %  + perturbation
  [V,D] = eig(L); D = diag(D);
  [D,ii] = sort(D); ii = ii(1:4); V = V(:,ii);

% Reshape them to 2D grid, interpolate to finer grid, and plot:
  [xx,yy] = meshgrid(x,y);
  fine = -1:.02:1; [xxx,yyy] = meshgrid(fine,fine);
  uu = zeros(N+1,N+1);
  [ay,ax] = meshgrid([.56 .04],[.1 .5]); clf
  for i = 1:4
    uu(2:N,2:N) = reshape(V(:,i),N-1,N-1);
    uu = uu/norm(uu(:),inf);
    uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
    subplot('position',[ax(i) ay(i) .38 .38])
    contour(fine,fine,uuu,-.9:.2:.9)
    colormap(1e-6*[1 1 1]); axis square
    title(['eig = ' num2str(D(i)/(pi^2/4),'%18.12f') '\pi^2/4'])
  end
% p23a.m - eigenvalues of UNperturbed Laplacian on [-1,1]x[-1,1]
%          (compare p16.m)

% Set up tensor product Laplacian and compute 4 eigenmodes:
  N = 16; [D,x] = cheb(N); y = x;
  [xx,yy] = meshgrid(x(2:N),y(2:N)); xx = xx(:); yy = yy(:);
  D2 = D^2; D2 = D2(2:N,2:N); I = eye(N-1);
  L = -kron(I,D2) - kron(D2,I);                %  Laplacian
% L = L + diag(exp(20*(yy-xx-1)));             %  + perturbation
  [V,D] = eig(L); D = diag(D);
  [D,ii] = sort(D); ii = ii(1:4); V = V(:,ii);

% Reshape them to 2D grid, interpolate to finer grid, and plot:
  [xx,yy] = meshgrid(x,y);
  fine = -1:.02:1; [xxx,yyy] = meshgrid(fine,fine);
  uu = zeros(N+1,N+1);
  [ay,ax] = meshgrid([.56 .04],[.1 .5]); clf
  for i = 1:4
    uu(2:N,2:N) = reshape(V(:,i),N-1,N-1);
    uu = uu/norm(uu(:),inf);
    uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
    subplot('position',[ax(i) ay(i) .38 .38])
    contour(fine,fine,uuu,-.9:.2:.9)
    colormap(1e-6*[1 1 1]); axis square
    title(['eig = ' num2str(D(i)/(pi^2/4),'%18.12f') '\pi^2/4'])
  end
% p24.m - pseudospectra of Davies's complex harmonic oscillator
%         (For finer, slower plot, change 0:2 to 0:.5.)

% Eigenvalues:
  N = 70; [D,x] = cheb(N); x = x(2:N);
  L = 6; x = L*x; D = D/L;                   % rescale to [-L,L]
  A = -D^2; A = A(2:N,2:N) + (1+3i)*diag(x.^2);
  clf, plot(eig(A),'.','markersize',14)
  axis([0 50 0 40]), drawnow, hold on
  
% Pseudospectra:
  x = 0:2:50; y = 0:2:40; [xx,yy] = meshgrid(x,y); zz = xx+1i*yy;
  I = eye(N-1); sigmin = zeros(length(y),length(x));
  h = waitbar(0,'please wait...');
  for j = 1:length(x), waitbar(j/length(x))
    for i = 1:length(y), sigmin(i,j) = min(svd(zz(i,j)*I-A)); end
  end, close(h)
  contour(x,y,sigmin,10.^(-4:.5:-.5)), colormap(1e-6*[1 1 1]);
% p24fine.m - pseudospectra of Davies's complex harmonic oscillator
%             (This is the finer, slower plot, with 0:2 changed to 0:.5.)

% Eigenvalues:
  N = 70; [D,x] = cheb(N); x = x(2:N);
  L = 6; x = L*x; D = D/L;                   % rescale to [-L,L]
  A = -D^2; A = A(2:N,2:N) + (1+3i)*diag(x.^2);
  clf, plot(eig(A),'.','markersize',14)
  axis([0 50 0 40]), drawnow, hold on
  
% Pseudospectra:
  x = 0:.5:50; y = 0:.5:40; [xx,yy] = meshgrid(x,y); zz = xx+1i*yy;
  I = eye(N-1); sigmin = zeros(length(y),length(x));
  h = waitbar(0,'please wait...');
  for j = 1:length(x), waitbar(j/length(x))
    for i = 1:length(y), sigmin(i,j) = min(svd(zz(i,j)*I-A)); end
  end, close(h)
  contour(x,y,sigmin,10.^(-4:.5:-.5)), colormap(1e-6*[1 1 1]);
% p25.m - stability regions for ODE formulas

% Adams-Bashforth:
  clf, subplot('position',[.1 .56 .38 .38])
  plot([-8 8],[0 0]), hold on, plot([0 0],[-8 8])
  z = exp(1i*pi*(0:200)/100); r = z-1;
  s = 1; plot(r./s)                                  % order 1
  s = (3-1./z)/2; plot(r./s)                         % order 2
  s = (23-16./z+5./z.^2)/12; plot(r./s)              % order 3
  axis([-2.5 .5 -1.5 1.5]), axis square, grid on
  title Adams-Bashforth

% Adams-Moulton:
  subplot('position',[.5 .56 .38 .38])
  plot([-8 8],[0 0]), hold on, plot([0 0],[-8 8])
  s = (5*z+8-1./z)/12; plot(r./s)                    % order 3
  s = (9*z+19-5./z+1./z.^2)/24; plot(r./s)           % order 4 
  s = (251*z+646-264./z+106./z.^2-19./z.^3)/720; plot(r./s)    % 5
  d = 1-1./z;
  s = 1-d/2-d.^2/12-d.^3/24-19*d.^4/720-3*d.^5/160; plot(d./s) % 6
  axis([-7 1 -4 4]), axis square, grid on, title Adams-Moulton

% Backward differentiation:
  subplot('position',[.1 .04 .38 .38])
  plot([-40 40],[0 0]), hold on, plot([0 0],[-40 40]) 
  r = 0; for i = 1:6, r = r+(d.^i)/i; plot(r), end   % orders 1-6
  axis([-15 35 -25 25]), axis square, grid on
  title('backward differentiation')

% Runge-Kutta:
  subplot('position',[.5 .04 .38 .38])
  plot([-8 8],[0 0]), hold on, plot([0 0],[-8 8])
  w = 0; W = w; for i = 2:length(z)                  % order 1
    w = w-(1+w-z(i)); W = [W; w]; end, plot(W)
  w = 0; W = w; for i = 2:length(z)                  % order 2
    w = w-(1+w+.5*w^2-z(i)^2)/(1+w); W = [W; w];
    end, plot(W)
  w = 0; W = w; for i = 2:length(z)                  % order 3
    w = w-(1+w+.5*w^2+w^3/6-z(i)^3)/(1+w+w^2/2); W = [W; w];
    end, plot(W)
  w = 0; W = w; for i = 2:length(z)                  % order 4
    w = w-(1+w+.5*w^2+w^3/6+w.^4/24-z(i)^4)/(1+w+w^2/2+w.^3/6);
    W = [W; w]; end, plot(W)
  axis([-5 2 -3.5 3.5]), axis square, grid on, title Runge-Kutta
% p26.m - eigenvalues of 2nd-order Chebyshev diff. matrix

  N = 60; [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
  [V,Lam] = eig(D2); 
  [foo,ii] = sort(-diag(Lam)); e = diag(Lam(ii,ii)); V = V(:,ii);

% Plot eigenvalues:
  clf, subplot('position',[.1 .62 .8 .3])
  loglog(-e,'.','markersize',10), ylabel eigenvalue
  title(['N = ' int2str(N) ...
     '      max |\lambda| = ' num2str(max(-e)/N^4) 'N^4'])
  hold on, semilogy(2*N/pi*[1 1],[1 1e6],'--r')
  text(2.1*N/pi,24,'2\pi / N','fontsize',12)

% Plot eigenmodes N/4 (physical) and N (nonphysical):
  vN4 = [0; V(:,N/4-1); 0];
  xx = -1:.01:1; vv = polyval(polyfit(x,vN4,N),xx);
  subplot('position',[.1 .36 .8 .15]), plot(xx,vv), hold on
  plot(x,vN4,'.','markersize',9), title('eigenmode N/4')
  vN = V(:,N-1);
  subplot('position',[.1 .1 .8 .15])
  semilogy(x(2:N),abs(vN)), axis([-1 1 5e-6 1]), hold on 
  plot(x(2:N),abs(vN),'.','markersize',9)
  title('absolute value of eigenmode N    (log scale)')
% p27.m - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
%         FFT with integrating factor v = exp(-ik^3t)*u-hat.

% Set up grid and two-soliton initial data:
  N = 256; dt = .4/N^2; x = (2*pi/N)*(-N/2:N/2-1)';
  A = 25; B = 16; clf, drawnow, set(gcf,'renderer','zbuffer')
  u = 3*A^2*sech(.5*(A*(x+2))).^2 + 3*B^2*sech(.5*(B*(x+1))).^2; 
  v = fft(u); k = [0:N/2-1 0 -N/2+1:-1]'; ik3 = 1i*k.^3;

% Solve PDE and plot results:
  tmax = 0.006; nplt = floor((tmax/25)/dt); nmax = round(tmax/dt);
  udata = u; tdata = 0; h = waitbar(0,'please wait...');
  for n = 1:nmax
    t = n*dt; g = -.5i*dt*k;
    E = exp(dt*ik3/2); E2 = E.^2;
    a = g.*fft(real( ifft(     v    ) ).^2);
    b = g.*fft(real( ifft(E.*(v+a/2)) ).^2);     % 4th-order
    c = g.*fft(real( ifft(E.*v + b/2) ).^2);     % Runge-Kutta
    d = g.*fft(real( ifft(E2.*v+E.*c) ).^2);
    v = E2.*v + (E2.*a + 2*E.*(b+c) + d)/6;
    if mod(n,nplt) == 0 
      u = real(ifft(v)); waitbar(n/nmax)
      udata = [udata u]; tdata = [tdata t];
    end
  end
  waterfall(x,tdata,udata'), colormap(1e-6*[1 1 1]); view(-20,25)
  xlabel x, ylabel t, axis([-pi pi 0 tmax 0 2000]), grid off
  set(gca,'ztick',[0 2000]), close(h), pbaspect([1 1 .13])
% p28.m - eigenmodes of Laplacian on the disk (compare p22.m)

% r coordinate, ranging from -1 to 1 (N must be odd):
  N = 25; N2 = (N-1)/2;
  [D,r] = cheb(N);
  D2 = D^2;
  D1 = D2(2:N2+1,2:N2+1); D2 = D2(2:N2+1,N:-1:N2+2);
  E1 =  D(2:N2+1,2:N2+1); E2 =  D(2:N2+1,N:-1:N2+2);

% t = theta coordinate, ranging from 0 to 2*pi (M must be even):
  M = 20; dt = 2*pi/M; t = dt*(1:M)'; M2 = M/2;
  D2t = toeplitz([-pi^2/(3*dt^2)-1/6 ...
                 .5*(-1).^(2:M)./sin(dt*(1:M-1)/2).^2]);

% Laplacian in polar coordinates:
  R = diag(1./r(2:N2+1));
  Z = zeros(M2); I = eye(M2);
  L = kron(D1+R*E1,eye(M)) + kron(D2+R*E2,[Z I;I Z]) ...
                           + kron(R^2,D2t);       

% Compute four eigenmodes:
  index = [1 3 6 10];
  [V,Lam] = eig(-L); Lam = diag(Lam); [Lam,ii] = sort(Lam);
  ii = ii(index); V = V(:,ii);
  Lam = sqrt(Lam(index)/Lam(1));

% Plot eigenmodes with nodal lines underneath:
  [rr,tt] = meshgrid(r(1:N2+1),[0;t]);
  [xx,yy] = pol2cart(tt,rr);
  z = exp(1i*pi*(-100:100)/100);
  [ay,ax] = meshgrid([.58 .1],[.1 .5]); clf
  for i = 1:4
    u = reshape(real(V(:,i)),M,N2);
    u = [zeros(M+1,1) u([M 1:M],:)];
    u = u/norm(u(:),inf);
    subplot('position',[ax(i) ay(i) .4 .4])
    plot(z), axis(1.05*[-1 1 -1 1 -1 1]), axis off, hold on
    mesh(xx,yy,u)
    view(0,20), colormap(1e-6*[1 1 1]); axis square
    contour3(xx,yy,u-1,[-1 -1])
    plot3(real(z),imag(z),-abs(z))
    text(-.8,4,['Mode ' int2str(index(i))],'fontsize',9)
    text(-.8,3.5, ['\lambda = ', num2str(Lam(i),...
                               '%16.10f')],'fontsize',9)
  end
% p28b.m - eigenmodes of Laplacian on the disk

% r coordinate, ranging from -1 to 1 (N must be odd)
  N = 25; [D,r] = cheb(N); N2 = (N-1)/2; D2 = D^2;
  D1 = D2(2:N2+1,2:N2+1); D2 = D2(2:N2+1,N:-1:N2+2);
  E1 =  D(2:N2+1,2:N2+1); E2 =  D(2:N2+1,N:-1:N2+2);

% t = theta coordinate, ranging from 0 to 2*pi (M must be even):
  M = 20; dt = 2*pi/M; t = dt*(1:M)'; M2 = M/2;
  D2t = toeplitz([-pi^2/(3*dt^2)-1/6 ...
                    .5*(-1).^(2:M)./sin(dt*(1:M-1)/2).^2]);

% Laplacian in polar coordinates:
  R = diag(1./r(2:N2+1)); Z = zeros(M2); I = eye(M2);
  L = kron(D1+R*E1,eye(M))+ kron(D2+R*E2,[Z I;I Z])+ kron(R^2,D2t);

% Compute four eigenmodes:
  index = 1:25;
  [V,Lam] = eig(-L); Lam = diag(Lam);
  [Lam,ii] = sort(real(Lam)); ii = ii(index); V = real(V(:,ii));
  Lam = sqrt(Lam(index)/Lam(1));

% Plot eigenmodes with nodal lines underneath:
  [rr,tt] = meshgrid(r(1:N2+1),[0;t]); [xx,yy] = pol2cart(tt,rr);
  z = exp(sqrt(-1)*pi*(-100:100)/100);
  [ay,ax] = meshgrid(.8:-.2:0,0:.16:.64); clf
  for i = 1:25
    u = reshape(real(V(:,i)),M,N2);
    u = [zeros(M+1,1) u([M 1:M],:)];
    u = u/norm(u(:),inf);
    subplot('position',[ax(i) ay(i) .16 .2]), plot(z)
    axis(1.25*[-1 1 -1 1 -1 1]), axis off, hold on
    view(0,90), colormap(1e-6*[1 1 1]); axis square
    contour3(xx,yy,u-1,[-1 -1]), plot3(real(z),imag(z),-abs(z))
    text(-.3,1.15,num2str(Lam(i)),'fontsize',7)
  end
% p29.m - solve Poisson equation on the unit disk
%         (compare p16.m and p28.m)

% Laplacian in polar coordinates:
  N = 31; [D,r] = cheb(N); N2 = (N-1)/2; D2 = D^2;
  D1 = D2(2:N2+1,2:N2+1); D2 = D2(2:N2+1,N:-1:N2+2);
  E1 =  D(2:N2+1,2:N2+1); E2 =  D(2:N2+1,N:-1:N2+2);
  M = 40; dt = 2*pi/M; t = dt*(1:M)'; M2 = M/2;
  D2t = toeplitz([-pi^2/(3*dt^2)-1/6 ...
                   .5*(-1).^(2:M)./sin(dt*(1:M-1)/2).^2]);
  R = diag(1./r(2:N2+1)); Z = zeros(M2); I = eye(M2);
  L = kron(D1+R*E1,eye(M))+kron(D2+R*E2,[Z I;I Z])+kron(R^2,D2t);

% Right-hand side and solution for u:
  [rr,tt] = meshgrid(r(2:N2+1),t); rr = rr(:); tt = tt(:);
  f = -rr.^2.*sin(tt/2).^4 + sin(6*tt).*cos(tt/2).^2; u = L\f;

% Reshape results onto 2D grid and plot them:
  u = reshape(u,M,N2); u = [zeros(M+1,1) u([M 1:M],:)];
  [rr,tt] = meshgrid(r(1:N2+1),t([M 1:M]));
  [xx,yy] = pol2cart(tt,rr);
  clf, subplot('position',[.1 .4 .8 .5])
  mesh(xx,yy,u), view(20,40), colormap(1e-6*[1 1 1]);
  axis([-1 1 -1 1 -.01 .05]), xlabel x, ylabel y, zlabel u 
% p3.m - band-limited interpolation

  h = 1; xmax = 10; clf
  x = -xmax:h:xmax;                     % computational grid
  xx = -xmax-h/20:h/10:xmax+h/20;       % plotting grid
  for plt = 1:3
    subplot(4,1,plt)
    switch plt
      case 1, v = (x==0);               % delta function
      case 2, v = (abs(x)<=3);          % square wave
      case 3, v = max(0,1-abs(x)/3);    % hat function
    end
    plot(x,v,'.','markersize',14), grid on
    p = zeros(size(xx));
    for i = 1:length(x),
      p = p + v(i)*sin(pi*(xx-x(i))/h)./(pi*(xx-x(i))/h);
    end
    line(xx,p), axis([-xmax xmax -.5 1.5])
    set(gca,'xtick',[]), set(gca,'ytick',[0 1])
  end
% p30.m - spectral integration, ODE style (compare p12.m)

% Computation: various values of N, four functions:
  Nmax = 50; E = zeros(4,Nmax); clf
  for N = 1:Nmax; i = 1:N;
    [D,x] = cheb(N); x = x(i); Di = inv(D(i,i)); w = Di(1,:);
    f = abs(x).^3;     E(1,N) = abs(w*f - .5);
    f = exp(-x.^(-2)); E(2,N) = abs(w*f - ...
                           2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
    f = 1./(1+x.^2);   E(3,N) = abs(w*f - pi/2);
    f = x.^10;         E(4,N) = abs(w*f - 2/11);
  end

% Plot results:
  labels = {'|x|^3','exp(-x^{-2})','1/(1+x^2)','x^{10}'};
  for iplot = 1:4, 
    subplot(3,2,iplot)
    semilogy(E(iplot,:)+1e-100,'.','markersize',12), hold on
    plot(E(iplot,:)+1e-100)
    axis([0 Nmax 1e-18 1e3]), grid on
    set(gca,'xtick',0:10:Nmax,'ytick',(10).^(-15:5:0))
    ylabel error, text(32,.004,labels(iplot))
  end
% p30b.m - spectral integration, Clenshaw-Curtis style (compare p30.m)

% Computation: various values of N, four functions:
  Nmax = 50; E = zeros(4,Nmax); clf
  for N = 1:Nmax;
    [x,w] = clencurt(N);
    f = abs(x).^3;     E(1,N) = abs(w*f - .5);
    f = exp(-x.^(-2)); E(2,N) = abs(w*f - ...
                           2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
    f = 1./(1+x.^2);   E(3,N) = abs(w*f - pi/2);
    f = x.^10;         E(4,N) = abs(w*f - 2/11);
  end

% Plot results:
  labels = {'|x|^3','exp(-x^{-2})','1/(1+x^2)','x^{10}'};
  for iplot = 1:4, 
    subplot(3,2,iplot)
    semilogy(E(iplot,:)+1e-100,'.','markersize',12), hold on
    semilogy(E(iplot,:)+1e-100)
    axis([0 Nmax 1e-18 1e3]), grid on
    set(gca,'xtick',0:10:Nmax,'ytick',(10).^(-15:5:0))
    ylabel error, text(32,.004,labels(iplot))
  end
% p30c.m - spectral integration, Gauss style (compare p30.m)

% Computation: various values of N, four functions:
  Nmax = 50; E = zeros(4,Nmax); clf
  for N = 1:Nmax;
    [x,w] = gauss(N);
    f = abs(x).^3;     E(1,N) = abs(w*f - .5);
    f = exp(-x.^(-2)); E(2,N) = abs(w*f - ...
                           2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
    f = 1./(1+x.^2);   E(3,N) = abs(w*f - pi/2);
    f = x.^10;         E(4,N) = abs(w*f - 2/11);
  end

% Plot results:
  labels = {'|x|^3','exp(-x^{-2})','1/(1+x^2)','x^{10}'};
  for iplot = 1:4, 
    subplot(3,2,iplot)
    semilogy(E(iplot,:)+1e-100,'.','markersize',12), hold on
    semilogy(E(iplot,:)+1e-100)
    axis([0 Nmax 1e-18 1e3]), grid on
    set(gca,'xtick',0:10:Nmax,'ytick',(10).^(-15:5:0))
    ylabel error, text(32,.004,labels(iplot))
  end
% p31.m - gamma function via complex integral, trapezoid rule

  N = 70; theta = -pi + (2*pi/N)*(.5:N-.5)';
  c = -11;                     % center of circle of integration
  r = 16;                      % radius of circle of integration
  x = -3.5:.1:4; y = -2.5:.1:2.5;
  [xx,yy] = meshgrid(x,y); zz = xx + 1i*yy; gaminv = 0*zz;
  for i = 1:N          
    t = c + r*exp(1i*theta(i));
    gaminv = gaminv + exp(t)*t.^(-zz)*(t-c);
  end
  gaminv = gaminv/N; gam = 1./gaminv; clf, mesh(xx,yy,abs(gam))
  axis([-3.5 4 -2.5 2.5 0 6]), xlabel Re(z), ylabel Im(z)
  text(4,-1.4,5.5,'|\Gamma(z)|','fontsize',20), colormap(1e-6*[1 1 1]);
% p32.m - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.m)

  N = 16;
  [D,x] = cheb(N);
  D2 = D^2;
  D2 = D2(2:N,2:N);
  f = exp(4*x(2:N));           
  u = D2\f;       
  u = [0;u;0] + (x+1)/2;              
  clf
  subplot('position',[.1 .4 .8 .5])
  plot(x,u,'.','markersize',16)
  xx = -1:.01:1;
  uu = polyval(polyfit(x,u,N),xx);
  line(xx,uu), grid on
  exact = (exp(4*xx) - sinh(4)*xx - cosh(4))/16 + (xx+1)/2;
  title(['max err = ' num2str(norm(uu-exact,inf))],'fontsize',12)
% p33.m - solve linear BVP u_xx = exp(4x), u'(-1)=u(1)=0

  N = 16;
  [D,x] = cheb(N); D2 = D^2;              
  D2(N+1,:) = D(N+1,:);            % Neumann condition at x = -1
  D2 = D2(2:N+1,2:N+1);                
  f = exp(4*x(2:N));           
  u = D2\[f;0];                       
  u = [0;u];              
  clf, subplot('position',[.1 .4 .8 .5])
  plot(x,u,'.','markersize',16)
  axis([-1 1 -4 0])
  xx = -1:.01:1;
  uu = polyval(polyfit(x,u,N),xx);   
  line(xx,uu)
  grid on
  exact = (exp(4*xx) - 4*exp(-4)*(xx-1) - exp(4))/16; 
  title(['max err = ' num2str(norm(uu-exact,inf))],'fontsize',12)
% p34.m - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
%         (compare p6.m and p32.m)

% Differentiation matrix and initial data:
  N = 20; [D,x] = cheb(N); D2 = D^2;     % use full-size matrix
  D2([1 N+1],:) = zeros(2,N+1);          % for convenience
  eps = 0.01; dt = min([.01,50*N^(-4)/eps]);
  t = 0; v = .53*x + .47*sin(-1.5*pi*x);

% Solve PDE by Euler formula and plot results:
  tmax = 100; tplot = 2; nplots = round(tmax/tplot);
  plotgap = round(tplot/dt); dt = tplot/plotgap;
  xx = -1:.025:1; vv = polyval(polyfit(x,v,N),xx);
  plotdata = [vv; zeros(nplots,length(xx))]; tdata = t;
  for i = 1:nplots
    for n = 1:plotgap
      t = t+dt; v = v + dt*(eps*D2*(v-x) + v - v.^3);    % Euler
    end
    vv = polyval(polyfit(x,v,N),xx);
    plotdata(i+1,:) = vv; tdata = [tdata; t];
  end
  clf, subplot('position',[.1 .4 .8 .5])
  mesh(xx,tdata,plotdata), grid on, axis([-1 1 0 tmax -1 1]),
  view(-60,55), colormap(1e-6*[1 1 1]); xlabel x, ylabel t, zlabel u
% p35.m - Allen-Cahn eq. as in p34.m, but with boundary condition
%         imposed explicitly ("method (II)")

% Differentiation matrix and initial data:
  N = 20; [D,x] = cheb(N); D2 = D^2;    
  eps = 0.01; dt = min([.01,50*N^(-4)/eps]);
  t = 0; v = .53*x + .47*sin(-1.5*pi*x);

% Solve PDE by Euler formula and plot results:
  tmax = 100; tplot = 2; nplots = round(tmax/tplot);
  plotgap = round(tplot/dt); dt = tplot/plotgap;
  xx = -1:.025:1; vv = polyval(polyfit(x,v,N),xx);
  plotdata = [vv; zeros(nplots,length(xx))]; tdata = t;
  for i = 1:nplots
    for n = 1:plotgap
      t = t+dt; v = v + dt*(eps*D2*v + v - v.^3);        % Euler
      v(1) = 1 + sin(t/5)^2;  v(end) = -1;               % BC
    end
    vv = polyval(polyfit(x,v,N),xx);
    plotdata(i+1,:) = vv; tdata = [tdata; t];
  end
  clf, subplot('position',[.1 .4 .8 .5])
  mesh(xx,tdata,plotdata), grid on, axis([-1 1 0 tmax -1 2]),
  view(-60,55), colormap(1e-6*[1 1 1]); xlabel x, ylabel t, zlabel u
% p36.m - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs

% Set up grid and 2D Laplacian, boundary points included:
  N = 24; [D,x] = cheb(N); y = x;
  [xx,yy] = meshgrid(x,y); xx = xx(:); yy = yy(:);        
  D2 = D^2; I = eye(N+1); L = kron(I,D2) + kron(D2,I);     

% Impose boundary conditions by replacing appropriate rows of L:
  b = find(abs(xx)==1 | abs(yy)==1);            % boundary pts
  L(b,:) = zeros(4*N,(N+1)^2); L(b,b) = eye(4*N);
  rhs = zeros((N+1)^2,1);
  rhs(b) = (yy(b)==1).*(xx(b)<0).*sin(pi*xx(b)).^4 + ...
           .2*(xx(b)==1).*sin(3*pi*yy(b));

% Solve Laplace equation, reshape to 2D, and plot:
  u = L\rhs; uu = reshape(u,N+1,N+1);
  [xx,yy] = meshgrid(x,y);
  [xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
  uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
  clf, subplot('position',[.1 .4 .8 .5])
  mesh(xxx,yyy,uuu), colormap(1e-6*[1 1 1]);
  axis([-1 1 -1 1 -.2 1]), view(-20,45)
  text(0,.8,.4,sprintf('u(0,0) = %12.10f',uu(N/2+1,N/2+1)))
% p37.m - 2D "wave tank" with Neumann BCs for |y|=1

% x variable in [-A,A], Fourier:
  A = 3; Nx = 50; dx = 2*A/Nx; x = -A+dx*(1:Nx)';
  D2x = (pi/A)^2*toeplitz([-1/(3*(dx/A)^2)-1/6 ...
           .5*(-1).^(2:Nx)./sin((pi*dx/A)*(1:Nx-1)/2).^2]);

% y variable in [-1,1], Chebyshev:
  Ny = 15; [Dy,y] = cheb(Ny); D2y = Dy^2;
  BC = -Dy([1 Ny+1],[1 Ny+1])\Dy([1 Ny+1],2:Ny);

% Grid and initial data:
  [xx,yy] = meshgrid(x,y);
  vv = exp(-8*((xx+1.5).^2+yy.^2));
  dt = 5/(Nx+Ny^2); 
  vvold = exp(-8*((xx+dt+1.5).^2+yy.^2));

% Time-stepping by leap frog formula:
  clf, plotgap = round(2/dt); dt = 2/plotgap;
  for n = 0:2*plotgap
    t = n*dt;
    if rem(n+.5,plotgap)<1
      subplot(3,1,n/plotgap+1), mesh(xx,yy,vv), view(-10,60)
      axis([-A A -1 1 -0.15 1]), colormap(1e-6*[1 1 1]);
      text(-2.5,1,.5,['t = ' num2str(t)],'fontsize',18),
      set(gca,'ztick',[]), grid off, drawnow
    end
    vvnew = 2*vv - vvold + dt^2*(vv*D2x +D2y*vv);
    vvold = vv; vv = vvnew;
    vv([1 Ny+1],:) = BC*vv(2:Ny,:);       % Neumann BCs for |y|=1
  end
% p38.m - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
%         (compare p13.m)

% Construct discrete biharmonic operator:
  N = 15; [D,x] = cheb(N);
  S = diag([0; 1 ./(1-x(2:N).^2); 0]);
  D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
  D4 = D4(2:N,2:N);

% Solve boundary-value problem and plot result:
  f = exp(x(2:N)); u = D4\f; u = [0;u;0];
  clf, subplot('position',[.1 .4 .8 .5])
  plot(x,u,'.','markersize',16)
  axis([-1 1 -.01 .06]), grid on
  xx = (-1:.01:1)';
  uu = (1-xx.^2).*polyval(polyfit(x,S*u,N),xx);           
  line(xx,uu)

% Determine exact solution and print maximum error:
  A = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3];
  V = vander(xx); V = V(:,end:-1:end-3);
  c = A\exp([-1 -1 1 1]'); exact = exp(xx) - V*c;
  title(['max err = ' num2str(norm(uu-exact,inf))],'fontsize',12)
% p39.m - eigenmodes of biharmonic on a square with clamped BCs
%         (compare p38.m)

% Construct spectral approximation to biharmonic operator:
  N = 17; [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
  S = diag([0; 1 ./(1-x(2:N).^2); 0]);
  D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
  D4 = D4(2:N,2:N); I = eye(N-1);
  L = kron(I,D4) + kron(D4,I) + 2*kron(D2,I)*kron(I,D2);

% Find and plot 25 eigenmodes:
  [V,Lam] = eig(-L); Lam = -real(diag(Lam));
  [Lam,ii] = sort(Lam); ii = ii(1:25); V = real(V(:,ii));
  Lam = sqrt(Lam/Lam(1));
  [xx,yy] = meshgrid(x,x); 
  [xxx,yyy] = meshgrid(-1:.01:1,-1:.01:1);
  [ay,ax] = meshgrid(.8:-.2:0,0:.16:.64);
  sq = [1+1i -1+1i -1-1i 1-1i 1+1i]; clf
  for i = 1:25
    uu = zeros(N+1,N+1); uu(2:N,2:N) = reshape(V(:,i),N-1,N-1);
    subplot('position',[ax(i) ay(i) .16 .2]), plot(sq)
    uuu = interp2(xx,yy,uu,xxx,yyy,'cubic');
    hold on, contour(xxx,yyy,uuu,[0 0]), axis square
    axis (1.25*[-1 1 -1 1]), axis off, colormap(1e-6*[1 1 1]);
    text(-.3,1.15,num2str(Lam(i)),'fontsize',7)
  end
% p4.m - periodic spectral differentiation

% Set up grid and differentiation matrix:
  N = 24; h = 2*pi/N; x = h*(1:N)';
  column = [0 .5*(-1).^(1:N-1).*cot((1:N-1)*h/2)]';
  D = toeplitz(column,column([1 N:-1:2]));

% Differentiation of a hat function:
  v = max(0,1-abs(x-pi)/2); clf
  subplot(3,2,1), plot(x,v,'.-','markersize',13)
  axis([0 2*pi -.5 1.5]), grid on, title('function')
  subplot(3,2,2), plot(x,D*v,'.-','markersize',13)
  axis([0 2*pi -1 1]), grid on, title('spectral derivative')

% Differentiation of exp(sin(x)):
  v = exp(sin(x)); vprime = cos(x).*v;
  subplot(3,2,3), plot(x,v,'.-','markersize',13)
  axis([0 2*pi 0 3]), grid on
  subplot(3,2,4), plot(x,D*v,'.-','markersize',13)
  axis([0 2*pi -2 2]), grid on
  error = norm(D*v-vprime,inf);
  text(2.2,1.4,['max error = ' num2str(error)])
% p40.m - eigenvalues of Orr-Sommerfeld operator (compare p38.m)

  R = 5772; clf, [ay,ax] = meshgrid([.56 .04],[.1 .5]);
  for N = 40:20:100

    % 2nd- and 4th-order differentiation matrices:
    [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
    S = diag([0; 1 ./(1-x(2:N).^2); 0]);
    D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
    D4 = D4(2:N,2:N);

    % Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);
    A = (D4-2*D2+I)/R - 2i*I - 1i*diag(1-x(2:N).^2)*(D2-I);
    B = D2-I;
    ee = eig(A,B);
    i = N/20-1; subplot('position',[ax(i) ay(i) .38 .38])
    plot(ee,'.','markersize',12)
    grid on, axis([-.8 .2 -1 0]), axis square
    title(['N = ' int2str(N) '    \lambda_{max} = ' ...
        num2str(max(real(ee)),'%16.12f')]), drawnow
  end
% p5.m - repetition of p4.m via FFT
%        For complex v, delete "real" commands.

% Differentiation of a hat function:
  N = 24; h = 2*pi/N; x = h*(1:N)';
  v = max(0,1-abs(x-pi)/2); v_hat = fft(v);
  w_hat = 1i*[0:N/2-1 0 -N/2+1:-1]' .* v_hat;
  w = real(ifft(w_hat)); clf
  subplot(3,2,1), plot(x,v,'.-','markersize',13)
  axis([0 2*pi -.5 1.5]), grid on, title('function')
  subplot(3,2,2), plot(x,w,'.-','markersize',13)
  axis([0 2*pi -1 1]), grid on, title('spectral derivative')

% Differentiation of exp(sin(x)):
  v = exp(sin(x)); vprime = cos(x).*v;
  v_hat = fft(v);
  w_hat = 1i*[0:N/2-1 0 -N/2+1:-1]' .* v_hat;
  w = real(ifft(w_hat));
  subplot(3,2,3), plot(x,v,'.-','markersize',13)
  axis([0 2*pi 0 3]), grid on
  subplot(3,2,4), plot(x,w,'.-','markersize',13)
  axis([0 2*pi -2 2]), grid on
  error = norm(w-vprime,inf);
  text(2.2,1.4,['max error = ' num2str(error)])
% p6.m - variable coefficient wave equation

% Grid, variable coefficient, and initial data:
  N = 128; h = 2*pi/N; x = h*(1:N); t = 0; dt = h/4;
  c = .2 + sin(x-1).^2;
  v = exp(-100*(x-1).^2); vold = exp(-100*(x-.2*dt-1).^2);

% Time-stepping by leap frog formula:
  tmax = 8; tplot = .15; clf, drawnow, set(gcf,'renderer','zbuffer')
  plotgap = round(tplot/dt); dt = tplot/plotgap;
  nplots = round(tmax/tplot);
  data = [v; zeros(nplots,N)]; tdata = t;
  for i = 1:nplots
    for n = 1:plotgap
      t = t+dt;
      v_hat = fft(v);
      w_hat = 1i*[0:N/2-1 0 -N/2+1:-1] .* v_hat;
      w = real(ifft(w_hat)); 
      vnew = vold - 2*dt*c.*w; vold = v; v = vnew;
    end
    data(i+1,:) = v; tdata = [tdata; t];
  end
  waterfall(x,tdata,data), view(10,70), colormap(1e-6*[1 1 1]);
  axis([0 2*pi 0 tmax 0 5]), ylabel t, zlabel u, grid off
% p6u.m - variable coefficient wave equation - UNSTABLE VARIANT

% Grid, variable coefficient, and initial data:
  N = 128; h = 2*pi/N; x = h*(1:N);
  c = .2 + sin(x-1).^2;
  t = 0; dt = 1.9/N;
  v = exp(-100*(x-1).^2);
  vold = exp(-100*(x-.2*dt-1).^2);

% Time-stepping by leap frog formula:
  tmax = 8; tplot = .15; clf, set(gcf,'renderer','zbuffer')
  plotgap = round(tplot/dt);
  nplots = round(tmax/tplot);
  data = [v; zeros(nplots,N)]; tdata = t;
  for i = 1:nplots
    for n = 1:plotgap
      t = t+dt;
      v_hat = fft(v);
      w_hat = 1i*[0:N/2-1 0 -N/2+1:-1] .* v_hat;
      w = real(ifft(w_hat)); 
      vnew = vold - 2*dt*c.*w;        % leap frog formula
      vold = v; v = vnew;
    end
    data(i+1,:) = v; tdata = [tdata; t];
    if max(abs(v)>2.5), data = data(1:i+1,:); break, end
  end

% Plot results:
  waterfall(x,tdata,data)
  axis([0 2*pi 0 tmax -3. 3.]), view(10,70), grid off
  colormap(1e-6*[1 1 1]); xlabel x, ylabel t, zlabel u
% p7.m - accuracy of periodic spectral differentiation

% Compute derivatives for various values of N:
  Nmax = 50; E = zeros(3,Nmax/2-2);
  for N = 6:2:Nmax;
    h = 2*pi/N; x = h*(1:N)';
    column = [0 .5*(-1).^(1:N-1).*cot((1:N-1)*h/2)]';
    D = toeplitz(column,column([1 N:-1:2]));
    v = abs(sin(x)).^3;                     % 3rd deriv in BV
    vprime = 3*sin(x).*cos(x).*abs(sin(x));
    E(1,N/2-2) = norm(D*v-vprime,inf);
    v = exp(-sin(x/2).^(-2));               % C-infinity
    vprime = .5*v.*sin(x)./sin(x/2).^4;
    E(2,N/2-2) = norm(D*v-vprime,inf);
    v = 1./(1+sin(x/2).^2);                 % analytic in a strip
    vprime = -sin(x/2).*cos(x/2).*v.^2;
    E(3,N/2-2) = norm(D*v-vprime,inf);
    v = sin(10*x); vprime = 10*cos(10*x);   % band-limited
    E(4,N/2-2) = norm(D*v-vprime,inf);
  end

% Plot results:
  titles = {'|sin(x)|^3','exp(-sin^{-2}(x/2))',...
       '1/(1+sin^2(x/2))','sin(10x)'}; clf
  for iplot = 1:4 
    subplot(2,2,iplot)
    semilogy(6:2:Nmax,E(iplot,:),'.','markersize',12)
    line(6:2:Nmax,E(iplot,:))
    axis([0 Nmax 1e-16 1e3]), grid on
    set(gca,'xtick',0:10:Nmax,'ytick',(10).^(-15:5:0))
    xlabel N, ylabel error, title(titles(iplot))
  end
% p8.m - eigenvalues of harmonic oscillator -u"+x^2 u on R

  format long, format compact
  L = 8;                             % domain is [-L L], periodic
  for N = 6:6:36
    h = 2*pi/N; x = h*(1:N); x = L*(x-pi)/pi;
    column = [-pi^2/(3*h^2)-1/6 ...
       -.5*(-1).^(1:N-1)./sin(h*(1:N-1)/2).^2];
    D2 = (pi/L)^2*toeplitz(column);  % 2nd-order differentiation
    eigenvalues = sort(eig(-D2 + diag(x.^2)));
    N, eigenvalues(1:4)
  end
% p9.m - polynomial interpolation in equispaced and Chebyshev pts

  N = 16;
  xx = -1.01:.005:1.01; clf
  for i = 1:2
    if i==1, s = 'equispaced points'; x = -1 + 2*(0:N)/N; end
    if i==2, s = 'Chebyshev points';  x = cos(pi*(0:N)/N); end
    subplot(2,2,i)
    u = 1./(1+16*x.^2);
    uu = 1./(1+16*xx.^2);
    p = polyfit(x,u,N);              % interpolation
    pp = polyval(p,xx);              % evaluation of interpolant
    plot(x,u,'.','markersize',13)
    line(xx,pp)
    axis([-1.1 1.1 -1 1.5]), title(s)
    error = norm(uu-pp,inf);
    text(-.5,-.5,['max error = ' num2str(error)])
  end
