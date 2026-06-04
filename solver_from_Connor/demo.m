%%  GENERATE DATA
clear; close all; clc; 

% Grid (must use ndgrid)
ds = 0.01;
xmin = -1; xmax = 1; ymin = -1; ymax = 1;
[X,Y] = ndgrid(xmin:ds:xmax, ymin:ds:ymax);
[nx,ny] = size(X);

% Clean Data (must use ndgrid gradient for g)
L0 = 1; 
x0 = -0.5; y0 = 0;
r = ((X-x0).^2+(Y-y0).^2).^(1/2);
p = -exp((-r.^2)/(L0^2));
[gx,gy] = ndgrid_gradient(p,ds,ds); 

% Artificial Noisy Data 
var_x = 1; var_y = 1;
bias_x = 0; bais_y = 0;
gx_noisy = gx + sqrt(var_x)*randn(nx,ny) + bias_x;
gy_noisy = gy + sqrt(var_y)*randn(nx,ny) + bais_y;

% Domain Mask
domain_mask = true(nx,ny);

% Solve system
tic()
[P,A,b] = poisson_solver(ds,gx_noisy,gy_noisy,domain_mask);
toc()

% Shift Solution
P = P - P(1,1) + p(1,1);

%% Plotting

min_p = min(p(:)) - 0.25;
max_p = max(p(:)) + 0.25;
solAxisLimts = [min_p,max_p];
levels_p = 10;

min_e = min(p(:)-P(:));
max_e = max(p(:)-P(:));
levels_e = 10;

tiledlayout(1,3,"Padding","compact")
nexttile()
contourf(X,Y,p,levels_p)
title("Ground Truth")
xlabel("X")
ylabel("Y")
clim(solAxisLimts)
colorbar
axis equal

nexttile()
contourf(X,Y,P,levels_p)
title("Solution")
xlabel("X")
ylabel("Y")
clim(solAxisLimts)
colorbar
axis equal

nexttile()
contourf(X,Y,p-P,levels_e)
title("Error")
xlabel("X")
ylabel("Y")
colorbar
axis equal

set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.3, 0.8, 0.45]); 
