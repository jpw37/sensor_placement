% Gradient for ndgrid data
function [gx_ndgrid,gy_ndgrid] = ndgrid_gradient(h,dx,dy)
    
    % Default matlab gradient function is programmed for a meshgrid

    % convert ndgrid to meshgrid  
    h_meshgrid = pagetranspose(h);
  
    % compute gradient
    [gx_meshgrid,gy_meshgrid] = gradient(h_meshgrid,dx,dy);
    
    % convert back to nd grid
    gx_ndgrid = pagetranspose(gx_meshgrid);
    gy_ndgrid = pagetranspose(gy_meshgrid);
    
end