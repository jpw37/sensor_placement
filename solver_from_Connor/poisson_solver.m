function [p,A,b] = poisson_solver(ds,gx,gy,domain_mask)
    
    [nx,ny] = size(gx);
    
    domain_mask_flat = domain_mask(:);

    array_index = reshape(1:(nx*ny),[nx,ny]);
    array_index = array_index(domain_mask_flat);
   
    n = nnz(domain_mask);

    index = zeros(nx,ny,"int32");
    index(domain_mask) = 1:n;
    
    N = [index(:,2:end),zeros(nx,1,"int32")];  
    E = [index(2:end,:);zeros(1,ny,"int32")];    
    S = [zeros(nx,1,"int32"),index(:,1:end-1)];  
    W = [zeros(1,ny,"int32");index(1:end-1,:)];  

    N = N(domain_mask_flat);
    E = E(domain_mask_flat);
    S = S(domain_mask_flat);
    W = W(domain_mask_flat);
    
    has_N_node = N~=0;
    has_E_node = E~=0;
    has_S_node = S~=0;
    has_W_node = W~=0;

    has_N_node_array_index = array_index(has_N_node);
    has_E_node_array_index = array_index(has_E_node);
    has_S_node_array_index = array_index(has_S_node);
    has_W_node_array_index = array_index(has_W_node);

    has_N_node_N_array_index = array_index(N(has_N_node));
    has_E_node_E_array_index = array_index(E(has_E_node));
    has_S_node_S_array_index = array_index(S(has_S_node));
    has_W_node_W_array_index = array_index(W(has_W_node));
    
    linear_index = int32((1:n)');

    A_row = [linear_index;
             linear_index(has_N_node);
             linear_index(has_E_node);
             linear_index(has_S_node);
             linear_index(has_W_node)];

    A_col = [linear_index;
             N(has_N_node);
             E(has_E_node);
             S(has_S_node);
             W(has_W_node)];

    A_value = [sum([N,E,S,W]~=0,2);
               -ones(nnz(N),1);
               -ones(nnz(E),1);
               -ones(nnz(S),1);
               -ones(nnz(W),1)];

    A = sparse(A_row,A_col,A_value,n,n);
   
    b = zeros(n,1);
    b(has_N_node) = b(has_N_node) - 0.5*ds*(gy(has_N_node_array_index) + gy(has_N_node_N_array_index));
    b(has_E_node) = b(has_E_node) - 0.5*ds*(gx(has_E_node_array_index) + gx(has_E_node_E_array_index));
    b(has_S_node) = b(has_S_node) + 0.5*ds*(gy(has_S_node_array_index) + gy(has_S_node_S_array_index));
    b(has_W_node) = b(has_W_node) + 0.5*ds*(gx(has_W_node_array_index) + gx(has_W_node_W_array_index));
    
    p = nan(nx,ny);
    p(domain_mask_flat) = A\b;


end

