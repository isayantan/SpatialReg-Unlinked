import torch

def compute_q_phi(phi, Dist, sig, mean, lambda_sigmasqa, lambda_sigmasqb):
    """
    Compute q(phi) as defined by the given expression.

    Args:
    - phi (scalar): The tensor for which q(phi) needs to be calculated (B x d).
    - Dist (torch.Tensor): The distance matrix (nB X nB).
    - sig (torch.Tensor): The matrix Sigma_W (nB x nB).
    - mean (torch.Tensor): The vector mu_W (d,).
    - lambda_sigmasqa (float): The scalar value for lambda_a1.
    - lambda_sigmasqb (float): The scalar value for lambda_b1.
    
    Returns:
    - q_phi (torch.Tensor): The calculated q(phi).
    """
    
    # Compute R(phi)
    R_phi = torch.exp(-phi*Dist)
    
    # Step 1: Compute the inverse of R(phi)
    R_phi_inv = torch.linalg.inv(R_phi)
    
    # Step 2: Compute the determinant of R(phi)
    R_phi_det = torch.det(R_phi)
    
    # Step 3: Compute the trace term
    trace_term = torch.trace(torch.matmul(R_phi_inv,sig))
    
    # Step 4: Compute the quadratic form term (mu_W^T * R(phi)^-1 * mu_W)
    mu_W_term = torch.matmul(mean.flatten().T, torch.matmul(R_phi_inv, mean.flatten()))
    
    # Step 5: Compute the exponent
    exponent = -0.5 * torch.log(R_phi_det) - (lambda_sigmasqa / (2 * lambda_sigmasqb)) * (trace_term + mu_W_term)
    
    # Step 6: Compute the final q(phi) which is the exponential of the exponent
    q_phi = torch.exp(exponent)
    
    return q_phi
    

def trainer(n_iter,
            n_blocks,
            n_locations, 
            X, Y, Dist):
    
    # Prior hyperparameters
    a1 = 100
    b1 = 10
    a2 = 100
    b2 = 10
    eta_X_sq = 0.1
    eta_S_sq = 0.1
    sigmasq_beta = 100
    # Phi_max = torch.sqrt(2)
    
    # Model parameters
    mu_lambda_beta = 0.1
    sigmasq_lambda_beta = 0.1
    mu_W = torch.zeros(n_blocks, n_locations)
    Sigma_W = torch.eye(n_blocks * n_locations) 
    sigmasq = 0.1
    mean_Rphi_inv = torch.eye(n_locations * n_blocks)
    M_S_star = (1/n_locations) * torch.ones(n_locations, n_locations)
    M_X_star = (1/n_locations) * torch.ones(n_locations, n_locations)
    V_S_star = torch.eye(n_locations)
    V_X_star = torch.eye(n_locations)
    lambda_a1 = n_blocks * n_locations * 0.5 + a1
    lambda_b1 = 0.5
    lambda_a2 = n_blocks * n_locations * 0.5 + a2
    lambda_b2 = 0.5
     
    for iter in range(n_iter):
        # compute sigmasq_lambda_beta
        X_V_X_star_X = torch.einsum('bi,ij,bj->b', X, V_X_star, X).sum()
        sigmasq_lambda_beta = 1.0 / (lambda_a2 * X_V_X_star_X) / lambda_b2 + 1.0 / sigmasq_beta
        
        # compute mu_lambda_beta
        residual = Y - (M_S_star @ mu_W.T).T
        X_M_X_star_residual = torch.einsum('bi,ij,bj->b', X, M_X_star.T, residual).sum()
        mu_lambda_beta = sigmasq_lambda_beta * X_M_X_star_residual
        
        # lambda_b1
        lambda_b1 = 0.5 * (torch.trace(mean_Rphi_inv @ Sigma_W) + mu_W.flatten().T @ mean_Rphi_inv @ mu_W.flatten()) + b1
        
        # lambda_b2
        term = torch.trace(Y.T @ Y)
        term += (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * X_V_X_star_X
        term += torch.einsum('bi,ij,bj->b', mu_W, V_S_star, mu_W).sum()
        term += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
        term -= 2 * mu_lambda_beta * X_M_X_star_residual
        term -= 2 * torch.einsum('bi,ij,bj->b',Y, M_S_star , mu_W).sum()
        lambda_b2 = 0.5 * term + b2
        
        # compute Sigma_W
        term1 = (lambda_a2 / lambda_b2) * torch.block_diag(*[V_S_star] * n_blocks)
        term2 = (lambda_a1 / lambda_b1) * mean_Rphi_inv
        # Sum the terms and take the inverse
        Sigma_W = torch.inverse(term1 + term2)
        
        # compute mu_W        
        mu_W = (Sigma_W @ (M_S_star.T @ Y.T - mu_lambda_beta * M_S_star.T @ M_X_star @ X.T).T.flatten()).reshape(n_blocks, n_locations)
            
            
        # compute Phi 
        # Generate phi from a uniform distribution between 0 and max(Dist)
        n_phi_samples = 100
        phi_samples = torch.rand(n_phi_samples) * torch.max(Dist)
        # Calculate q_phi for each phi_sample
        q_phi_values = torch.tensor([compute_q_phi(phi, Dist, Sigma_W, mu_W, lambda_a1, lambda_b1) for phi in phi_samples])

        # Normalize the importance weights
        importance_weights = q_phi_values / q_phi_values.sum()
        
        Rphi_inv_sum = torch.zeros_like(Dist)
        for i in range(n_phi_samples):
            # Compute the weighted sum of phi_samples
            Rphi_inv = torch.linalg.inv(torch.exp(-phi_samples[i] * Dist))
            Rphi_inv_sum += importance_weights[i] * Rphi_inv
            
        # Compute the mean of R(phi)^-1
        mean_Rphi_inv = Rphi_inv_sum
        
        
        # Update piX
        optimizer_piX.zero_grad()
        loss_piX = model()
        loss_piX.backward()
        optimizer_piX.step()
        
        # Update other parameters
        
        # Update piS
        optimizer_piS.zero_grad()        
        loss_piS = model()
        loss_piS.backward()
        optimizer_piS.step()
        