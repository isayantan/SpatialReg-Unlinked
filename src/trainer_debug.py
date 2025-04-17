import torch
import torch.optim as optim
from elbo import vi_piX, vi_piS
from tqdm import tqdm

def nearest_pd_torch(A, epsilon=1e-6):
    """
    Find the nearest positive definite matrix to A using eigenvalue clipping.

    Args:
        A (torch.Tensor): A symmetric matrix (n x n).
        epsilon (float): Minimum eigenvalue threshold to ensure positive definiteness.

    Returns:
        torch.Tensor: A positive definite matrix close to A.
    """
    # Ensure symmetry
    A_sym = (A + A.T) / 2

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(A_sym)

    # Clip eigenvalues to be at least epsilon
    eigvals_clipped = torch.clamp(eigvals, min=epsilon)

    # Reconstruct matrix: V Λ V^T
    A_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    # Ensure symmetry again (numerical stability)
    A_pd = (A_pd + A_pd.T) / 2

    return A_pd

def compute_q_phi(phi, Dist, sig, mean, lambda_sigmasqa, lambda_sigmasqb, eps = 1e-6):
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
    R_phi = torch.exp(-phi * Dist)

    # Optionally: Ensure R_phi is positive-definite
    R_phi += eps * torch.eye(R_phi.size(0), device=R_phi.device)

    # Compute log(det(R_phi)) safely
    sign, logdet = torch.linalg.slogdet(R_phi)
    if sign <= 0:
        # Handle log of non-positive determinant safely
        logdet = torch.tensor(float('-inf'), device=R_phi.device)

    # Solve R_phi x = sig instead of inverting
    R_phi_inv_sig = torch.linalg.solve(R_phi, sig)
    trace_term = torch.trace(R_phi_inv_sig)

    # Solve R_phi x = mean
    mean_flat = mean.flatten()
    R_phi_inv_mean = torch.linalg.solve(R_phi, mean_flat)
    mu_W_term = torch.dot(mean_flat, R_phi_inv_mean)

    # Compute exponent
    exponent = -0.5 * logdet - (lambda_sigmasqa / (2 * lambda_sigmasqb)) * (trace_term + mu_W_term)

    # Return log of q_phi (numerically stable)
    return exponent  # this is log(q_phi), better for log-domain work
 
    

def trainer(n_iter,
            n_blocks,
            n_locations, 
            X, Y, Dist, 
            n_steps=10, 
            phi_init=3,
            phi_prior_ub= 10,
            n_phi_samples=100,
            n_piX_sample=10, 
            n_piS_sample=10,
            tau_X= 0.1, tau_S= 0.1,
            seed=100, fix_mu_lambda_beta=False,
            fix_sigmasq_lambda_beta=False,
            fix_lambda_b1=False,
            fix_lambda_b2=False,
            fix_piX=False,
            fix_piS=False,
            fix_mu_W=False,
            fix_Sigma_W=False,
            fix_mean_Rphi_inv=False,
            sigmasq_lambda_beta_fixed=0.1,
            mu_lambda_beta_fixed=0.1,
            lambda_b1_fixed=0.1,
            lambda_b2_fixed=0.1,
            M_X_star_fixed=None,
            V_X_star_fixed=None,
            M_S_star_fixed=None,
            V_S_star_fixed=None,
            mu_W_fixed=None,
            Sigma_W_fixed=None,
            mean_Rphi_inv_fixed=None,
            ):
    
    # Prior hyperparameters
    a1 = 0.1
    b1 = 0.1
    a2 = 0.1
    b2 = 0.1
    eta_X_sq = 0.1
    eta_S_sq = 0.1
    sigmasq_beta = 100
    # Phi_max = torch.sqrt(2)
    
    # Model parameters
    mu_lambda_beta = 0.1
    sigmasq_lambda_beta = 0.1
    mu_W = torch.zeros(n_blocks, n_locations)
    Sigma_W = torch.eye(n_blocks * n_locations) 
    R_phi = torch.exp(-phi_init * Dist)
    mean_Rphi_inv = torch.linalg.inv(nearest_pd_torch(R_phi))
    # M_S_star = (1/n_locations) * torch.ones(n_locations, n_locations)
    # M_X_star = (1/n_locations) * torch.ones(n_locations, n_locations)
    # V_S_star = torch.eye(n_locations)
    # V_X_star = torch.eye(n_locations)
    lambda_a1 = n_blocks * n_locations * 0.5 + a1
    lambda_b1 = 0.5
    lambda_a2 = n_blocks * n_locations * 0.5 + a2
    lambda_b2 = 0.5

    # Initialize the model
    model_piX = vi_piX(n_locations=n_locations)
    model_piS = vi_piS(n_locations=n_locations)

    # # Initialize parameters
    M_X_star = model_piX.current_M_X_star.clone().data 
    V_X_star = model_piX.current_V_X_star.clone().data
    M_S_star = model_piS.current_M_S_star.clone().data
    V_S_star = model_piS.current_V_S_star.clone().data


    # Define the optimizer
    optimizer_piX = optim.AdamW(model_piX.parameters(), lr=0.1, weight_decay=1e-2)  
    optimizer_piS = optim.AdamW(model_piS.parameters(), lr=0.1, weight_decay=1e-2)

     
    for iter in tqdm(range(n_iter)):
        # compute sigmasq_lambda_beta
        X_V_X_star_X = torch.einsum('bi,ij,bj->b', X, V_X_star, X).sum()
        #print("Norm of X_V_X_star_X:", torch.norm(X_V_X_star_X))
        if(fix_sigmasq_lambda_beta == False):
            sigmasq_lambda_beta = 1.0 / (((lambda_a2 * X_V_X_star_X) / lambda_b2) + (1.0 / sigmasq_beta))
        else: 
            sigmasq_lambda_beta = sigmasq_lambda_beta_fixed
        
        # compute mu_lambda_beta
        residual = Y - (M_S_star @ mu_W.T).T
        X_M_X_star_residual = torch.einsum('bi,ij,bj->b', X, M_X_star.T, residual).sum()
        #print("Norm of X_M_X_star_residual:", torch.norm(X_M_X_star_residual))
        if(fix_mu_lambda_beta == False):
            mu_lambda_beta = sigmasq_lambda_beta * (lambda_a2 / lambda_b2) * X_M_X_star_residual
        else:
            mu_lambda_beta = mu_lambda_beta_fixed
        
        # lambda_b1
        if(fix_lambda_b1 == False):
            lambda_b1 = 0.5 * (torch.trace(mean_Rphi_inv @ Sigma_W) + mu_W.flatten().T @ mean_Rphi_inv @ mu_W.flatten()) + b1
        else:
            lambda_b1 = lambda_b1_fixed
        
        # lambda_b2
        # term = torch.trace(Y.T @ Y)
        # term += (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * X_V_X_star_X
        # term += torch.einsum('bi,ij,bj->b', mu_W, V_S_star, mu_W).sum()
        # term += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
        # term -= 2 * mu_lambda_beta * X_M_X_star_residual
        # term -= 2 * torch.einsum('bi,ij,bj->b',Y, M_S_star , mu_W).sum()
        # if(fix_lambda_b2 == False):
        #     lambda_b2 = 0.5 * term + b2
        # else:
        #     lambda_b2 = lambda_b2_fixed
        
        res = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
        term = torch.trace(res.T @ res)  # (1, 1)
        term += mu_lambda_beta ** 2 * torch.trace(X @ (V_X_star - M_X_star.T @ M_X_star) @ X.T)
        min_eigen_value = torch.min(torch.linalg.eigvalsh(V_X_star - M_X_star.T @ M_X_star))
        print(f"Minimum eigenvalue of V_X_star - M_X_star.T @ M_X_star: {min_eigen_value:.4e}")
        term += sigmasq_lambda_beta * torch.trace(X @ V_X_star @ X.T)
        term += torch.trace(mu_W @ (V_S_star - M_S_star.T @ M_S_star) @ mu_W.T)
        term += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)          
        if(fix_lambda_b2 == False):
            lambda_b2 = 0.5 * term + b2
        else:
            lambda_b2 = lambda_b2_fixed
        
        # modified lambda_b2
        # MX_x = torch.einsum('ij,bj->bi', M_X_star, X)  # (B, d_y)
        # MS_muW = torch.einsum('ij,bj->bi', M_S_star, mu_W)  # (B, d_y)
        # residual = Y - mu_lambda_beta * MX_x - MS_muW  # (B, d_y)
        # term = torch.trace(residual.T @ residual)  # (1, 1)
        # term += mu_lambda_beta ** 2 * torch.einsum('bi,ij,bj->', X, V_X_star - M_X_star.T @ M_X_star, X)
        # term += sigmasq_lambda_beta * X_V_X_star_X
        # term += torch.einsum('bi,ij,bj->b', mu_W, V_S_star - M_S_star.T @ M_S_star, mu_W).sum()
        # term += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
        # if(fix_lambda_b2 == False):
        #     lambda_b2 = 0.5 * term + b2
        # else:
        #     lambda_b2 = lambda_b2_fixed
            
        
        # compute Sigma_W
        term1 = (lambda_a2 / lambda_b2) * torch.block_diag(*[V_S_star] * n_blocks)
        term2 = (lambda_a1 / lambda_b1) * mean_Rphi_inv
        # Sum the terms and take the inverse
        if(fix_Sigma_W == False):
            Sigma_W = torch.linalg.pinv(term1 + term2, rtol = 1e-3)
        else:
            Sigma_W = Sigma_W_fixed
        
        # compute mu_W        
        #print("Norm of residual2:", torch.norm((M_S_star.T @ Y.T - mu_lambda_beta * M_S_star.T @ M_X_star @ X.T).T.flatten()))

        if(fix_mu_W == False):
            mu_W = (lambda_a2 / lambda_b2) * (Sigma_W @ (M_S_star.T @ Y.T - mu_lambda_beta * M_S_star.T @ M_X_star @ X.T).T.flatten()).reshape(n_blocks, n_locations)
        else:
            mu_W = mu_W_fixed

        # print("mean_Rphi_inv:", mean_Rphi_inv)
        # print("mu_W:", mu_W)
        # print("V_S_star:", V_S_star)
        # print("Sigma_W:", Sigma_W)

                
        # print("Any NaNs in Sigma_W?", torch.isnan(Sigma_W).any())    
        # print("Any NaNs in M_S_star?", torch.isnan(M_S_star).any())  
        # print("Any NaNs in M_X_star?", torch.isnan(M_X_star).any())  
        # print("Any NaNs in mu_lambda_beta?", torch.isnan(mu_lambda_beta).any())    

            
        #compute Phi 
        # Generate phi from a uniform distribution between 1/max(Dist) and 10
        torch.manual_seed(seed=seed)  #set seed for stochastic optimzation

        phi_samples = torch.rand(n_phi_samples) * (phi_prior_ub - (1 / torch.max(Dist))) + (1 / torch.max(Dist))
        # Calculate q_phi for each phi_sample
        q_phi_values = torch.tensor([compute_q_phi(phi, Dist, Sigma_W, mu_W, lambda_a1, lambda_b1) for phi in phi_samples])

        # Normalize the importance weights
        importance_weights = torch.nn.functional.softmax(q_phi_values, dim=0)
        
        Rphi_inv_sum = torch.zeros_like(Dist)
        for i in range(n_phi_samples):
            # Compute the weighted sum of phi_samples
            Rphi_inv = torch.linalg.pinv(torch.exp(-phi_samples[i] * Dist), rtol=1e-3)
            Rphi_inv_sum += importance_weights[i] * Rphi_inv
            
        # Compute the mean of R(phi)^-1
        # make this stable
        if(fix_mean_Rphi_inv == False):
            mean_Rphi_inv = nearest_pd_torch(mean_Rphi_inv)
        else:
            mean_Rphi_inv = mean_Rphi_inv_fixed    
        # print("mean_Rphi_inv:", mean_Rphi_inv)

        
        if(fix_piX== False):
            # Minimize the model for piX
            for step in range(n_steps):
                optimizer_piX.zero_grad()  # Clear gradients
                loss = model_piX(Y, X, mu_lambda_beta, sigmasq_lambda_beta, M_S_star, mu_W, eta_X_sq, lambda_a2, lambda_b2, tau_X, n_piX_sample,seed=seed)
                #torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
                loss.backward()  # Compute gradients
                optimizer_piX.step()  # Update parameters

                # # Print loss every 100 steps
                # if step % 1 == 0:
                #     print(f"Step {step}, Loss: {loss.item()}")
                #     print("Current MX:")
                #     print(torch.exp(model_piX.MX.data))

                #     print("\nCurrent VX:")
                #     print(torch.exp(model_piX.VX.data))
                    
                #     # Stopping rule: Stop if the loss change is below a threshold
                #     if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                #         print(f"Stopping early at step {step} due to minimal loss change.")
                #         break
                #     prev_loss = loss.item()
            
            #print("Pix_loss", loss.item())

            # Extract the updated parameters
       
            M_X_star = model_piX.current_M_X_star.clone().data 
            V_X_star = model_piX.current_V_X_star.clone().data
        else:
            M_X_star = M_X_star_fixed
            V_X_star = V_X_star_fixed

        
        if(fix_piS == False):
            # Minimize the model for piS
            for step in range(n_steps):
                optimizer_piS.zero_grad()  # Clear gradients
                loss = model_piS(Y, X, mu_lambda_beta, M_X_star, lambda_a2, lambda_b2,
                                mu_W, Sigma_W, eta_S_sq, tau_S, n_piS_sample, seed=seed)
                loss.backward()  # Compute gradients
                optimizer_piS.step()  # Update parameters

                # # Print loss every step (or change 1 to 100 if you want sparser output)
                # if step % 1 == 0:
                #     print(f"Step {step}, Loss: {loss.item()}")
                #     print("Current MS:")
                #     print(torch.exp(model_piS.MS.data))

                #     print("\nCurrent VS:")
                #     print(torch.exp(model_piS.VS.data))

                #     # Stopping rule: Stop if the loss change is below a threshold
                #     if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                #         print(f"Stopping early at step {step} due to minimal loss change.")
                #         break
                #     prev_loss = loss.item()
                
            #print("PiS_loss", loss.item())


        # Extract the updated parameters
        
            M_S_star = model_piS.current_M_S_star.clone().data
            V_S_star = model_piS.current_V_S_star.clone().data
        else:
            M_S_star = M_S_star_fixed
            V_S_star = V_S_star_fixed
        
        print(f"Iter {iter+1}/{n_iter} | mu_lambda_beta: {mu_lambda_beta:.4f} | \n sigmasq_lambda_beta: {sigmasq_lambda_beta:.4f} | \n lambda_a1: {lambda_a1:.4f} | lambda_b1: {lambda_b1:.4f} | lambda_a2: {lambda_a2:.4f} | lambda_b2: {lambda_b2:.4f}")
        norm_M_X_star = torch.norm(M_X_star)
        norm_V_X_star = torch.norm(V_X_star)
        norm_M_S_star = torch.norm(M_S_star)
        norm_V_S_star = torch.norm(V_S_star)
        norm_mean_Rphi_inv = torch.norm(mean_Rphi_inv)
        print(f"‣ ||M_X_star||: {norm_M_X_star:.4f}, ||V_X_star||: {norm_V_X_star:.4f} | "
              f"‣ ||M_S_star||: {norm_M_S_star:.4f}, ||V_S_star||: {norm_V_S_star:.4f} | "
              f"‣ ||E[R(ϕ)]⁻¹||: {norm_mean_Rphi_inv:.4f} | "
              f"‣ cond(Rphi_inv): {torch.linalg.cond(mean_Rphi_inv):.4e} | "
              f"‣ ||Sigma_W||: {torch.norm(Sigma_W):.4f} | "
              f"‣ cond(Sigma_W): {torch.linalg.cond(Sigma_W):.4e} | "
              f"‣ ||mu_W||: {torch.norm(mu_W):.4f}")
        
        total_loss = torch.trace(Y.T @ Y)
        total_loss += (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * X_V_X_star_X
        total_loss += torch.einsum('bi,ij,bj->b', mu_W, V_S_star, mu_W).sum()
        total_loss += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)
        total_loss -= 2 * mu_lambda_beta * X_M_X_star_residual
        total_loss -= 2 * torch.einsum('bi,ij,bj->b', Y, M_S_star, mu_W).sum()
        print(f"Total Loss: {torch.sqrt(total_loss/(n_locations * n_blocks)):.4f}")



        # print(f"Iteration {iter+1}/{n_iter} completed.")
        # print("mu_lambda_beta:", mu_lambda_beta)
        # print("sigmasq_lambda_beta:", sigmasq_lambda_beta)
        # print("mean_Rphi_inv:", mean_Rphi_inv)
        # print("M_X_star:", M_X_star)
        # print("V_X_star:", V_X_star)
        # print("lambda_b1", lambda_b1)
        # print("lambda_b2", lambda_b2)
    
    return mu_W, Sigma_W, M_X_star, mean_Rphi_inv, V_X_star, M_S_star, V_S_star, mu_lambda_beta, sigmasq_lambda_beta, lambda_a1, lambda_b1, lambda_a2, lambda_b2   
        