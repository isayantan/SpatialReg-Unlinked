import torch
import torch.optim as optim
from elbo_stick_breaking import vi_piX, vi_piS
from tqdm import tqdm
from utils import nearest_pd_torch, compute_q_phi, round_to_perm

def VIGP_Unlinked(n_iter,
            n_blocks,
            n_locations, 
            X, Y, Dist, 
            n_steps=10, 
            phi_init=3,
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
            pi_X_true = None,
            pi_S_true = None,
            VX_ub= 2,
            VS_ub= 2,
            lr_piX=0.1,
            lr_piS=0.1,
            prior_parameters = {}
            ):
    
    # Prior hyperparameters
    a1 = prior_parameters["a1"]
    b1 = prior_parameters["b1"]
    a2 = prior_parameters["a2"]
    b2 = prior_parameters["b2"]
    eta_X_sq = prior_parameters["eta_X_sq"]
    eta_S_sq = prior_parameters["eta_S_sq"]
    mu_beta = prior_parameters["mu_beta"]
    sigmasq_beta = prior_parameters["sigmasq_beta"]
    phi_prior_lb = prior_parameters["phi_prior_lb"]
    phi_prior_ub = prior_parameters["phi_prior_ub"]
    # Phi_max = torch.sqrt(2)
    
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
    model_piS = vi_piS(n_locations=n_locations, n_blocks=n_blocks)

    # # Initialize parameters
    M_X_star = model_piX.current_M_X_star.clone().data 
    V_X_star = model_piX.current_V_X_star.clone().data
    M_S_star = model_piS.current_M_S_star.clone().data
    V_S_star = model_piS.current_V_S_star.clone().data


    # Define the optimizer
    optimizer_piX = optim.AdamW(model_piX.parameters(), lr=lr_piX, weight_decay=1e-2)  
    optimizer_piS = optim.AdamW(model_piS.parameters(), lr=lr_piS, weight_decay=1e-2)

     
    for iter in tqdm(range(n_iter)):
        # compute sigmasq_lambda_beta
        if(fix_sigmasq_lambda_beta == False):
            X_V_X_star_X = torch.einsum('bi,ij,bj->b', X, V_X_star, X).sum()
            sigmasq_lambda_beta = 1.0 / (((lambda_a2 * X_V_X_star_X) / lambda_b2) + (1.0 / sigmasq_beta))
        else: 
            sigmasq_lambda_beta = sigmasq_lambda_beta_fixed
        
        # compute mu_lambda_beta
        if(fix_mu_lambda_beta == False):
            residual = Y - (M_S_star @ mu_W.T).T
            X_M_X_star_residual = torch.einsum('bi,ij,bj->b', X, M_X_star.T, residual).sum()
            mu_lambda_beta = sigmasq_lambda_beta * ((lambda_a2 / lambda_b2) * X_M_X_star_residual + mu_beta/sigmasq_beta)
        else:
            mu_lambda_beta = mu_lambda_beta_fixed
        
        # lambda_b1
        if(fix_lambda_b1 == False):
            mu_W_flat = mu_W.flatten()
            V_W = Sigma_W + torch.outer(mu_W_flat, mu_W_flat)
            lambda_b1 = 0.5 * (torch.trace(mean_Rphi_inv @ V_W)) + b1
        else:
            lambda_b1 = lambda_b1_fixed
        
                 
        if(fix_lambda_b2 == False):
            res = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
            term = torch.trace(res.T @ res)  # (1, 1)
            term += mu_lambda_beta ** 2 * torch.trace(X @ (V_X_star - M_X_star.T @ M_X_star) @ X.T)
            min_eigen_value = torch.min(torch.linalg.eigvalsh(V_X_star - M_X_star.T @ M_X_star))
            print(f"Minimum eigenvalue of V_X_star - M_X_star.T @ M_X_star: {min_eigen_value:.4e}")
            term += sigmasq_lambda_beta * torch.trace(X @ V_X_star @ X.T)
            term += torch.trace(mu_W @ (V_S_star - M_S_star.T @ M_S_star) @ mu_W.T)
            min_eigen_value = torch.min(torch.linalg.eigvalsh(V_S_star - M_S_star.T @ M_S_star))
            print(f"Minimum eigenvalue of V_S_star - M_S_star.T @ M_S_star: {min_eigen_value:.4e}")
            term += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W) 
            lambda_b2 = 0.5 * term + b2
        else:
            lambda_b2 = lambda_b2_fixed
        
      
        # compute Sigma_W
        # Sum the terms and take the inverse
        if(fix_Sigma_W == False):
            term1 = (lambda_a2 / lambda_b2) * torch.block_diag(*[V_S_star] * n_blocks)
            term2 = (lambda_a1 / lambda_b1) * mean_Rphi_inv
            Sigma_W_inv = (term1 + term2)
            Sigma_W_inv = (Sigma_W_inv + Sigma_W_inv.T)/2
            Sigma_W = torch.linalg.pinv(Sigma_W_inv, hermitian=True, rtol = 1e-4)
            Sigma_W = (Sigma_W + Sigma_W.T)/2
        else:
            Sigma_W = Sigma_W_fixed
        
        # compute mu_W        
        if(fix_mu_W == False):
            mu_W = (lambda_a2 / lambda_b2) * (Sigma_W @ (M_S_star.T @ Y.T - mu_lambda_beta * M_S_star.T @ M_X_star @ X.T).T.flatten()).reshape(n_blocks, n_locations)
        else:
            mu_W = mu_W_fixed
        
        # Compute the mean of R(phi)^-1
        # make this stable
        if(fix_mean_Rphi_inv == False):
            phi_samples = torch.rand(n_phi_samples) * (phi_prior_ub - phi_prior_lb) + phi_prior_lb
            # Calculate q_phi for each phi_sample
            q_phi_values = torch.tensor([compute_q_phi(phi, Dist, mu_W, Sigma_W, lambda_a1, lambda_b1) for phi in phi_samples])

            # Normalize the importance weights
            importance_weights = torch.nn.functional.softmax(q_phi_values, dim=0)
            
            Rphi_inv_sum = torch.zeros_like(Dist)
            phi_sum = 0 
            importance_weights_sum = 0
            for i in range(n_phi_samples):
                # Compute the weighted sum of phi_samples
                if(importance_weights[i] > 10e-5):
                    # Compute the weighted inverse of R(phi)
                    Rphi_inv = torch.linalg.pinv(torch.exp(-phi_samples[i] * Dist), rtol=1e-4)
                    Rphi_inv = (Rphi_inv + Rphi_inv.T)/2
                    Rphi_inv_sum += importance_weights[i] * Rphi_inv
                    importance_weights_sum += importance_weights[i]
            
                phi_sum += importance_weights[i] * phi_samples[i]
                
            mean_Rphi_inv = Rphi_inv_sum/importance_weights_sum
            mean_phi = phi_sum/importance_weights_sum
            #mean_Rphi_inv = nearest_pd_torch(mean_Rphi_inv)
        else:
            mean_Rphi_inv = mean_Rphi_inv_fixed 
            mean_phi = phi_init   
        # print("mean_Rphi_inv:", mean_Rphi_inv)

        
        if(fix_piX== False):
            # Minimize the model for piX
            for step in range(n_steps):
                optimizer_piX.zero_grad()  # Clear gradients
                loss = model_piX(Y, X, mu_lambda_beta, sigmasq_lambda_beta, M_S_star, mu_W, eta_X_sq, lambda_a2, lambda_b2, tau_X, 
                                n_piX_sample,VX_ub= VX_ub, seed=seed)
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
                    
                # Stopping rule: Stop if the loss change is below a threshold
                if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                    print(f"Stopping early at step {step} due to minimal loss change.")
                    break
                prev_loss = loss.item()
            
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
                                mu_W, Sigma_W, eta_S_sq, tau_S, n_piS_sample, VS_ub = VS_ub, 
                                seed=seed)
                loss.backward()  # Compute gradients
                optimizer_piS.step()  # Update parameters

                # # Print loss every step (or change 1 to 100 if you want sparser output)
                # if step % 1 == 0:
                #     print(f"Step {step}, Loss: {loss.item()}")
                #     print("Current MS:")
                #     print(torch.exp(model_piS.MS.data))

                #     print("\nCurrent VS:")
                #     print(torch.exp(model_piS.VS.data))

                # Stopping rule: Stop if the loss change is below a threshold
                if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                    print(f"Stopping early at step {step} due to minimal loss change.")
                    break
                prev_loss = loss.item()
                
            #print("PiS_loss", loss.item())


        # Extract the updated parameters
        
            M_S_star = model_piS.current_M_S_star.clone().data
            V_S_star = model_piS.current_V_S_star.clone().data
        else:
            M_S_star = M_S_star_fixed
            V_S_star = V_S_star_fixed
        
        print(f"Iter {iter+1}/{n_iter} | mu_lambda_beta: {mu_lambda_beta:.4f} | \n sigmasq_lambda_beta: {sigmasq_lambda_beta:.4f} | \n lambda_a1: {lambda_a1:.4f} | lambda_b1: {lambda_b1:.4f} | lambda_a2: {lambda_a2:.4f} | lambda_b2: {lambda_b2:.4f}")
        print(f"‣  E[ϕ]: {mean_phi:.4f} | "
              f"‣ ||mu_W||: {torch.norm(mu_W):.4f}")
        if pi_X_true is not None:
            est_perm_piX= round_to_perm(M_X_star.detach().numpy())
            correct_permutations = torch.sum(torch.tensor(est_perm_piX) * pi_X_true)
            print(f"Number of correct permutations recognized for piX: {correct_permutations}")
        if pi_S_true is not None:
            est_perm_piS= round_to_perm(M_S_star.detach().numpy())
            correct_permutations = torch.sum(torch.tensor(est_perm_piS) * pi_S_true)
            print(f"Number of correct permutations recognized for piS: {correct_permutations}")
        

        resid = Y - mu_lambda_beta * (M_X_star @ X.T).T - (M_S_star @ mu_W.T).T
        total_loss = torch.trace(resid.T @ resid)  # (1, 1)
        total_loss += mu_lambda_beta ** 2 * torch.trace(X @ (V_X_star - M_X_star.T @ M_X_star) @ X.T)
        total_loss += sigmasq_lambda_beta * torch.trace(X @ V_X_star @ X.T)
        total_loss += torch.trace(mu_W @ (V_S_star - M_S_star.T @ M_S_star) @ mu_W.T)
        total_loss += torch.trace(torch.block_diag(*[V_S_star] * n_blocks) @ Sigma_W)          
        print(f"Total Loss: {torch.sqrt(total_loss/(n_locations * n_blocks)):.4f}")
        # Store the loss in a vector
        if iter == 0:
            loss_vector = torch.zeros(n_iter)
        loss_vector[iter] = total_loss/(n_locations * n_blocks)
        



        # print(f"Iteration {iter+1}/{n_iter} completed.")
        # print("mu_lambda_beta:", mu_lambda_beta)
        # print("sigmasq_lambda_beta:", sigmasq_lambda_beta)
        # print("mean_Rphi_inv:", mean_Rphi_inv)
        # print("M_X_star:", M_X_star)
        # print("V_X_star:", V_X_star)
        # print("lambda_b1", lambda_b1)
        # print("lambda_b2", lambda_b2)
        
        # Save parameters in a dictionary
        parameters = {
            "M_X_star": M_X_star,
            "mean_Rphi_inv": mean_Rphi_inv,
            "V_X_star": V_X_star,
            "M_S_star": M_S_star,
            "V_S_star": V_S_star,
            "mu_lambda_beta": mu_lambda_beta,
            "sigmasq_lambda_beta": sigmasq_lambda_beta,
            "lambda_a1": lambda_a1,
            "lambda_b1": lambda_b1,
            "lambda_a2": lambda_a2,
            "lambda_b2": lambda_b2,
            "mean_phi": mean_phi, 
            "loss_vector": loss_vector,
        }
    return parameters
        