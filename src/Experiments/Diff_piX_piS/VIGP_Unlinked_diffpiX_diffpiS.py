import torch
import torch.optim as optim
from elbo_diff_piX_piS import vi_piX, vi_piS
from tqdm import tqdm
from utils import nearest_pd_torch, compute_q_phi, round_to_perm

def VIGP_Unlinked(n_iter,
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
            pi_X_true = None,
            pi_S_true = None,
            VX_ub= 2,
            VS_ub= 2,
            lr_piX=0.1,
            lr_piS=0.1
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
    model_piX = [vi_piX(n_locations=n_locations, block_idx=_) for _ in range(n_blocks)]
    model_piS = [vi_piS(n_locations=n_locations, block_idx=_) for _ in range(n_blocks)]

    # # Initialize parameters
    M_X_star = [torch.tensor(m.current_M_X_star).clone().data for m in model_piX]
    V_X_star = [torch.tensor(v.current_V_X_star).clone().data for v in model_piX]
    M_S_star = [torch.tensor(m.current_M_S_star).clone().data for m in model_piS]
    V_S_star = [torch.tensor(v.current_V_S_star).clone().data for v in model_piS]


    # Define the optimizer
    optimizer_piX = [optim.AdamW(model.parameters(), lr=lr_piX, weight_decay=1e-2) for model in model_piX]
    optimizer_piS = [optim.AdamW(model.parameters(), lr=lr_piS, weight_decay=1e-2) for model in model_piS]

     
    for iter in tqdm(range(n_iter)):
        # compute sigmasq_lambda_beta
        if(fix_sigmasq_lambda_beta == False):
            X_V_X_star_X = sum(X[block_idx].T @ V_X_star[block_idx] @ X[block_idx] for block_idx in range(n_blocks))
            sigmasq_lambda_beta = 1.0 / (((lambda_a2 * X_V_X_star_X) / lambda_b2) + (1.0 / sigmasq_beta))
        else: 
            sigmasq_lambda_beta = sigmasq_lambda_beta_fixed
        
        # compute mu_lambda_beta
        if(fix_mu_lambda_beta == False):
            X_M_X_star_residual = sum(
                X[block_idx].T @ M_X_star[block_idx].T @ (Y[block_idx] - M_S_star[block_idx] @ mu_W[block_idx])
                for block_idx in range(n_blocks)
            )
            mu_lambda_beta = sigmasq_lambda_beta * (lambda_a2 / lambda_b2) * X_M_X_star_residual
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
            res = torch.stack([
                Y[block_idx] - mu_lambda_beta * (M_X_star[block_idx] @ X[block_idx]) - (M_S_star[block_idx] @ mu_W[block_idx])
                for block_idx in range(n_blocks)
            ])
            term = torch.trace(res.T @ res)  # (1, 1)
            term += mu_lambda_beta ** 2 * sum(
                (X[block_idx].T @ (V_X_star[block_idx] - M_X_star[block_idx].T @ M_X_star[block_idx]) @ X[block_idx])
                for block_idx in range(n_blocks)
            )
            min_eigen_value = torch.min(torch.cat([
                torch.linalg.eigvalsh(V_X_star[block_idx] - M_X_star[block_idx].T @ M_X_star[block_idx])
                for block_idx in range(n_blocks)
            ]))
            print(f"Minimum eigenvalue of V_X_star - M_X_star.T @ M_X_star: {min_eigen_value:.4e}")
            term += sigmasq_lambda_beta * sum(X[block_idx].T @ V_X_star[block_idx] @ X[block_idx] for block_idx in range(n_blocks))
            term += sum(
                (mu_W[block_idx].T @ (V_S_star[block_idx] - M_S_star[block_idx].T @ M_S_star[block_idx]) @ mu_W[block_idx])
                for block_idx in range(n_blocks)
            )
            min_eigen_value = torch.min(torch.cat([
                torch.linalg.eigvalsh(V_S_star[block_idx] - M_S_star[block_idx].T @ M_S_star[block_idx])
                for block_idx in range(n_blocks)
            ]))
            print(f"Minimum eigenvalue of V_S_star - M_S_star.T @ M_S_star: {min_eigen_value:.4e}")
            term += torch.trace(torch.block_diag(*[V_S_star[block_idx] for block_idx in range(n_blocks)]) @ Sigma_W)
            lambda_b2 = 0.5 * term + b2
        else:
            lambda_b2 = lambda_b2_fixed
        
      
        # compute Sigma_W
        # Sum the terms and take the inverse
        if(fix_Sigma_W == False):
            term1 = (lambda_a2 / lambda_b2) * torch.block_diag(*[V_S_star[block_idx] for block_idx in range(n_blocks)])
            term2 = (lambda_a1 / lambda_b1) * mean_Rphi_inv
            Sigma_W_inv = (term1 + term2)
            Sigma_W_inv = (Sigma_W_inv + Sigma_W_inv.T)/2
            Sigma_W = torch.linalg.pinv(Sigma_W_inv, hermitian=True, rtol = 1e-4)
            Sigma_W = (Sigma_W + Sigma_W.T)/2
        else:
            Sigma_W = Sigma_W_fixed
        
        # compute mu_W        
        if(fix_mu_W == False):
            rhs = torch.stack([
            (M_S_star[block_idx] @ Y[block_idx] - mu_lambda_beta * M_S_star[block_idx] @ M_X_star[block_idx] @ X[block_idx])
                for block_idx in range(n_blocks)
            ])  # shape: (B, n)

            rhs = rhs.flatten()  # shape: (B*n,)

            mu_W_flat = (lambda_a2 / lambda_b2) * Sigma_W @ rhs  # (B*n,)

            mu_W = mu_W_flat.reshape(n_blocks, n_locations)  # reshape into (B, n)
        else:
            mu_W = mu_W_fixed
        
        # Compute the mean of R(phi)^-1
        # make this stable
        if(fix_mean_Rphi_inv == False):
            phi_samples = torch.rand(n_phi_samples) * (phi_prior_ub - (1 / torch.max(Dist))) + (1 / torch.max(Dist))
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
            for block_idx in range(n_blocks):
                for step in range(n_steps):
                    optimizer_piX[block_idx].zero_grad()  # Clear gradients
                    loss = model_piX[block_idx](Y, X, mu_lambda_beta, sigmasq_lambda_beta, 
                                                 M_S_star, mu_W, eta_X_sq, lambda_a2, lambda_b2, 
                                                 tau_X, n_piX_sample, VX_ub=VX_ub, seed=seed)
                    loss.backward()  # Compute gradients
                    optimizer_piX[block_idx].step()  # Update parameters

                    # Stopping rule: Stop if the loss change is below a threshold
                    if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                        print(f"Block {block_idx}, stopping early at step {step} due to minimal loss change.")
                        break
                    prev_loss = loss.item()
                    print(f"Block {block_idx}, Step {step}, Loss: {loss.item():.4f}")


                # Extract the updated parameters for the current block
                M_X_star[block_idx] = torch.tensor(model_piX[block_idx].current_M_X_star).clone().data
                V_X_star[block_idx] = torch.tensor(model_piX[block_idx].current_V_X_star).clone().data
            
        else:
            M_X_star = M_X_star_fixed
            V_X_star = V_X_star_fixed

        
        if(fix_piS == False):
            for block_idx in range(n_blocks):
                for step in range(n_steps):
                    optimizer_piS[block_idx].zero_grad()  # Clear gradients
                    loss = model_piS[block_idx](Y, X, mu_lambda_beta, M_X_star, 
                             lambda_a2, lambda_b2, mu_W, Sigma_W, eta_S_sq, tau_S, 
                             n_piS_sample, VS_ub=VS_ub, seed=seed)
                    loss.backward()  # Compute gradients
                    optimizer_piS[block_idx].step()  # Update parameters

                    # Stopping rule: Stop if the loss change is below a threshold
                    if step > 0 and abs(prev_loss - loss.item()) < 1e-4:
                        break
                    prev_loss = loss.item()
                    print(f"Block {block_idx}, Step {step}, Loss: {loss.item():.4f}")

            # Extract the updated parameters for the current block
            M_S_star[block_idx] = torch.tensor(model_piS[block_idx].current_M_S_star).clone().data
            V_S_star[block_idx] = torch.tensor(model_piS[block_idx].current_V_S_star).clone().data
        else:
            M_S_star = M_S_star_fixed
            V_S_star = V_S_star_fixed
        
        print(f"Iter {iter+1}/{n_iter} | mu_lambda_beta: {mu_lambda_beta:.4f} | \n sigmasq_lambda_beta: {sigmasq_lambda_beta:.4f} | \n lambda_a1: {lambda_a1:.4f} | lambda_b1: {lambda_b1:.4f} | lambda_a2: {lambda_a2:.4f} | lambda_b2: {lambda_b2:.4f}")
        print(f"‣  E[ϕ]: {mean_phi:.4f} | "
              f"‣ ||mu_W||: {torch.norm(mu_W):.4f}")
        if pi_X_true is not None:
            correct_permutations_list = []
            for block_idx in range(n_blocks):
                est_perm_piX = round_to_perm(M_X_star[block_idx].detach().numpy())
                correct_permutations = torch.sum(torch.tensor(est_perm_piX).T * pi_X_true[block_idx])
                correct_permutations_list.append(correct_permutations)
            avg_correct_permutations = sum(correct_permutations_list) / n_blocks
            print(f"Average correct permutations recognized for piX across all blocks: {avg_correct_permutations}")
        if pi_S_true is not None:
            correct_permutations_list = []
            for block_idx in range(n_blocks):
                est_perm_piS = round_to_perm(M_S_star[block_idx].detach().numpy())
                correct_permutations = torch.sum(torch.tensor(est_perm_piS).T * pi_S_true[block_idx])
                correct_permutations_list.append(correct_permutations)
            avg_correct_permutations = sum(correct_permutations_list) / n_blocks
            print(f"Average correct permutations recognized for piS across all blocks: {avg_correct_permutations}")

        total_loss = 0
        for block_idx in range(n_blocks):
            resid = Y[block_idx] - mu_lambda_beta * (M_X_star[block_idx] @ X[block_idx]) - (M_S_star[block_idx] @ mu_W[block_idx])
            total_loss += (resid.T @ resid)  # (1, 1)
            total_loss += mu_lambda_beta ** 2 * (X[block_idx].T @ (V_X_star[block_idx] - M_X_star[block_idx].T @ M_X_star[block_idx]) @ X[block_idx])
            total_loss += sigmasq_lambda_beta * (X[block_idx].T @ V_X_star[block_idx] @ X[block_idx])
            total_loss += (mu_W[block_idx].T @ (V_S_star[block_idx] - M_S_star[block_idx].T @ M_S_star[block_idx]) @ mu_W[block_idx])
            
        total_loss += torch.trace(torch.block_diag(*[V_S_star[block_idx] for block_idx in range(n_blocks)]) @ Sigma_W)
        print(f"Total Loss: {torch.sqrt(total_loss/(n_blocks*n_locations)):.4f}")
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
        