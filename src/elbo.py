from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import math
from scipy.optimize import linear_sum_assignment
import numpy as np

def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    row, col = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row, col] = 1.0
    return P


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def min_zero_row(zero_mat: Tensor) -> Tuple[Tensor, Tensor]:
#     sum_zero_mat = zero_mat.sum(1)
#     sum_zero_mat[sum_zero_mat == 0] = 9999

#     zero_row = sum_zero_mat.min(0)[1]
#     zero_column = zero_mat[zero_row].nonzero()[0]

#     zero_mat[zero_row, :] = False
#     zero_mat[:, zero_column] = False

#     mark_zero = torch.tensor([[zero_row, zero_column]], device = device)
#     return zero_mat, mark_zero

# def mark_matrix(mat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#     zero_bool_mat = (mat == 0)
#     zero_bool_mat_copy = zero_bool_mat.clone()

#     marked_zero = torch.tensor([], device = device)
#     while (True in zero_bool_mat_copy):
#         zero_bool_mat_copy, mark_zero = min_zero_row(zero_bool_mat_copy)
#         marked_zero = torch.concat([marked_zero, mark_zero], dim = 0)

#     marked_zero_row = marked_zero[:, 0]
#     marked_zero_col = marked_zero[:, 1]

#     arange_index_row = torch.arange(mat.shape[0], dtype=torch.float, device = device).unsqueeze(1)
    
#     repeated_marked_row = marked_zero_row.repeat(mat.shape[0], 1)
#     bool_non_marked_row = torch.all(arange_index_row != repeated_marked_row, dim = 1)
#     non_marked_row = arange_index_row[bool_non_marked_row].squeeze()

#     non_marked_mat = zero_bool_mat[non_marked_row.long(), :]
#     marked_cols = non_marked_mat.nonzero().unique()

#     is_need_add_row = True
#     while is_need_add_row:
#         repeated_non_marked_row = non_marked_row.repeat(marked_zero_row.shape[0], 1)
#         repeated_marked_cols = marked_cols.repeat(marked_zero_col.shape[0], 1)

#         first_bool = torch.all(marked_zero_row.unsqueeze(1) != repeated_non_marked_row, dim = 1)
#         second_bool = torch.any(marked_zero_col.unsqueeze(1) == repeated_marked_cols, dim = 1)

#         addit_non_marked_row = marked_zero_row[first_bool & second_bool]

#         if addit_non_marked_row.shape[0] > 0:
#             non_marked_row = torch.concat([non_marked_row.reshape(-1), addit_non_marked_row[0].reshape(-1)])
#         else:
#             is_need_add_row = False

#     repeated_non_marked_row = non_marked_row.repeat(mat.shape[0], 1)
#     bool_marked_row = torch.all(arange_index_row != repeated_non_marked_row, dim = 1)
#     marked_rows = arange_index_row[bool_marked_row].squeeze(0)

#     return marked_zero, marked_rows, marked_cols

# def adjust_matrix(mat: Tensor, cover_rows: Tensor, cover_cols: Tensor) -> Tensor:
#     bool_cover = torch.zeros_like(mat)
#     bool_cover[cover_rows.long()] = True
#     bool_cover[:, cover_cols.long()] = True

#     non_cover = mat[bool_cover != True]
#     min_non_cover = non_cover.min()

#     mat[bool_cover != True] = mat[bool_cover != True] - min_non_cover

#     double_bool_cover = torch.zeros_like(mat)
#     double_bool_cover[cover_rows.long(), cover_cols.long()] = True

#     mat[double_bool_cover == True] = mat[double_bool_cover == True] + min_non_cover

#     return mat

# def hungarian_algorithm(mat: Tensor) -> Tensor:
#     dim = mat.shape[0]
#     cur_mat = mat.clone()

#     cur_mat = cur_mat - cur_mat.min(1, keepdim=True)[0]
#     cur_mat = cur_mat - cur_mat.min(0, keepdim=True)[0]

#     zero_count = 0
#     iters = 0
#     while zero_count < dim and iters < 100:
#         ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
#         zero_count = len(marked_rows) + len(marked_cols)

#         if zero_count < dim:
#             cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
#         iters += 1

#     # Create permutation matrix
#     perm_matrix = torch.zeros_like(mat)
#     for pos in ans_pos:
#         i, j = int(pos[0]), int(pos[1])  # Ensure indices are integers
#         perm_matrix[i, j] = 1

#     return perm_matrix

def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - torch.logsumexp(logP, dim=0, keepdim=True)
        # Normalize rows and take the log again
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)
    return logP

class vi_piX(nn.Module):
    def __init__(self, n_locations):
        super(vi_piX, self).__init__()
        self.n_locations = n_locations
        # Initialize with a random doubly stochastic matrix
        random_matrix = torch.rand(n_locations, n_locations)
        log_doubly_stochastic = sinkhorn_logspace((random_matrix), niters=10)
        self.MX = nn.Parameter(log_doubly_stochastic, requires_grad=True)
        self.VX = nn.Parameter((-2)*torch.ones(n_locations, n_locations),requires_grad=True)
        self.current_M_X_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_X_star = torch.eye(n_locations, n_locations)
        #self.VX_unconstrained = nn.Parameter(torch.full((n_locations, n_locations), 0.2))
            
    def forward(self, Y, X, mu_lambda_beta,
                sigmasq_lambda_beta, M_S_star, mu_W,
                eta_X_sq, lambda_a2, lambda_b2, 
                tau_X = 0.1, n_piX_sample = 100, seed = 100):
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # sample piX
         #set seed for stochastic optimzation

        # calculate nearest doubly stochastic matrix to MX once
        log_MX = self.MX
        log_MX_tilde = sinkhorn_logspace(log_MX, niters=10)
        MX_tilde = torch.exp(log_MX_tilde)
        #VX = torch.nn.functional.softplus(self.VX_unconstrained)
    

        # Compute the ELBO
        B = Y.shape[0]
        elbo = 0.0
        current_M_X_star = self.current_M_X_star
        current_V_X_star = self.current_V_X_star
        for i in range(n_piX_sample):
            torch.manual_seed(i * seed)
            #Phi = MX_tilde + torch.sqrt(torch.exp(self.VX)) * torch.randn(self.n_locations, self.n_locations)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MX_tilde + torch.sqrt(torch.special.expit(self.VX)*(2-0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi - 0.95 * Phi.min()).detach().numpy())
            current_piX = tau_X * Phi + (1 - tau_X) * torch.tensor(round_Phi, dtype=Phi.dtype)  # Access the current sample of piS
            #current_piX = tau_X * Phi + (1 - tau_X) * hungarian_algorithm(-Phi)
            #current_piX = tau_X * Phi + (1 - tau_X) * torch.tensor(round_to_perm((Phi - 0.95*Phi.min()).detach().numpy()), dtype=torch.float) # Access the current sample of piX
            current_M_X_star += current_piX
            current_V_X_star += current_piX.T @ current_piX

            # Computation for all regions
            term1 = 0.
            for j in range(B):
                Y_i = Y[j]                    # (d,)
                X_i = X[j]                    # (d,)
                mu_Wi = mu_W[j]              # (d,)

                # (1) 2 * Y_i^T * pi_x * X_i * mu
                temp = current_piX @ X_i 
                part1 = -2 * mu_lambda_beta * torch.dot(Y_i, temp)

                # (2) (mu^2 + sigma^2) * X_i^T * pi_x^T * pi_x * X_i
                part2 = (mu_lambda_beta ** 2 + sigmasq_lambda_beta) * temp.dot(temp)



                # (3) 2 * mu * X_i^T * pi_x^T * M_star_S * mu_Wi
                part3 = 2 * mu_lambda_beta * (temp.T @ M_S_star @ mu_Wi)

                term1 += part1 + part2 + part3

            coeff = -lambda_a2 / (2 * lambda_b2)
            total_term1 = coeff * term1

            # Second summation: over entries of pi_x
            x_mk_squared = current_piX.pow(2)
            x_mk_minus1_squared = (current_piX - 1).pow(2)

            exponent1 = -x_mk_squared / (2 * eta_X_sq)
            exponent2 = -x_mk_minus1_squared / (2 * eta_X_sq)

            # Stable log-sum-exp
            log_term = torch.logsumexp(torch.stack([exponent1, exponent2]), dim=0)
            total_log_term = (log_term).sum()

            # Final terms
            neg_log_tauX = self.n_locations ** 2 * torch.log(torch.tensor(tau_X))

            # Update ELBO
            #elbo += total_term1 + total_log_term + neg_log_tauX + 0.5 * self.VX.sum()
            elbo += total_term1 + total_log_term + neg_log_tauX + 0.5* torch.log(torch.special.expit(self.VX)*(2-0.01) + 0.01).sum()  # Ensure positive definiteness
            #elbo += total_term1 + neg_log_tauX + 2 * self.VX.sum()

        
        elbo = elbo / n_piX_sample 
        self.current_M_X_star = current_M_X_star / n_piX_sample
        self.current_V_X_star = current_V_X_star / n_piX_sample
        return -elbo


class vi_piS(nn.Module):
    def __init__(self, n_locations):
        super(vi_piS, self).__init__()
        self.n_locations = n_locations
        self.MS = nn.Parameter(torch.log(1 / torch.tensor(n_locations)) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.VS = nn.Parameter((-2) * torch.ones(n_locations, n_locations, requires_grad=True))
        self.current_M_S_star = (1/torch.tensor(n_locations)) * torch.ones(n_locations, n_locations)
        self.current_V_S_star = torch.eye(n_locations, n_locations)

    def forward(self, Y, X, mu_lambda_beta, M_X_star,
                lambda_a2, lambda_b2, mu_W, Sigma_W,
                eta_S_sq, tau_S=0.1, n_piS_sample=100,seed=100):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
          #set seed for stochastic optimzation
        # set seed for reproducibility

        # Sinkhorn to project to the doubly stochastic matrix
        log_MS = self.MS
        log_MS_tilde = sinkhorn_logspace(log_MS, niters=10)
        MS_tilde = torch.exp(log_MS_tilde)

        B = Y.shape[0]
        elbo = 0.0
        current_M_S_star = self.current_M_S_star
        current_V_S_star = self.current_V_S_star

        for i in range(n_piS_sample):
            torch.manual_seed(i * seed)
            #Phi = MS_tilde + torch.sqrt(torch.exp(self.VS)) * torch.randn(self.n_locations, self.n_locations)
            z = torch.randn(self.n_locations, self.n_locations)
            Phi = MS_tilde + torch.sqrt(torch.special.expit(self.VS)*(2-0.01) + 0.01) * z
            round_Phi = round_to_perm((Phi - 0.95 * Phi.min()).detach().numpy())
            current_piS = tau_S * Phi + (1 - tau_S) * torch.tensor(round_Phi, dtype=Phi.dtype)  # Access the current sample of piS

            current_piS_sq = current_piS.T @ current_piS
            current_M_S_star += current_piS
            current_V_S_star += current_piS_sq

            term1 = 0.
            for j in range(B):
                Y_j = Y[j]            # (d,)
                X_j = X[j]            # (d,)
                mu_W_j = mu_W[j]      # (d,)
                Sigma_W_j = Sigma_W[j*self.n_locations : (j+1)*self.n_locations, j*self.n_locations : (j+1)*self.n_locations]  # (d, d)

                # Terms in the summation
                part1 = -2 * torch.dot(Y_j, current_piS @ mu_W_j)
                part2 = 2 * mu_lambda_beta * torch.dot(X_j, M_X_star.T @ current_piS @ mu_W_j)
                part3 = (mu_W_j.T @ current_piS_sq @ mu_W_j)
                part4 = torch.trace(Sigma_W_j @ current_piS_sq)

                term1 += part1 + part2 + part3 + part4

            coeff = -lambda_a2 / (2 * lambda_b2)
            total_term1 = coeff * term1

            # Prior term over pi_S entries
            s_mk_squared = current_piS.pow(2)
            s_mk_minus1_squared = (current_piS - 1).pow(2)

            exponent1 = -s_mk_squared / (2 * eta_S_sq)
            exponent2 = -s_mk_minus1_squared / (2 * eta_S_sq)

            log_term = torch.logsumexp(torch.stack([exponent1, exponent2]), dim=0)
            total_log_term = log_term.sum()

            # Final terms
            neg_log_tauS = self.n_locations ** 2 * torch.log(torch.tensor(tau_S))

            # Update ELBO
            elbo += total_term1 + total_log_term + neg_log_tauS + 0.5* torch.log(torch.special.expit(self.VS)*(2-0.01) + 0.01).sum()

        elbo = elbo / n_piS_sample
        self.current_M_S_star = current_M_S_star / n_piS_sample
        self.current_V_S_star = current_V_S_star / n_piS_sample
        return -elbo

