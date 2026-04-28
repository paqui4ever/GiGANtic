import torch

def gaussian_negative_log_likelihood(codes, mu, log_var):
    squared_error = (codes - mu) ** 2
    # torch.exp(-logvar) = 1 / sigma**2
    weighted_squared_error = squared_error * torch.exp(-log_var)

    penalty_term = 0.5 * log_var

    return weighted_squared_error + penalty_term

