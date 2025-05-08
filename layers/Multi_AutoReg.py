import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

def get_ffd_weight(d, thres, x_enc):
    """
    Args:
        d: fracrtional order
        thres: threshold
        x_enc: input series x
    Returns:
        fractional differential weight
    """
    lim = len(x_enc)
    w, k = [1.], 1
    ctr = 0
    while True:
        _w = -w[-1] / k * (d - k + 1)
        if abs(_w) < thres:
          break
        w.append(_w)
        k += 1
        ctr += 1
        if ctr == lim - 1:
          break
    #w[0] = 0
    w = torch.tensor(w[:-1:], device=x_enc.device).reshape(-1, 1) 
    #x_w = torch.zeros(x_enc.shape[0], x_enc.shape[2], x_enc.shape[2])
    #x_w = torch.zeros(x_enc.shape)
    #x_w[:L, :L] = torch.diag_embed(w)
    #x_w[:, ] = torch.diag_embed(w.squeeze())
    
    return w


def creat_lag(x, lags):
    """
    Args:
        x: batch data with shape [batch_size, seq_len, variates]
        lags: number of lag
    Returns:
        out_x: data of lags [batch_size, seq_len - lags, variates * lags]
        out_y: current data [batch_size, seq_len - lags, variates]
    """

    L = x.shape[1]
    #ffd__weight = get_ffd_weight(0.1, 0.001, x).unsqueeze(0).repeat(B, 10, D)
    #out_x = torch.roll(x[:, :L - lags, :], -lags, dims=-1)
    out_x = x[:, :L - lags, :] 
    #out_x = torch.mul(torch.roll(lag_x, -lags, dims=-1), ffd_weight)
    #out_y = F.pad(x[:, lags:, :], pad=(0, 1), mode='constant', value=0)
    out_y = x[:, lags:, :]
    #print(out_x.shape, out_y.shape)
    
    return out_x, out_y

def fit_ar_batch(x, y, n_heads):
    """
    Batch data for AutoReg model fit.
    Args:
        x: lag data [batch_size, seq_len - lags, variates * lags]
        y: current_data [batch_size, seq_len - lags, variates]

    Returns:
        model parameter metrics [batch_size, variates * lags, variates]
    """
    #B, D, L = x.shape
    #x = torch.reshape(x, (B, D, n_heads, -1))
    #y = torch.reshape(y, (B, D, n_heads, -1))

    #a = torch.einsum("bdhl,bchl->bhdc", x, x)
    #b = torch.einsum("bdhl,bchl->bhdc", x, y)
    a = torch.einsum("bhl,bhs->bls", x, x)
    b = torch.einsum("bhl,bhs->bls", x, y)
    # least square
    #a = torch.matmul(x, x.transpose(1, 2))
    #b = torch.matmul(y, x.transpose(1, 2))
    beta = torch.linalg.solve(a + 1e-5 * torch.eye(a.shape[1], device=a.device), b)

    return beta

def predict_ar_batch(x, beta):
    """
    Batch data for AutoReg model predict.
    Args:
        x: lag data [batch_size, seq_len - lags, variates * lags]
        beta: parameter data [batch_size, variates * lags, variates]

    Returns:
        Y: predict data metrics [batch_size, seq_len - lag, variates]
    """
    Y = beta @ x # lag matrix multiply parameter matrix

    return Y
