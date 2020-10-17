import torch
import numpy as np

def complex_matmul(A,B):
    '''
    Matrix multiplication for complex tensors. 
    Tensors have to have a last dimension of size 2 for real and imaginary parts.
    The -2 and -3 dimensions are the 2 dimensions to multiply.
    Other previous dimensions are considered as batch dimensions (cf PyTorch matmul() function).
    '''
    return torch.stack((A[...,0].matmul(B[...,0])-A[...,1].matmul(B[...,1]),
                    A[...,0].matmul(B[...,1])+A[...,1].matmul(B[...,0])),dim=-1)

def complex_mul(A,B):
    '''
    Element-wise multiplication for complex tensors. 
    Tensors have to have a last dimension of size 2 for real and imaginary parts.
    The -2 and -3 dimensions are the 2 dimensions to multiply.
    Other previous dimensions are considered as batch dimensions (cf PyTorch mul() function).
    '''
    return torch.stack((A[...,0].mul(B[...,0])-A[...,1].mul(B[...,1]),
                    A[...,0].mul(B[...,1])+A[...,1].mul(B[...,0])),dim=-1)


def pi2_shift(A):
    return torch.stack((-A[...,1],A[...,0]),dim=-1)


def conjugate(A):
    return torch.stack((A[...,0],-A[...,1]), dim=-1)



def complex_fftshift(A):
    n_x = A.shape[-3]
    n_y = A.shape[-2]

    return torch.cat( \
               (torch.cat((A[...,n_x//2:,n_y//2:,:],A[...,:n_x//2,n_y//2:,:]), dim = -3),
               torch.cat((A[...,n_x//2:,:n_y//2,:],A[...,:n_x//2,:n_y//2,:]), dim = -3)), dim = -2)

def complex_ifftshift(A):
    n_x = A.shape[-3]
    n_y = A.shape[-2]
    offset_x = n_x%2
    offset_y = n_y%2
    return torch.cat( \
               (torch.cat((A[...,n_x//2+offset_x:,n_y//2+offset_y:,:],A[...,:n_x//2+offset_x,n_y//2+offset_y:,:]), dim = -3),
               torch.cat((A[...,n_x//2+offset_x:,:n_y//2+offset_y,:],A[...,:n_x//2+offset_x,:n_y//2+offset_y,:]), dim = -3)), 
                     dim = -2)


# def get_tilt_tensor(shape, angle, dtype=torch.float64):
#     X, _ = torch.meshgrid(torch.arange(0,1,1./shape[0]),torch.arange(0,1,1./shape[1]))
#     return torch.stack((torch.cos(angle*X),torch.sin(angle*X)), dim = -1)

def crop_center(input, size):
    x = input.shape[1]
    y = input.shape[2]
    start_x = x//2-(size//2)
    start_y = y//2-(size//2)
    return input[:,start_x:start_x+size,start_y:start_y+size,...]

def pt_to_cpx(A):
    return np.array(A[...,0])+1j*np.array(A[...,1])


def cpx_to_pt(A, device, dtype = torch.float32):
    return torch.stack((torch.from_numpy(A.real),
                        torch.from_numpy(A.imag)), dim = -1).type(dtype).to(device)

def norm2(A,device):
    return torch.sqrt(torch.sum(complex_mul(A,conjugate(A))[...,0],dim = 1))

def normalize(A,device):
    b = norm2(A, device = device)

    mid_dim = A.shape[1]
    zeros = torch.zeros(mid_dim,device = device)
    divider = torch.meshgrid(b,zeros)[0]
    normalized = torch.stack((A[:,:,0] / divider,
                 A[:,:,1] / divider), dim = -1)
    return normalized


def tm_to_pt(A, device, dtype = torch.float32):
    if len(A.shape) == 2:
        return torch.stack((torch.from_numpy(A.real),
                            torch.from_numpy(A.imag))).permute((1,2,0)).type(dtype).to(device)
    else:
        return torch.stack((torch.from_numpy(A.real),
                            torch.from_numpy(A.imag))).permute((1,2,3,0)).type(dtype).to(device)
            