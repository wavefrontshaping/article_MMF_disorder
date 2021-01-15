import torch
from torch.nn import Module, ZeroPad2d
from PyTorchAberrations.aberration_functions import complex_mul, conjugate, pi2_shift





################################################################
################### AUTOGRAD FUNCTIONS #########################
################################################################
     
class ComplexZernikeFunction(torch.autograd.Function):
    '''
    Function that apply a complex Zernike polynomial to the phase of a batch 
    of compleximages (or a matrix).
    '''
    @staticmethod
    def forward(ctx, input, alpha, j):
        
        
        
        nx = torch.arange(0,1,1./input.shape[1], dtype = input.dtype)
        ny = torch.arange(0,1,1./input.shape[2], dtype = input.dtype)

        X0, Y0 = 0.5+0.5/input.shape[1], 0.5+0.5/input.shape[2]
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)-X0
        Y = Y.to(input.device)-Y0
        
        # see https://en.wikipedia.org/wiki/Zernike_polynomials
        if j == 0:
            F = torch.ones_like(X)
        elif j == 1:
            F = X
        elif j == 2:
            F = Y
        elif j == 3:
            # Oblique astigmatism
            F = 2.*X.mul(Y)
        elif j == 4:
            # Defocus
            F = X**2+Y**2
        elif j == 5:
            # Vertical astigmatism
            F = X**2-Y**2
        else:
            R = torch.sqrt(X**2+Y**2)
            THETA = torch.atan2(Y, X)
            if j == 6:
                # Vertical trefoil 
                F = torch.mul(R**3, torch.sin(3.*THETA))
            elif j == 7:
                # Vertical coma
                F = torch.mul(3.*R**3,torch.sin(3.*THETA))
            elif j == 8:
                # Horizontal coma 
                F = torch.mul(3.*R**3,torch.cos(3.*THETA))
            elif j == 9:
                # Oblique trefoil 
                F = torch.mul(R**3, torch.cos(3.*THETA))
            elif j == 10:
                # Oblique quadrafoil 
                F = 2.*torch.mul(R**4, torch.sin(4.*THETA))
            elif j == 11:
                # Oblique secondary astigmatism 
                F = 2.*torch.mul(4.*R**4-3.*R**2, torch.sin(2.*THETA))
            elif j == 12:
                # Primary spherical
                F = 6.*R**4-6.*R**2 + torch.ones_like(R)
            elif j == 13:
                # Vertical secondary astigmatism 
                F = 2.*torch.mul(4.*R**4-3.*R**2, torch.cos(2.*THETA))
            elif j == 14:
                # Vertical quadrafoil 
                F = 2.*torch.mul(R**4, torch.cos(4.*THETA))
            else:
                raise
        
        weight = torch.stack((torch.cos(alpha*F),
                              torch.sin(alpha*F)), dim = -1)

        
        output = complex_mul(input,weight)
            
        
        ctx.save_for_backward(input, weight, alpha, F)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, alpha, F = ctx.saved_tensors
        
        grad_input = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = complex_mul(grad_output,conjugate(weight))

        if ctx.needs_input_grad[1]:

                
            F = torch.stack((F,F),dim = -1).expand_as(grad_output)
            grad_alpha = -torch.sum(torch.sin(alpha*F)*F*input*grad_output) \
                         + torch.sum(torch.cos(alpha*F)*F*pi2_shift(input)*grad_output)
            grad_alpha.unsqueeze_(0)

        return grad_input, grad_alpha, None
    
    

#######################################################
#################### MODULES ##########################
#######################################################

class ComplexZeroPad2d(Module):
    '''
    Apply zero padding to a batch of 2D complex images (or matrix)
    '''
    def __init__(self, padding):
        super(ComplexZeroPad2d, self).__init__()
        self.pad_r = ZeroPad2d(padding)
        self.pad_i = ZeroPad2d(padding)

    def forward(self,input):
        return torch.stack((self.pad_r(input[...,0]), 
                           self.pad_i(input[...,1])), dim = -1)     

class ComplexZernike(Module):
    '''
    Layer that apply a complex Zernike polynomial to the phase of a batch 
    of compleximages (or a matrix).
    Only one parameter, the strenght of the polynomial, is learned.
    Initial value is 0.
    '''
    def __init__(self, j):
        super(ComplexZernike, self).__init__()
        assert j in range(15)
        self.j = j
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):
        return ComplexZernikeFunction.apply(input, self.alpha, self.j)

class ComplexScaling(Module):
    '''
    Layer that apply a global scaling to a stack of 2D complex images (or matrix).
    Only one parameter, the scaling factor, is learned. 
    Initial value is 1.
    '''
    def __init__(self):
        super(ComplexScaling, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = input.permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta)*(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 
                                         
            return torch.nn.functional.grid_sample(input, grid).permute((0,2,3,1))
        
class ComplexDeformation(Module):
    '''
    Layer that apply a global affine transformation to a stack of 2D complex images (or matrix).
    6 parameters are learned.
    '''
    def __init__(self):
        super(ComplexDeformation, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.tensor([0., 0, 0, 0, 0., 0]))
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = input.permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta).mul(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 

            return torch.nn.functional.grid_sample(input, grid).permute((0,2,3,1))
