import torch
from torch.nn import Module, ZeroPad2d
from PyTorchAberrations.aberration_functions import complex_mul, conjugate, pi2_shift





################################################################
################### AUTOGRAD FUNCTIONS #########################
################################################################
     

class ComplexTiltFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, theta, axis = 0):
        nx = torch.arange(0,1,1./input.shape[1], dtype = input.dtype)
        ny = torch.arange(0,1,1./input.shape[2], dtype = input.dtype)
        
#         nx = torch.arange(0,input.shape[1], dtype = input.dtype)
#         ny = torch.arange(0,input.shape[2], dtype = input.dtype)
        grid = torch.meshgrid(nx,ny)
        
        X = grid[axis].type(input.dtype).to(input.device)
        
        weight = torch.stack((torch.cos(theta*X),torch.sin(theta*X)), dim = -1)
        output = complex_mul(input,weight)
        
        ctx.save_for_backward(input, weight,theta, X)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, theta, X = ctx.saved_tensors

        grad_input = grad_theta = None

        if ctx.needs_input_grad[0]:
            grad_input = complex_mul(grad_output,conjugate(weight))

        if ctx.needs_input_grad[1]:
            # Calculation of the gradient w.r.t the parameter (here the phase tilt)
            # CAN BE OPTIMIZED!!!
            XX = torch.stack((X,X),dim = -1).expand_as(grad_output)
            grad_theta = torch.stack((-torch.sum(torch.sin(theta*XX)*XX*input*grad_output),
                                      torch.sum(torch.cos(theta*XX)*XX*pi2_shift(input)*grad_output)),dim=-1)

        return grad_input, grad_theta, None 
        # None gradient for the axis argument, obviously

class ComplexDefocusFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha):
        nx = torch.arange(0,1,1./input.shape[1], dtype = input.dtype)
        ny = torch.arange(0,1,1./input.shape[2], dtype = input.dtype)
#         nx = torch.arange(0,input.shape[1], dtype = input.dtype)
#         ny = torch.arange(0,input.shape[2], dtype = input.dtype)
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)
        Y = Y.to(input.device)
        X0, Y0 = 0.5+0.5/input.shape[1], 0.5+0.5/input.shape[2]
#         X0, Y0 = (input.shape[1]-1)*.5, (input.shape[2]-1)*.5

        Rsq = (torch.abs(X-X0)**2+torch.abs(Y-Y0)**2)
        weight = torch.stack((torch.cos(alpha*Rsq),
                              torch.sin(alpha*Rsq)), dim = -1)

        output = complex_mul(input,weight)
            
        
        ctx.save_for_backward(input, weight, alpha, Rsq)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, alpha, Rsq = ctx.saved_tensors
        

        grad_input = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = complex_mul(grad_output,conjugate(weight))

        if ctx.needs_input_grad[1]:

            Rsq = torch.stack((Rsq,Rsq),dim = -1).expand_as(grad_output)
            grad_alpha = torch.stack((-torch.sum(torch.sin(alpha*Rsq)*Rsq*input*grad_output),
                                      torch.sum(torch.cos(alpha*Rsq)*Rsq*pi2_shift(input)*grad_output)),dim=-1)


        return grad_input, grad_alpha, None
    
    
class ComplexZernikeFunction(torch.autograd.Function):

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
            grad_alpha = torch.stack((-torch.sum(torch.sin(alpha*F)*F*input*grad_output),
                                      torch.sum(torch.cos(alpha*F)*F*pi2_shift(input)*grad_output)),dim=-1)


        return grad_input, grad_alpha, None
    
class ComplexAstigmatismFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha, j):
        
        assert j in [0,1]
        
        nx = torch.arange(0,1,1./input.shape[1], dtype = input.dtype)
        ny = torch.arange(0,1,1./input.shape[2], dtype = input.dtype)

        X0, Y0 = 0.5+0.5/input.shape[1], 0.5+0.5/input.shape[2]
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)-X0
        Y = Y.to(input.device)-Y0
        
        
        if j == 0:
            XY = 2.*X.mul(Y)
        elif j ==1:
            XY = X**2-Y**2
        
        weight = torch.stack((torch.cos(alpha*XY),
                              torch.sin(alpha*XY)), dim = -1)

        output = complex_mul(input,weight)
            
        
        ctx.save_for_backward(input, weight, alpha, XY)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, alpha, XY = ctx.saved_tensors
        

        grad_input = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = complex_mul(grad_output,conjugate(weight))

        if ctx.needs_input_grad[1]:

            XY = torch.stack((XY,XY),dim = -1).expand_as(grad_output)
            grad_alpha = torch.stack((-torch.sum(torch.sin(alpha*XY)*XY*input*grad_output),
                                      torch.sum(torch.cos(alpha*XY)*XY*pi2_shift(input)*grad_output)),dim=-1)


        return grad_input, grad_alpha, None
    
    

#######################################################
#################### MODULES ##########################
#######################################################

class ComplexZeroPad2d(Module):

    def __init__(self, padding):
        super(ComplexZeroPad2d, self).__init__()
        self.pad_r = ZeroPad2d(padding)
        self.pad_i = ZeroPad2d(padding)

    def forward(self,input):
        return torch.stack((self.pad_r(input[...,0]), 
                           self.pad_i(input[...,1])), dim = -1)     

class ComplexZernike(Module):
    def __init__(self, j):
        super(ComplexZernike, self).__init__()
        assert j in range(15)
        self.j = j
#         self.theta = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.theta.data.uniform_(-1, 1)

    def forward(self, input):
        return ComplexZernikeFunction.apply(input, self.alpha, self.j)

class ComplexTilt(Module):
    def __init__(self, axis = 0):
        super(ComplexTilt, self).__init__()
        self.axis = axis
#         self.theta = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.theta.data.uniform_(-1, 1)

    def forward(self, input):
        return ComplexTiltFunction.apply(input, self.theta, self.axis)
    
    
class ComplexAstigmatism(Module):
    def __init__(self, j):
        super(ComplexAstigmatism, self).__init__()
        self.j = j
#         self.theta = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.theta.data.uniform_(-1, 1)

    def forward(self, input):
        return ComplexAstigmatismFunction.apply(input, self.theta, self.j)
    
class ComplexDefocus(Module):
    def __init__(self):
        super(ComplexDefocus, self).__init__()
#         self.factor = torch.nn.Parameter(torch.tensor(factor), requires_grad=False)
#         self.alpha = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.alpha.data.uniform_(-.1, .1)

    def forward(self, input):
        return ComplexDefocusFunction.apply(input, self.alpha)
 



    
class ComplexBatchDeformation(Module):
    def __init__(self, features):
        super(ComplexBatchDeformation, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.zeros((features,2,3)))
                                        
        # mask to keep only scaling parameters
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        self.mask = torch.tensor([1., 0, 0, 0, 1., 0]).reshape((2,3)).expand(features,2,3)

    def forward(self, input):
            input = input.permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid((1.+5e-2*self.theta).mul(self.mask.to(input.device)).type(input.dtype), 
                                                    input.size())

            return torch.nn.functional.grid_sample(input, grid).permute((0,2,3,1))

class ComplexScaling(Module):
    def __init__(self):
        super(ComplexScaling, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
#             print(self.theta.device)
#             print(input.device)
#             print(torch.tensor([1, 0, 0, 0, 1, 0], dtype=input.dtype).device)
            input = input.permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta)*(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device)) #+ \
                #(5e-2*self.theta).mul(torch.tensor([0, 1., 1., 1., 0, 1.],
                                         #dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 
                                         
                                         
                                         
#             grid = torch.nn.functional.affine_grid(
#                 (1.+5e-2*self.theta).mul(torch.tensor([1, 1., 1., 1., 1, 1.],
#                                          dtype=input.dtype).to(input.device)).reshape((2,3)).expand((input.shape[0],2,3)), 
#                 input.size())

            return torch.nn.functional.grid_sample(input, grid).permute((0,2,3,1))
        
class ComplexDeformation(Module):
    def __init__(self):
        super(ComplexDeformation, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.tensor([0., 0, 0, 0, 0., 0]))
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
#             print(self.theta.device)
#             print(input.device)
#             print(torch.tensor([1, 0, 0, 0, 1, 0], dtype=input.dtype).device)
            input = input.permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta).mul(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device)) #+ \
                #(5e-2*self.theta).mul(torch.tensor([0, 1., 1., 1., 0, 1.],
                                         #dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 
                                         
                                         
                                         
#             grid = torch.nn.functional.affine_grid(
#                 (1.+5e-2*self.theta).mul(torch.tensor([1, 1., 1., 1., 1, 1.],
#                                          dtype=input.dtype).to(input.device)).reshape((2,3)).expand((input.shape[0],2,3)), 
#                 input.size())

            return torch.nn.functional.grid_sample(input, grid).permute((0,2,3,1))

    
# class ComplexScaling2D(Module):
#     def __init__(self):
#         super(ComplexScaling2D, self).__init__()
#         self.scaling = torch.nn.Parameter(torch.Tensor(2))
#         self.scaling.data.uniform_(-0.1, 0.1)

#     def forward(self, input):
#         input = input.permute((0,3,1,2))
#         return torch.nn.functional.interpolate(input,
# #         return torch.nn.Upsample.apply(input,
#                                                scale_factor = (1.+self.scaling[0],1.+self.scaling[1]),
# #                                                mode = 'nearest'
#                                               ).permute((0,2,3,1))

    
