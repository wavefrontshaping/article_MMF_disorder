import torch
import numpy as np
from torch.nn import Module, Sequential, Identity
from PyTorchAberrations.aberration_layers import ComplexZeroPad2d, ComplexDeformation
from PyTorchAberrations.aberration_layers import ComplexZernike, ComplexScaling
from PyTorchAberrations.aberration_functions import crop_center, complex_fftshift, complex_ifftshift, conjugate, normalize

class AberrationModes(torch.nn.Module):
    '''
    Model for input and output aberrations.
    Apply an `Aberration` model to the input and output mode basis.
    '''
    def __init__(self, 
                     inpoints,
                     onpoints,
                     padding_coeff = 0.,
                     list_zernike_ft = list(range(3)),
                     list_zernike_direct = list(range(3)),
                     deformation = 'single'):
        super(AberrationModes, self).__init__()
        self.abberation_output = Aberration(onpoints,
                                            list_zernike_ft = list_zernike_ft,
                                            list_zernike_direct = list_zernike_direct, 
                                            padding_coeff = padding_coeff,
                                            deformation = deformation)
        self.abberation_input = Aberration(inpoints,
                                            list_zernike_ft = list_zernike_ft,
                                            list_zernike_direct = list_zernike_direct, 
                                            padding_coeff = padding_coeff,
                                            deformation = deformation)
        self.inpoints = inpoints
        self.onpoints = onpoints

    def forward(self,input, output):
        
        output_modes = output
        output_modes = self.abberation_output(output_modes)
        # output_modes = normalize(output_modes.reshape((-1,self.onpoints**2,2)),device = self.device).reshape((-1,self.onpoints,self.onpoints,2))
        

        input_modes = input
        input_modes = self.abberation_input(input_modes)
        # input_modes = normalize(input_modes.reshape((-1,self.inpoints**2,2)),device = self.device).reshape((-1,self.inpoints,self.inpoints,2))

        return output_modes, input_modes

    
class Aberration(torch.nn.Module):
    '''
    Model that apply aberrations (direct and Fourier plane) and a global scaling
    at the input dimension of a matrix.
    '''
    def __init__(self, 
                 shape,
                 list_zernike_ft,
                 list_zernike_direct,
                 padding_coeff = 0., 
                 deformation = 'single',
                 features = None):
        # Here we define the type of Model we want to be using, the number of polynoms and if we want to implement a deformation.
        super(Aberration, self).__init__()
        
        #Check whether the model is given the lists of zernike polynoms to use or simply the total number to use
        if type(list_zernike_direct) not in [list, np.ndarray]:
            list_zernike_direct = range(0,list_zernike_direct)
        if type(list_zernike_ft) not in [list, np.ndarray]:
            list_zernike_ft = range(0,list_zernike_ft)

        self.nxy = shape
        
        # padding layer, to have a good FFT resolution
        # (requires to crop after IFFT)
        padding = int(padding_coeff*self.nxy)
        self.pad = ComplexZeroPad2d(padding)
        
        # scaling x, y
        if deformation == 'single':
            self.deformation = ComplexDeformation()
        elif deformation == 'scaling':
            self.deformation = ComplexScaling()
        else:
            self.deformation = Identity()
        
        self.zernike_ft = Sequential(*(ComplexZernike(j=j + 1) for j in list_zernike_ft))
        self.zernike_direct = Sequential(*(ComplexZernike(j=j + 1) for j in list_zernike_direct))
       
      
    def forward(self,input):
        assert(input.shape[1] == input.shape[2])
        
        # padding
        input = self.pad(input)
        
        # scaling
        input = self.deformation(input)
        
        # to Fourier domain
        input = complex_ifftshift(input) 
        input = torch.fft(input, 2)
        input = complex_fftshift(input)

        # Zernike layers in the Fourier plane
        input = self.zernike_ft(input)

        # to direct domain
        input = complex_ifftshift(input)
        input = torch.ifft(input, 2)
        input = complex_fftshift(input)
         
        # Zernike layers in the direct plane
        input = self.zernike_direct(input)
        
        # Crop at the center (because of coeff) 
        input = crop_center(input,self.nxy)

        return input
      