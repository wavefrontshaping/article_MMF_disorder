import torch
import matplotlib.pyplot as plt
import numpy as np

def getZernikeCoefs(states):
    '''
    Get list of Zernike Coefficients from the model paramters
    '''
    
    coefs_list = [torch.Tensor.cpu(states[name]).numpy()[0] for name in states.keys()]
    return coefs_list

def showZernikeCoefs(zernike_coefs, thresh = 10, **kwargs):
    '''
    Allows a nice display of the amplitude for each Zernike coefficient

    input:
    zernike_coefs: list: list of zernike coefficients

    returns:
    fig and ax: in order to add more coefs to campare later on
    '''
    zernike_names = ['dilat',
                 'ft_tilt_H', 'ft_tilt_V',
                 'ft_astigm_H','ft_defoc','ft_astigm_V',
                 'ft_tref_V','ft_coma_V','ft_coma_H','ft_tref_H',
                 'tilt_H', 'tilt_V',
                 'astigm_H','defoc','astigm_V',
                 'tref_V','coma_V','coma_H','tref_H',
                 'quad_H','sec_astigm_H','spherical','sec_astigm_V','quad_V']
    
    important_coef_index = [] 
    for i in range(len(zernike_names)):
        if np.abs(zernike_coefs[i]) > thresh: # completely arbitrary value
            important_coef_index.append(i)
            
    if 'fig' not in kwargs:
        fig = plt.figure(figsize = (12,7))
    else:
        fig = kwargs['fig']
    if 'ax1' not in kwargs:
        ax1 = fig.add_subplot(111)
    else:
        ax1 = kwargs['ax1']
    ax1.plot(zernike_names,zernike_coefs,'o')
    ax1.set_xticklabels(zernike_names, rotation=40, ha='right')
    ax1.grid(axis = 'x',ls = ':')
    ax1.set_xlabel('Name of correction function')
    ax1.set_ylabel('Amplitude of correction')
    # ax2.set_xlim(ax1.get_xlim())
    ax2 = ax1.twiny()
    ax1Xs = ax1.get_xticks()
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(range(len(zernike_names)))
    ax2.set_xlabel('Index of correction function')

    ylims = plt.gca().get_ylim()
    xlims = plt.gca().get_xlim()
    ax1.vlines(important_coef_index, ymin = ylims[0] ,ymax = ylims[1],ls = 'dashed')
    ax1.hlines([-thresh, thresh], xmin = xlims[0] ,xmax = xlims[1], ls = 'dotted')
    ax1.set_ylim(ymin = ylims[0] ,ymax = ylims[1])
    ax1.set_xlim(xmin = xlims[0] ,xmax = xlims[1])
    for index_label in important_coef_index:
        ax1.get_xticklabels()[index_label].set_color('red')
        ax2.get_xticklabels()[index_label].set_color('red')
    fig.subplots_adjust(bottom=0.15)
    plt.title('Zernike Coefficients values')

    return fig, ax1, important_coef_index