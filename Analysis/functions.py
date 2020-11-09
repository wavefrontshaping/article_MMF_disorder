from colorsys import hls_to_rgb
import numpy as np
import matplotlib.pyplot as plt

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
 
def show_colormap_image(to_img, save_name = None):

    n_phi = 30
    n_amp = 100
    X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,n_phi),np.linspace(1.,0.,n_amp))

    cm = Y*np.exp(1j*X)

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(to_img(cm.transpose()), extent = [-np.pi, np.pi, 0., 1.], aspect = 30)

    ax.set_xticks([-np.pi,0,np.pi])
    ax.set_xticklabels([r'$\pi$', '0', r'$\pi$'])
    ax.set_yticks([0,0.5,1])
    if save_name:
        plt.savefig(save_name, dpi = 200)
    
def complex_correlation(Y1,Y2):
    Y1 = Y1-Y1.mean()
    Y2 = Y2-Y2.mean()
    return np.abs(np.sum(Y1.ravel() * Y2.ravel().conj())) \
           / np.sqrt(np.sum(np.abs(Y1.ravel())**2) *np.sum(np.abs(Y2.ravel())**2))


tr = lambda A,B: np.trace(np.abs(A@B.transpose().conjugate())**2)

fidelity = lambda A,B: tr(A,B)/(np.sqrt(tr(A,A)*tr(B,B)))