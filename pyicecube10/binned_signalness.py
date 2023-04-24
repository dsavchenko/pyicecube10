#%%
import numpy as np
from scipy import signal
from pyicecube10.datareader import psf, bins_psf_min, bins_psf_max
#%%
ROI = 6
pixsize = 0.05
Rmax = 1.5
Ang_err_cut = 1. # TODO: do not forget to increase

#%%
Npix=int(ROI/pixsize) #number of pixels in model images
Rbins=int(Rmax/pixsize)
Ang_err_bins=int((Ang_err_cut-0.2+pixsize)/pixsize)
radii=np.linspace(pixsize,Rmax,Rbins) #disk or gaussian radii 
ang_errs=np.linspace(0.2+pixsize,Ang_err_cut,Ang_err_bins)
npix_y=int(ROI/pixsize)

def Src_templates(profile, psf_model = 'real'):
    models=np.zeros((Rbins,2*Npix+1,2*Npix+1))
    kernels=np.zeros((Ang_err_bins,2*Npix+1,2*Npix+1))
    x=np.linspace(-ROI,ROI,2*Npix+1)
    y=np.linspace(-ROI,ROI,2*Npix+1)

    # We create source model templates first:
    for i in range(2*Npix+1):
        for j in range(2*Npix+1):
            r=np.sqrt(x[i]**2+y[j]**2)
            for k in range(Rbins):
                if(profile=='gaussian'):
                    models[k,i,j]=1/((2*np.pi)*(radii[k])**2)*np.exp(-(r/radii[k])**2/2.)
                elif(profile=='disk'):
                    if(r<radii[k]):
                        models[k,i,j]=1/(np.pi*((radii[k]))**2)

    #Check normalisation of the disk models:
    for k in range(Rbins):
        tmp=np.sum(models[k]*pixsize**2)
        models[k]=models[k]/tmp
    
    # We create gaussian kernels with which we will "blur" the models:
    if psf_model == 'gaussian':
        for i in range(2*Npix+1):
            for j in range(2*Npix+1):
                r=np.sqrt(x[i]**2+y[j]**2)
                for k in range(Ang_err_bins):
                    kernels[k,i,j]=1/((2*np.pi)*(ang_errs[k])**2)*np.exp(-(r/ang_errs[k])**2/2.)
    elif psf_model == 'real':
        for i in range(2*Npix+1):
            for j in range(2*Npix+1):
                r=np.sqrt(x[i]**2+y[j]**2)
                r_ind = max(np.argmax(r < bins_psf_max), 14) # TODO: it's very hacky
                for k in range(Ang_err_bins):
                    log_emu = 3.1
                    kernels[k,i,j]=psf['6y'](log_emu, 41, ang_errs[k], 3.)[r_ind] / np.sqrt(bins_psf_max[r_ind]*bins_psf_min[r_ind]) 
                    # TODO: variable energy 
    else:
        raise NotImplementedError
        

    #Check normalisation of the kernels:
    for k in range(Ang_err_bins):
        tmp=np.sum(kernels[k]*pixsize**2)
        kernels[k]=kernels[k]/tmp

    # Now we blur the models with the kernels:
    models_smoothed=np.zeros((Ang_err_bins,Rbins,2*Npix+1,2*Npix+1))
    for i in range(Ang_err_bins):
        for  k in range(Rbins):
            tmp=signal.fftconvolve(models[k],kernels[i], mode='same')
            models_smoothed[i,k]=tmp/np.sum(tmp*pixsize**2)

    return models_smoothed
# %%
gaussian_models=Src_templates('gaussian')

#%%
#disk_models=Src_templates('disk')

# %%

def space_sness(ra, dec, sigma,ra_src, dec_src, R_src, profile="gaussian"):
    res=[]
    for i in range(len(ra)):
        ind_sig=((sigma[i]-0.2+pixsize)/pixsize).astype(int)-1 ## <------
        ind_r=int(R_src/pixsize)-1
        dra=ra[i]-ra_src
        ddec=dec[i]-dec_src
        ind_x=(Npix+dra*np.cos(np.deg2rad(dec_src))/ROI*Npix).astype(int)-1
        ind_y=(Npix+ddec/ROI*Npix).astype(int)-1
        ind_x=ind_x*(ind_x<2*Npix)
        ind_y=ind_y*(ind_y<2*Npix)
        if(profile=='gaussian'):
            tmp1=gaussian_models[ind_sig,ind_r,ind_x,ind_y] 
        elif(profile=='disk'):
            tmp1=disk_models[ind_sig,ind_r,ind_x,ind_y]
        res.append(tmp1 * 3282.8) # conv. to sr

    return res


# FIXME: psf should be nearly gaussian on 10 TeV 
# # is it ever possible?
