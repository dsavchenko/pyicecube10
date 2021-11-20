import os
import pandas as pd
from . import years, data_dir
import numpy as np
from scipy.stats import rv_histogram
from numpy.random import default_rng
rng = default_rng()

events = {}
for year in years:
    fn = os.path.join(data_dir, 'events', f'IC{year}_exp.csv')
    with open(fn, 'r') as fd:
        s = fd.readline()
        names = s[1:].strip().split()
    events[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names=names)

uptime = {}
for year in years:
    fn = os.path.join(data_dir, 'uptime', f'IC{year}_exp.csv')
    with open(fn, 'r') as fd:
        s = fd.readline()
        names = s[1:].strip().split()
    uptime[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names = names)

exptime = {k: sum(v['MJD_stop[days]'] - v['MJD_start[days]']) * 86400 for k, v in uptime.items()}

nu_area = {}
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        fn = os.path.join(data_dir, 'irfs', f'IC{year}_effectiveArea.csv')
        with open(fn, 'r') as fd:
            s = fd.readline()
            names = s[1:].strip().split()
        nu_area[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names=names)
    else:
        nu_area[year] = nu_area['86_II']

# default binning in neutrino energy space will be the same as in a_eff
bins_Enu_min = nu_area['86_I']['log10(E_nu/GeV)_min'].unique()
bins_Enu_min = bins_Enu_min[bins_Enu_min<9.0] #we will see that a_eff = 0 afterwards
bins_Enu_max = nu_area['86_I']['log10(E_nu/GeV)_max'].unique()
bins_Enu_max = bins_Enu_max[:len(bins_Enu_min)]
bins_Enu_mean = (bins_Enu_max + bins_Enu_min) / 2

Aeff = {}
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        Aeff[year] = lambda dec: nu_area[year][np.logical_and(
                                                              np.logical_and(nu_area[year]['Dec_nu_min[deg]'] <= dec, 
                                                                             dec < nu_area[year]['Dec_nu_max[deg]']),
                                                             nu_area[year]['log10(E_nu/GeV)_min'] < 9.0 ) # effective area is always 0 above
                                              ].groupby('log10(E_nu/GeV)_min')['A_Eff[cm^2]'].apply(sum)
        
    else:
        Aeff[year] = Aeff['86_II']
        
smearing_raw = {}
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        fn = os.path.join(data_dir, 'irfs', f'IC{year}_smearing.csv')
        with open(fn, 'r') as fd:
            s = fd.readline()
            names = s[1:].strip().split()
        smearing_raw[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names=names)
    else:
        smearing_raw[year] = smearing_raw['86_II']    
        
pdf = {}
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        def tmp(logEnu, dec, beta):
            mask = np.logical_and(np.logical_and(
                            np.logical_and(smearing_raw[year]['log10(E_nu/GeV)_min'] <= logEnu, logEnu < smearing_raw[year]['log10(E_nu/GeV)_max']),
                            np.logical_and(smearing_raw[year]['Dec_nu_min[deg]'] <= dec, dec < smearing_raw[year]['Dec_nu_max[deg]'])),
                        smearing_raw[year]['AngErr_max[deg]'] < beta)
            pdf = smearing_raw[year][mask].groupby('log10(E/GeV)_min').aggregate({'log10(E/GeV)_max': 'mean', 'Fractional_Counts': 'sum'}).reset_index() # here we sum over PSF and AngErr below cut
            return pdf 
        pdf[year] = tmp
    else:
        pdf[year] = pdf['86_II']
        

#depending on E_nu there are different binning over log10_E
#let's rebin log10E by 0.2 in the range 0..10
binning_min, binning_max, step = 0, 10, 0.2
bins_E_min, bins_E_max = np.arange(binning_min, binning_max, step), np.arange(binning_min + step, binning_max + step, step)
bins_E_mean = (bins_E_min + bins_E_max) / 2

# rebinning by sampling
rmf = {}
N_samples = 10000
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        def tmp_rmf(dec, beta):
            rebinned_pdfs = []
            for Enu in bins_Enu_mean:
                pdf_current = pdf[year](Enu, dec, beta)
                prob_good = pdf_current['Fractional_Counts'].sum()
                norm_pdf = pdf_current['Fractional_Counts'].ravel() / prob_good
                boundaries = np.append(pdf_current['log10(E/GeV)_min'].ravel(), pdf_current['log10(E/GeV)_max'].iloc[-1])
                distribution = rv_histogram( (norm_pdf, boundaries) )

                is_good = rng.random(N_samples) < prob_good 
                samples = distribution.rvs(size=N_samples)
                samples = samples[is_good]
                rebinned_pdfs.append(np.histogram(samples, np.append(bins_E_min, bins_E_max[-1]))[0] / N_samples)
            return np.stack(rebinned_pdfs).transpose()
        rmf[year] = tmp_rmf
    else:
        rmf[year] = rmf['86_II']