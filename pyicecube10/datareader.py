import os
import pandas as pd
from . import years, data_dir
import numpy as np
from scipy.stats import rv_histogram
from numpy.random import default_rng
rng = default_rng()
from functools import lru_cache

events = {}
for year in years:
    fn = os.path.join(data_dir, 'events', f'IC{year}_exp.csv')
    with open(fn, 'r') as fd:
        s = fd.readline()
        names = s[1:].strip().split()
    events[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names=names)
events['6y'] = pd.concat([events[i] for i in ('86_II', '86_III', '86_IV', '86_V', '86_VI', '86_VII')]).reset_index().drop('index', axis=1)

uptime = {}
for year in years:
    fn = os.path.join(data_dir, 'uptime', f'IC{year}_exp.csv')
    with open(fn, 'r') as fd:
        s = fd.readline()
        names = s[1:].strip().split()
    uptime[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names = names)

exptime = {k: sum(v['MJD_stop[days]'] - v['MJD_start[days]']) * 86400 for k, v in uptime.items()}
exptime['6y'] = sum(v for k, v in exptime.items() if k in ('86_II', '86_III', '86_IV', '86_V', '86_VI', '86_VII') )


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
        Aeff[year] = lambda dec: nu_area[year].\
            query("`Dec_nu_min[deg]` <= @dec < `Dec_nu_max[deg]` and `log10(E_nu/GeV)_min` < 9.0").\
                groupby('log10(E_nu/GeV)_min')['A_Eff[cm^2]'].apply(sum)
    else:
        Aeff[year] = Aeff['86_II']
Aeff['6y'] = Aeff['86_II']

smearing_raw = {}
for year in years:
    if year in ('40', '59', '79', '86_I', '86_II'):
        fn = os.path.join(data_dir, 'irfs', f'IC{year}_smearing.csv')
        with open(fn, 'r') as fd:
            s = fd.readline()
            names = s[1:].strip().split()
        smearing_raw[year] = pd.read_csv(fn, delim_whitespace=True, comment='#', names=names)
        # TODO special treatment for southern events in early years (see first rows until 86_I)
    
        #fix buggy bin (will also remove abovementioned rows, but then table structure is a corrupted)
        smearing_raw[year].query("`AngErr_min[deg]` != `AngErr_max[deg]`", inplace = True)
    else:
        smearing_raw[year] = smearing_raw['86_II']    


# FIXME:   YEAR INSIDE TEMP FUCTIONS ARE GLOBAL VARIABLES 
# probably make factory function

def pdf_factory(iyear):
    if iyear in ('40', '59', '79', '86_I', '86_II'):
        def tmp(logEnu, dec, beta, dist):
            if dist is None:
                psf_corr = ""
            else:
                psf_corr = " and `PSF_max[deg]` < @dist"
            pdf = smearing_raw[iyear].query("`log10(E_nu/GeV)_min` <= @logEnu < `log10(E_nu/GeV)_max`\
                                            and `Dec_nu_min[deg]` <= @dec < `Dec_nu_max[deg]`\
                                            and `AngErr_max[deg]` < @beta" + psf_corr).\
                                                groupby('log10(E/GeV)_min').\
                                                    aggregate({'log10(E/GeV)_max': 'mean', 'Fractional_Counts': 'sum'}).\
                                                        reset_index()
            return pdf 
        return tmp
    else:
        return pdf_factory('86_II')
pdf = {year: pdf_factory(year) for year in years}
pdf['6y'] = pdf['86_II']

#depending on E_nu there are different binning over log10_E
#let's rebin log10E by 0.2 in the range 0..10
binning_min, binning_max, step = 0, 10, 0.2
bins_E_min, bins_E_max = np.arange(binning_min, binning_max, step), np.arange(binning_min + step, binning_max + step, step)
bins_E_mean = (bins_E_min + bins_E_max) / 2

# rebinning by sampling
def rmf_factory(jyear):
    N_samples = 1000000
    if jyear in ('40', '59', '79', '86_I', '86_II'):
        def tmp_rmf(dec, beta = 20, dist = None):
            rebinned_pdfs = []
            for Enu in bins_Enu_mean:
                pdf_current = pdf[jyear](Enu, dec, beta, dist)
                prob_good = pdf_current['Fractional_Counts'].sum()
                norm_pdf = pdf_current['Fractional_Counts'].ravel() / prob_good
                boundaries = np.append(pdf_current['log10(E/GeV)_min'].ravel(), pdf_current['log10(E/GeV)_max'].iloc[-1])
                distribution = rv_histogram( (norm_pdf, boundaries) )

                is_good = rng.random(N_samples) < prob_good 
                samples = distribution.rvs(size=N_samples)
                samples = samples[is_good]
                rebinned_pdfs.append(np.histogram(samples, np.append(bins_E_min, bins_E_max[-1]))[0] / N_samples)
            return np.stack(rebinned_pdfs).transpose()
        return tmp_rmf
    else:
        return rmf_factory('86_II')
rmf = {year: rmf_factory(year) for year in years}
rmf['6y'] = rmf['86_II']

