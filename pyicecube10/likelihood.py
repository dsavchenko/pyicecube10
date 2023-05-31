from os.path import isfile
import numpy as np
from .datareader import (events, 
                         exptime, 
                         rmf, 
                         pdf,
                         Aeff, 
                         bins_Enu_min, 
                         bins_Enu_max, 
                         bins_Enu_mean, 
                         bins_E_min, 
                         bins_E_max, 
                         bins_E_mean)
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache
from . import pdf_store_dir
import os
from numpy.random import default_rng
import logging
from scipy.stats import rv_histogram
logger = logging.getLogger(__name__)

rng = default_rng()

Enu_min, Enu_max = 10**bins_Enu_min, 10**bins_Enu_max
Enu = 10**bins_Enu_mean
Emu_min, Emu_max = 10**bins_E_min, 10**bins_E_max

def get_dist(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.deg2rad, [ra1, dec1, ra2, dec2])

    dra = ra2 - ra1
    ddec = dec2 - dec1

    a = np.sin(ddec/2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2.0)**2

    return 2 * np.arcsin(np.sqrt(a)) 

def space_signalness(ra, dec, sigma, ra_src, dec_src, R_src):     
        dist = get_dist(ra, dec, ra_src, dec_src)
        sig, rsrc = np.deg2rad(sigma), np.deg2rad(R_src)
        return 1/(2*np.pi * (sig**2 + rsrc**2)) * np.exp( - dist**2 / (sig**2 + rsrc**2) / 2 )

@lru_cache
def powerlaw(gamma, A=1e-10, Enorm = 1e3):
    return A * 1e-3 * (Enu / Enorm)**(-gamma) # NOTE: normalisation now at TeV (was at 100 TeV): 1e-20 et 1e5
                                              # NOTE: A is in 1 / (TeV cm^2 s); thats why 1e-3. Overall is GeV

def blind():
    global blind_events
    blind_events = {}
    for year in events.keys():
        blind_events[year] = events[year].copy()
        blind_events[year]['RA[deg]'] = rng.permutation(blind_events[year]['RA[deg]']) 
blind()

def inject(N_events = 50, ra = 307.65, dec = 40.93, width = 0.5):
    global blind_events
    for year in events.keys():
        # events should be constrained by dec_min dec_max 
        beta_distr = rv_histogram(np.histogram(blind_events['year']['AngErr[deg]'])) 
        # TODO:         
        

class LikelihoodAnalysisSinglePeriod:
    def __init__(self, 
                 year, 
                 angular_reconstruction_cut = 20, 
                 use_pdf_cache=True, 
                 dec_min = 0, 
                 dec_max = 90, 
                 unblind = True, 
                 reshuffle = False,
                 src_model = powerlaw):  # _en_signalness_factory isn't universal so other model may not work
        
        if unblind:
            self.events = events[year].copy()
        else:
            self.events = blind_events[year]
        
        self.events = self.events.query('`AngErr[deg]` < @angular_reconstruction_cut')\
                                 .query('`Dec[deg]` >= @dec_min')\
                                 .query('`Dec[deg]` <= @dec_max')\
                                 .query('`log10(E/GeV)` >= 2') \
                                 .reset_index().drop('index', axis=1)
        if reshuffle:
            self.events['RA[deg]'] = rng.permutation(self.events['RA[deg]'])
                                  
        self.exptime = exptime[year]
        self.year = year
        self.angular_reconstruction_cut = angular_reconstruction_cut
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.use_pdf_cache = use_pdf_cache
        self.src_model = src_model
        
        self.backgroundness_func = self._backgroundness_factory()
        self.en_signalness_func = self._en_signalness_factory()
       
    @property
    def ntot(self):
        return len(self.events)
        
    def _backgroundness_factory(self):     
        # compute background pdf in strips over dec; now with energy dependence
        deltadec = 1 #deg
        back_bins_s_edges = np.arange(self.dec_min, self.dec_max+0.1, 2 * deltadec)
        back_bins_s_c = ( back_bins_s_edges[1:] + back_bins_s_edges[:-1] ) / 2

        omegas = 4 * np.pi * np.sin(np.deg2rad(deltadec)) * np.cos(np.deg2rad(back_bins_s_c))

        #back_bins_en_edges = np.linspace(1, 7.6, 34)
        back_bins_en_edges = np.concatenate([bins_E_min[:-1], bins_E_max[-1:]])
        en_binwidth = back_bins_en_edges[1] - back_bins_en_edges[0]
        back_bins_en_c = ( back_bins_en_edges[1:] + back_bins_en_edges[:-1] ) / 2

        hst = np.histogram2d(self.events['Dec[deg]'], self.events['log10(E/GeV)'], bins=(back_bins_s_edges, back_bins_en_edges))

        bkgrnds = hst[0] / self.ntot / omegas[:,np.newaxis] / en_binwidth

        backgroundness = RegularGridInterpolator((back_bins_s_c, back_bins_en_c), bkgrnds, method = 'nearest', bounds_error=False, fill_value=None)
        return lambda *x: backgroundness(x)
    
    def rmf(self, dec):
        return rmf[self.year](dec, self.angular_reconstruction_cut, None)
    
    def acc(self, dec):
        return (Enu_max - Enu_min) * Aeff[self.year](dec).ravel() * exptime[self.year]
    
    def count_spec(self, dec, *args):
        flux = self.src_model(*args)
        nu_counts = flux * self.acc(dec)
        return np.matmul(self.rmf(dec), nu_counts)
    
    def en_pdf(self, dec, *args):
        mu_counts = self.count_spec(dec, *args)
        return mu_counts / sum(mu_counts) / (bins_E_max - bins_E_min)
    
    def _en_signalness_factory(self):
        gammas = np.linspace(1, 5, 41)
        decs = np.array([#-89.99, -81.87 , -70.335, -64.285, -59.39 , -55.135, -51.295, -47.755,
                        #-44.445, -41.315, -38.33 , -35.465, -32.695, -30.01 , -27.395,
                        #-24.84 , -22.34 , -19.88 , -17.46 , -15.075, -12.715, -10.375,
                        -8.05 ,  -5.74 ,  -3.44 ,  -1.145,   1.145,   3.44 ,   5.74 ,
                        8.05 ,  10.375,  12.715,  15.075,  17.46 ,  19.88 ,  22.34 ,
                        24.84 ,  27.395,  30.01 ,  32.695,  35.465,  38.33 ,  41.315,
                        44.445,  47.755,  51.295,  55.135,  59.39 ,  64.285,  70.335,
                        81.87, 89.99])
       
        sg_pdf_file = os.path.join(pdf_store_dir, f'sgnlns-{self.year}-{self.angular_reconstruction_cut}.npy')          
        if self.use_pdf_cache and os.path.isfile(sg_pdf_file):
            logger.info('Using cached signalness table %s', sg_pdf_file)
            dec_en_signalness = np.load(sg_pdf_file)
        else:
            dec_en_signalness = []
            for i, dec in enumerate(decs):
                dec_en_signalness.append([])
                for gamma in gammas:
                    dec_en_signalness[-1].append(self.en_pdf(dec, gamma))
            dec_en_signalness = np.array(dec_en_signalness)
            
            if self.use_pdf_cache:
                os.makedirs(os.path.dirname(sg_pdf_file), exist_ok=True)
                logger.info('Caching signalness table %s', sg_pdf_file)
                np.save(sg_pdf_file, dec_en_signalness)
        
        sgnlns = RegularGridInterpolator((decs, gammas, bins_E_mean), dec_en_signalness, method='nearest', bounds_error=False, fill_value=None)
        return lambda *x: sgnlns(x)
    
    def signalness_func(self, ra, dec, sigma, ra_src, dec_src, R_src, log10_En, gamma):
        return space_signalness(ra, dec, sigma, ra_src, dec_src, R_src) * self.en_signalness_func(dec_src, gamma, log10_En)

    def set_src_parameters(self, ra_src, dec_src, R_src, gamma):
        self.ra_src = ra_src
        self.dec_src = dec_src
        self.R_src = R_src
        self.gamma = gamma
        
        mask = np.rad2deg(get_dist(self.events['RA[deg]'], self.events['Dec[deg]'], ra_src, dec_src)) < 5 * (self.events['AngErr[deg]'] + R_src)
        self._sel_events = self.events[mask].reset_index().drop('index', axis=1)
                
        self.sel_backgroundness = self.backgroundness_func(self._sel_events['Dec[deg]'], self._sel_events['log10(E/GeV)'])
        zero_back = self.sel_backgroundness == 0
        if zero_back.any():
            for idx in np.where(zero_back)[0]:
                self.sel_backgroundness[idx] = max(self.backgroundness_func(self._sel_events.iloc[idx]['Dec[deg]'], 
                                                                            self._sel_events.iloc[idx]['log10(E/GeV)']+0.001),
                                                   self.backgroundness_func(self._sel_events.iloc[idx]['Dec[deg]'], 
                                                                            self._sel_events.iloc[idx]['log10(E/GeV)']-0.001),
                                                   1e-9)
            
                    
        self.sel_signalness = self.signalness_func( self._sel_events['RA[deg]'], 
                                                self._sel_events['Dec[deg]'], 
                                                self._sel_events['AngErr[deg]'], 
                                                ra_src, 
                                                dec_src, 
                                                R_src, 
                                                self._sel_events['log10(E/GeV)'], 
                                                gamma )
        
        self.signal_yield = self._calc_signal_yield(dec_src, gamma)
        
        
    
    def _calc_signal_yield(self, dec_src, *en_args):
        flux = self.src_model(*en_args)
        nu_counts = flux * self.acc(dec_src) # TODO: don't we need to multiply by rmf here? e.g. to account for beta cut
        return sum(nu_counts)

    def TS(self, ns):
        return 2 * ( np.log(ns * self.sel_signalness / self.ntot + (1 - ns/self.ntot) * self.sel_backgroundness).sum()\
                   - np.log(self.sel_backgroundness).sum()\
                   + (self.ntot - len(self.sel_signalness))*np.log(1 - ns / self.ntot) )

    def TS_norm(self, A):
        ns = np.rint(self.count_spec(self.dec_src, self.gamma, A).sum())
        return self.TS(ns)
        
    
class LikelihoodAnalysisMultiPeriod:
    def __init__(self, 
                 years, 
                 angular_reconstruction_cut = 20, 
                 use_pdf_cache=True, 
                 dec_min = 0, 
                 dec_max = 90, 
                 unblind = True, 
                 reshuffle=False,
                 src_model = powerlaw):
        self.period_analysis_list = [LikelihoodAnalysisSinglePeriod(year=year, 
                                                                    angular_reconstruction_cut=angular_reconstruction_cut,
                                                                    use_pdf_cache=use_pdf_cache,
                                                                    dec_min=dec_min,
                                                                    dec_max=dec_max,
                                                                    unblind=unblind,
                                                                    reshuffle=reshuffle,
                                                                    src_model = src_model) for year in years]
    
    def set_src_parameters(self, *args):
        for p in self.period_analysis_list:
            p.set_src_parameters(*args)
    
    def count_spec(self, *args):
        return sum([p.count_spec(*args) for p in self.period_analysis_list])
    
    def TS(self, ns):
        signal_yields = [p.signal_yield for p in self.period_analysis_list]
        numbers = [np.round(ns * sy / sum(signal_yields)) for sy in signal_yields]
        numbers[-1] = ns - sum(numbers[:-1])
        ts_per_period = [p.TS(numbers[i]) for i, p in enumerate(self.period_analysis_list)]
        return sum(ts_per_period)
            
    def TS_norm(self, A):
        return sum(p.TS_norm(A) for p in self.period_analysis_list)
