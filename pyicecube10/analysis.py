from .datareader import events, Aeff, rmf, exptime
from .datareader import bins_Enu_min, bins_Enu_mean, bins_Enu_max
from .datareader import bins_E_min, bins_E_mean, bins_E_max
from functools import cached_property, lru_cache
import numpy as np
from astropy.coordinates import SkyCoord
import inspect

Enu_min, Enu_max = 10**bins_Enu_min, 10**bins_Enu_max
Emu_min, Emu_max = 10**bins_E_min, 10**bins_E_max
Enu = (Enu_max + Enu_min) / 2
Emu = (Emu_max + Emu_min) / 2 

energybins = dict(Enu_min = Enu_min, 
                  Enu_max = Enu_max, 
                  Enu = Enu,
                  Emu_min = Emu_min, 
                  Emu_max = Emu_max, 
                  Emu = Emu)

coords = {}
for year in events.keys():
    coords[year] = SkyCoord(ra=events[year]['RA[deg]'],
                            dec = events[year]['Dec[deg]'],
                            frame='fk5', unit='deg')

class AnalyzeCircle:
    def __init__(self, 
                 src_ra, 
                 src_dec, 
                 radius, 
                 beta, 
                 years=('86_I', '86_II', '86_III', '86_IV', '86_V', '86_VI', '86_VII'),
                 back_method = 'band'):
        
        self.src_ra, self.src_dec = src_ra, src_dec # deg
        self.radius = radius # deg
        self.beta = beta # deg, quality cut
        self.years = years
        self.src_coord = SkyCoord(ra=src_ra, dec=src_dec, frame='fk5', unit='deg')
        
        self.back_method = back_method
        if back_method == 'circles':
            try:
                n_b = int(SkyCoord(0, src_dec, unit='deg').separation(SkyCoord(180, src_dec, unit='deg')).deg // radius)
                self._background_shifts = np.linspace(0, 360, n_b, endpoint=False)[1:]
                self.alpha_back = 1 / len(self._background_shifts)
            except ZeroDivisionError:
                raise ValueError("Unable to estimate background: radius or dec too high.")
        elif back_method == 'band':
            area_b = 2 * np.pi * (np.sin(np.deg2rad(src_dec + radius)) - np.sin(np.deg2rad(src_dec - radius)))
            area_r = 2 * np.pi * (1 - np.cos(np.deg2rad(radius)))
            self.alpha_back = area_r / area_b 
        else:
            raise NotImplementedError('Unknown background estimation method %s' % back_method)

        # # don't use lru_cache decorators, it prevents garbage collector from destroying objects
        # # https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
        # # instead:
        # self._good_masks = lru_cache()(self.__good_masks)
        # self._good_events = lru_cache()(self.__good_events)
        # self._coords = lru_cache()(self.__coords)
        # self._dist = lru_cache()(self.__dist)
        # self.acc = lru_cache()(self._acc)
        # self.rmf = lru_cache()(self._rmf)
        # self.total_per_bin = lru_cache()(self._total_per_bin)
        # self.background_per_bin = lru_cache()(self._background_per_bin)
    
    @lru_cache
    def _good_masks(self):
        masks = {}
        for year in self.years:
            masks[year] = events[year]['AngErr[deg]'] < self.beta
        return masks
    
    @lru_cache
    def _good_events(self):
        good_events = {}
        for year in self.years:
            good_events[year] = events[year][self._good_masks()[year]]
        return good_events
    
    @lru_cache()
    def _coords(self):
        c_coords = {}
        for year in self.years:
            c_coords[year] = coords[year][self._good_masks()[year]]
        return c_coords
    
    @lru_cache
    def _dist(self):
        dist = {}
        for year in self.years:
            dist[year] = self._coords()[year].separation(self.src_coord).deg
        return dist
    
    @lru_cache
    def acc(self):
        acc = {}
        for year in self.years:
            acc[year] = (Enu_max - Enu_min) * Aeff[year](self.src_dec).ravel() * exptime[year] 
        return acc
    
    @lru_cache
    def rmf(self):
        c_rmf = {}
        for year in self.years:
            c_rmf[year] = rmf[year](self.src_dec, self.beta, self.radius)
        return c_rmf
    
    @lru_cache
    def total_per_bin(self):
        binning = np.append(bins_E_min, bins_E_max[-1])
        total = np.zeros_like(bins_E_min)
        for year in self.years:
            mask = self._dist()[year] < self.radius
            binned = np.histogram(self._good_events()[year][mask]['log10(E/GeV)'], binning)[0]
            total += binned
        return total

    @lru_cache
    def background_per_bin(self):
        if self.back_method == 'circles':
            return self._background_per_bin_circle()
        elif self.back_method == 'band':
            return self._background_per_bin_band()
        
    def _background_per_bin_band(self):
        binning = np.append(bins_E_min, bins_E_max[-1])
        
        bg = np.zeros_like(bins_E_min)
        for year in self.years:
            mask = np.logical_and(self._good_events()[year]['Dec[deg]'] > self.src_dec - self.radius,
                                  self._good_events()[year]['Dec[deg]'] < self.src_dec + self.radius)
            binned = np.histogram(self._good_events()[year][mask]['log10(E/GeV)'], binning)[0]
            bg += binned
        bg *= self.alpha_back
        return bg
    
    def _background_per_bin_circle(self):
        binning = np.append(bins_E_min, bins_E_max[-1])
        
        bg = np.zeros_like(bins_E_min)
        for shift in self._background_shifts:
            for year in self.years:
                bg_coord = SkyCoord(ra = self.src_ra + shift, dec = self.src_dec, unit='deg')
                mask = self._coords()[year].separation(bg_coord).deg < self.radius
                binned = np.histogram(self._good_events()[year][mask]['log10(E/GeV)'], binning)[0]
                bg += binned
        bg *= self.alpha_back
        return bg

    def cumulative_total(self):
        return np.flip(np.cumsum(np.flip(self.total_per_bin())))
    
    def cumulative_background(self):
        return np.flip(np.cumsum(np.flip(self.background_per_bin())))
    
    def expected_counts(self, func, *args):
        flux = func(*args)
        total_mu_counts = np.zeros_like(bins_E_mean)
        for year in self.years:
            nu_counts = flux * self.acc()[year]
            mu_counts = np.matmul(self.rmf()[year], nu_counts)
            total_mu_counts += mu_counts
        return total_mu_counts

    def _clear_caches(self):
        for attr_n in dir(self):
            attr = getattr(self, attr_n)
            if hasattr(attr, 'cache_clear'):
                attr.cache_clear()
