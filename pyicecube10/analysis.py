from .datareader import events, Aeff, rmf, exptime
from .datareader import bins_Enu_min, bins_Enu_mean, bins_Enu_max
from .datareader import bins_E_min, bins_E_mean, bins_E_max
from functools import cached_property
import numpy as np
from astropy.coordinates import SkyCoord


Enu_min, Enu_max, Enu = 10**bins_Enu_min, 10**bins_Enu_max, 10**bins_Enu_mean
Emu_min, Emu_max, Emu = 10**bins_E_min, 10**bins_E_max, 10**bins_E_mean

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
                 years=('86_I', '86_II', '86_III', '86_IV', '86_V', '86_VI', '86_VII')):

        self.src_ra, self.src_dec = src_ra, src_dec # deg
        self.radius = radius # deg
        self.beta = beta # deg, quality cut
        self.years = years
        self.src_coord = SkyCoord(ra=src_ra, dec=src_dec, frame='fk5', unit='deg')

    @cached_property
    def _good_masks(self):
        masks = {}
        for year in self.years:
            masks[year] = events[year]['AngErr[deg]'] < self.beta
        return masks
    
    @cached_property
    def _good_events(self):
        good_events = {}
        for year in self.years:
            good_events[year] = events[year][self._good_masks[year]]
        return good_events
    
    @cached_property
    def _coords(self):
        c_coords = {}
        for year in self.years:
            c_coords[year] = coords[year][self._good_masks[year]]
        return c_coords
    
    @cached_property
    def acc(self):
        acc = {}
        for year in self.years:
            acc[year] = Aeff[year](self.src_dec).ravel() * exptime[year] * (Enu_max - Enu_min)
        return acc
        
    @cached_property
    def rmf(self):
        c_rmf = {}
        for year in self.years:
            c_rmf[year] = rmf[year](self.src_dec, self.beta)
        return c_rmf
    
    def total_per_bin(self):
        binning = np.append(bins_E_min, bins_E_max[-1])
        total = np.zeros_like(bins_E_min)
        for year in self.years:
            mask = self._coords[year].separation(self.src_coord).deg < self.radius
            binned = np.histogram(self._good_events[year][mask]['log10(E/GeV)'], binning)[0]
            total += binned
        return total

    def background_per_bin(self):
        binning = np.append(bins_E_min, bins_E_max[-1])
        shifts = range(5, 360, 5)
        bg = np.zeros_like(bins_E_min)
        for shift in shifts:
            for year in self.years:
                bg_coord = SkyCoord(ra = self.src_ra + shift, dec = self.src_dec, unit='deg')
                mask = self._coords[year].separation(bg_coord).deg < self.radius
                binned = np.histogram(self._good_events[year][mask]['log10(E/GeV)'], binning)[0]
                bg += binned
        bg /= len(shifts)
        return bg

    def cumulative_total(self):
        return np.flip(np.cumsum(np.flip(self.total_per_bin())))
    
    def cumulative_background(self):
        return np.flip(np.cumsum(np.flip(self.background_per_bin())))
    
    def expected_counts(self, func, *args):
        flux = func(*args)
        total_mu_counts = np.zeros_like(bins_E_mean)
        for year in self.years:
            nu_counts = flux * self.acc[year]
            mu_counts = np.matmul(self.rmf[year], nu_counts)
            total_mu_counts += mu_counts
        return total_mu_counts
