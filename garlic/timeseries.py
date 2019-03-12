import numpy as np
import logging
import time
import psutil
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.stats import norm, truncnorm
from scipy.special import gammaln, gamma
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import binned_statistic
from multiprocessing import Pool, cpu_count

# From scikit-learn utilities:
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
' instance' % seed)

def opt_bins(data,maxM=100):
    """
    Python version of the 'optBINS' algorithm by Knuth et al. (2006) - finds 
    the optimal number of bins for a one-dimensional data set using the 
    posterior probability for the number of bins. WARNING sometimes doesn't
    seem to produce a high enough number by some way...

    Adopted from https://github.com/samconnolly/DELightcurveSimulation/blob/master/DELCgen.py
                            
    Parameters:
    -----------
    data: `~numpy.ndarray
        The data set to be binned
    maxM: int, optional
         The maximum number of bins to consider
                                                            
    Returns
    -------
    maximum:  int
        The optimum number of bins

    Notes
    -----
    Ref: https://arxiv.org/pdf/physics/0605197.pdf
    """
    N = len(data)
    # loop through the different numbers of bins
    # and compute the posterior probability for each.
                                                                                                               
    logp = np.zeros(maxM)
    for M in range(1,maxM+1):
        n = np.histogram(data,bins=M)[0] # Bin the data (equal width bins)
        # calculate posterior probability
        part1 = N * np.log(M) + gammaln(M/2.0)
        part2 = - M * gammaln(0.5)  - gammaln(N + M/2.0)
        part3 = np.sum(gammaln(n+0.5))
        logp[M-1] = part1 + part2 + part3 # add to array of posteriors
        maximum = np.argmax(logp) + 1 # find bin number of maximum probability
    return maximum + 10 


def window_function(t, T, type = "hanning"):
    """Calculate window function"""
    m = (t >= 0.) & (t <= T)
    logging.debug("t size: {0:n}, sum mask: {1:n}".format(t.size, m.sum()))
    result = np.zeros_like(t)
    if type == "hanning":
        result[m] = np.cos(np.pi * (t[m] - 0.5 * T) / T)**2.
    elif type == "hann":
        result[m] = np.sin(np.pi * t[m] / T)**2.
    elif type == "rect":
        result[m] = np.ones(np.sum(m))
    elif type == "none":
        result = np.ones_like(t)
    return result

def var_normalization(fsim, axis = -1):
    """
    Normalize the variances of simulated light curves to 1.

    Parameters
    ----------
    fsim: `~numpy.ndarray`
    (n [[x m x] k])-dimensional array with simulated light curves

    axis: int
        axis over which variance is calculated 

    Returns
    -------
    array with same shape as input array with simulated light curves 
    normalized to variance = 1
    """
    result = (fsim.T / np.sqrt(fsim.var(axis = axis).T)).T
    return result

# spectral models 
power_law = lambda nu, **p: p['norm'] * (nu / p['nu0'])**(-p['beta']) + p['const']
def broken_power_law(nu, **p):
    m = nu >= p['nubreak']
    result = np.zeros_like(nu)
    result[m] = p['norm'] * (nu[m] / p['nubreak'])**(-p['beta_hi'])
    result[~m] = p['norm'] * (nu[~m] / p['nubreak'])**(-p['beta_lo'])
    result += p['const']
    return result

def interp_lc(t,f,interp_tstep, interpolation = 'linear', lctype = 'density', axis = -1):
    """Interpolate the light curve on linear grid

    Parameters
    ----------
    t: `~numpy.ndarray`
        time points for interpolation, 1d array.

    f: `~numpy.ndarray`
        flux values for interpolation, 1d or 2d array.

    {options}

    axis: int
        axis for interpolation if f is 2d array

    lctype: str, either density or average
        if density, use linear interpolation at new points, 
        otherwise, form average
    
    """
    if interpolation == 'none':
        ti = t
        fi = f

    else:
        #print t.shape, f.shape
        interp = interp1d(t, f, kind = interpolation,
                            fill_value = 'extrapolate', axis = axis)
        # new regular grid points 
        if interp_tstep < 1.: 
            logging.warning("Interp step < 1, setting to 1.")
            interp_tstep = 1.
        ti = np.linspace(t.min(), t.max(), int(np.floor((t.max() - t.min()) / interp_tstep)) + 1)

        if lctype == 'density':
            fi = interp(ti)
#
        elif lctype == 'average':
            for ix, xi in enumerate(ti):
                if not ix:
                    xx = np.linspace(xi, xi + interp_tstep / 2., 10)
                elif ix == ti.size - 1:
                    xx = np.vstack([xx, np.linspace( xi - interp_tstep / 2.,
                            xi, 11, endpoint = True)[1:]])
                else:
                    xx = np.vstack([xx, np.linspace( xi - interp_tstep / 2.,
                            xi + interp_tstep / 2., 11)[1:]])
            fi = interp(xx).mean(axis = -1)
        else:
            raise ValueError("lctype unknown")

    dti = np.diff(ti)
    return ti, fi, dti


logmean = lambda x : (np.log10(x).sum() / x.size)
logvar = lambda x : (((np.log10(x) - np.log10(x).sum() / x.size)**2.).sum() / x.size**2. )

class TimeSeries(object):
    def __init__(self, t, f, e, dt = None,
        interpolation = 'nearest', lctype = 'average'): 
        #interp_tstep = None):
        """
        Initialize the time series class

        Parameters
        ----------
        t: `~numpy.ndarray`
            times of the series (bin centers)
        f: `~numpy.ndarray`
            values measures at times t
        e: `~numpy.ndarray`
            uncertainties of values measures at times t
        dt: `~numpy.ndarray` or None, optional
            widths of time bins
        interpolation: str, optional
            interpolation method to fill in gaps in the light curve
            to ensure an even sampling. 
            Options are the same as `scipy.interpolate.interp1d`.
            Default is nearest (recommended for gamma-ray light curves). 
        lctype: str, optional
            either density or average,
            determines how gaps will be filled in interpolation.
            If "density", interpolated value will be used. 
            If "average", average value over sampling length will be use.
            Default is average (recommended for gamma-ray light curves). 
        interp_tstep: float, optional
            determines the sampling of the light curve after interpolation
        """
        self._t = t
        self._f = f
        self._e = e
        self._interpolation = interpolation
        self._lctype = lctype

        if dt is None:
            self._dt = np.concatenate([np.diff(t), [np.diff(t)][-1]])
        else:
            self._dt = dt

        #if type(interp_tstep) == type(None):
            #self._interp_tstep = np.median(self._dt)
        #else:
            #self._interp_tstep = interp_tstep

        #logging.info("Using {0:.2f} as interpolation time step".format(self._interp_tstep))

        #self._ti, self._fi, self._dti = self.__interp_lc(self._t, self._f,
        #    self._interp_tstep,
        #    self._interpolation, self._lctype)

        #logging.info("Number of observed data points / number" \
        #    " of interpolated data points: {0:.3f}".format(float(self._t.size) / float(self._ti.size)))

        # get the interpolated flux PDF
        # bins are in logspace
        # number of bins automatically with chosen from Knuth 2006 algorithm
        #self._pdf, self._pdf_edges = np.histogram(self._fi,
                        #bins = np.logspace(np.log10(self._fi.min()),
                                #np.log10(self._fi.max()),
                                #opt_bins(np.log10(self._fi))))
        self._pdf, self._pdf_edges = np.histogram(self._f,
                        bins = np.logspace(np.log10(self._f.min()),
                                np.log10(self._f.max()),
                                opt_bins(np.log10(self._f))))

        # calculate the variance of the observed light curve
        # needed for rescaling the simulated light curves
        self._var_data = np.var(self._f) 
        self._esq_mean = (self._e**2.).mean()

        # do a linear regression on flux vs error
        #self._sle, self._y0e, r, p, std = linregress(np.log10(self._f), np.log10(self._e))
        # do a lowess regression instead
        l = lowess(np.log10(self._e),np.log10(self._f), return_sorted = True)
        self._interp_unc_lowess = interp1d(10.**l[:,0], 10.**l[:,1], fill_value = 'extrapolate')
        # do an inverse interpolation of the CDF of the errors
        cs = np.cumsum(self._e) 
        self._interp_csinv = interp1d((cs - cs.min()) / (cs.max() - cs.min()), np.sort(self._e))

        # get the mask for the interpolated light curve 
        # that gives the time values of the interpolated light curve 
        # that are in the original light curve
        # note that self._dti's should all be the same
        #tibins = np.concatenate([self._ti - self._dti[0] / 2., [self._ti.max() + self._dti[0] / 2.]])
        #self._m_t_in_ti = np.digitize(self._t, bins = tibins)
        # this mask starts at 1, 0 is underflow
        # since underflow not possible here, subtract 1 to get indices
        #self._m_t_in_ti -= 1

    @property
    def t(self):
        return self._t
    @property
    def dt(self):
        return self._dt
    @property
    def f(self):
        return self._f
    @property
    def e(self):
        return self._e
    #@property
    #def ti(self):
        #return self._ti
    #@property
    #def dti(self):
        #return self._dti
    #@property
    #def fi(self):
        #return self._fi
    #@property
    #def m_t_in_ti(self):
        #return self._m_t_in_ti
    @property
    def lctype(self):
        return self._lctype
    @property
    def interpolation(self):
        return self._interpolation
    #@property
    #def interp_tstep(self):
        #return self._interp_tstep
    @property
    def pdf(self):
        return self._pdf
    @property
    def pdf_edges(self):
        return self._pdf_edges

    @lctype.setter
    def lctype(self, lctype):
        self._lctype = lctype
        #self._ti, self._fi, self._dti = self.__interp_lc(self._t, self._f,
            #self._interp_tstep,
            #self._interpolation, self._lctype)
        return
    @interpolation.setter
    def interpolation(self, interpolation):
        self._interpolation = interpolation
        #self._ti, self._fi, self._dti = self.__interp_lc(self._t, self._f,
            #self._interp_tstep,
            #self._interpolation, self._lctype)
        return
    #@interp_tstep.setter
    #def interp_tstep(self, interp_tstep): 
        #self._interp_tstep = interp_tstep
        #self._ti, self._fi, self._dti = self.__interp_lc(self._t, self._f,
            #self._interp_tstep,
            #self._interpolation, self._lctype)
        #return

#    @staticmethod
#    def __interp_lc(t,f,interp_tstep, interpolation, lctype, 
#                    sim = False):
#        """Interpolate the light curve"""
#        if interpolation == 'none':
#            ti = t
#            fi = f
#
#        else:
#            interp = interp1d(t, f, kind = interpolation,
#                                fill_value = 'extrapolate')
#            #ti = np.arange(t.min(),
#            #        t.max() + interp_tstep if not sim else t.max(),
#            #        interp_tstep)
#            ti = np.linspace(t.min(), t.max(), int(np.floor((t.max() - t.min()) / interp_tstep)) + 1)
#            #if ti.max() > t.max():
#            #    ti = np.arange(t.min(),
#            #        t.max(),
#            #        interp_tstep)
#
#            if lctype == 'density':
#                fi = interp(ti)
##
#            elif lctype == 'average':
#                for ix, xi in enumerate(ti):
#                    if not ix:
#                        xx = np.linspace(xi, xi + interp_tstep / 2., 10)
#                    elif ix == ti.size - 1:
#                        xx = np.vstack([xx, np.linspace( xi - interp_tstep / 2.,
#                                xi, 11, endpoint = True)[1:]])
#                    else:
#                        xx = np.vstack([xx, np.linspace( xi - interp_tstep / 2.,
#                                xi + interp_tstep / 2., 11)[1:]])
#                fi = interp(xx).mean(axis = 1)
#            else:
#                raise ValueError("lctype unknown")
#
#        dti = np.diff(ti)
#        return ti, fi, dti

    @classmethod
    def readformfermifits(cls,fitsfile, tsmin = 4., 
        interpolation = 'linear', lctype = 'density'):
        #interp_tstep = None):
        """
        Read a fermipy created light curve fits file 
        and instantiate the TimeSeries class with it.

        Parameters
        ----------
        fitsfile: str
            path the fitsfile
        tsmin: float, optional
            Only include light curve bins above this TS threshold
        interpolation: str, optional
            interpolation method to fill in gaps in the light curve
            to ensure an even sampling. 
            Options are 'linear', or 'none'
        lctype: str, optional
            either density or average,
            determines how gaps will be filled in interpolation.
            If "density", interpolated value will be used. 
            If "average", average value over sampling length will be used
        interp_tstep: float, optional
            determines the sampling of the light curve after interpolation
        """
        lc = Table.read(fitsfile)
        m = lc['ts'] >= tsmin
        dt = lc['tmax'] - lc['tmin']
        t = 0.5 * (lc['tmax'] + lc['tmin'])
        f = lc['flux']
        e = lc['flux_err']

        frac_ul = float((~m).sum()) / m.size
        logging.info("Total number of LC bins: {0:n}".format(
            t.size))
        logging.info("Median sampling: every {0:n} {1:s}".format(
            np.median(np.diff(t)), lc['tmin'].unit))
        logging.info("Median bin width: {0:n} {1:s}".format(
            np.median(dt),lc['tmin'].unit))
        logging.info("Fraction of bins below TS = {0:.2f}: {1:.3f}%".format(
            tsmin, frac_ul * 100))

        # split the mask array where you have a jump between detection and non detection
        # this results in a list with arrays 
        groups = np.split(m, np.where(np.abs(np.diff(m)) == 1)[0] + 1)
        idx = 0
        maxgap = 0
        for g in groups:
            if np.all(g):
                pass
            else:
                # calculate the length of the gap
                gap = t[idx + g.size] - t[idx]
                if gap > maxgap:
                    maxgap = gap
            idx += g.size

        logging.info("Largest gap between time bins with TS > {0:.2f}: {1:.3f} {2:s}".format(
            tsmin, maxgap, lc['tmax'].unit))

        # perform the interpolation
        return cls(t.data[m],f.data[m],e.data[m],dt = dt.data[m], 
            interpolation=interpolation, lctype = lctype) #, 
            #interp_tstep = interp_tstep)

    @staticmethod
    def periodogram(t, f, window = 'hanning', detrend = 'none', norm = 'var', axis = -1):
        """
        Calculate the periodogram
        of the observed data

        Parameters
        ----------
        t: `~numpy.ndarray`
            times of the series (bin centers)
        f: `~numpy.ndarray`
            values measured at times t
        detrend: str, optional
            detrend the light curve.
            Options are 'none', 'constant', 'linear'
        window: str, optional
            window function applied to light curve
            Options are 'none', 'hanning', 'hann', and 'rect'
        rawlc: bool, optional
            if true, calculate periodogram of raw lc, 
            if false, use interpolated light curve
        norm: str, optional
            normalization of periodogram. Options are:
            'var': integrated periodogram equal to variance, 
                same as Max-Morbeck et al. 2014
            'rms': integrated periodogram equal to rms, 
                same as Emmanoupoulous et al. 2013

        Returns
        -------
        Periodogram and natural frequencies

        """

        t = t - t.min()

        N = t.size
        
        #if N % 2: # N uneven
        #    k = np.arange(1, (N - 1) / 2 + 1,1)
        #else:
        #    k = np.arange(1, N / 2 + 1,1)

        if detrend == 'const' and not norm == 'rms':
            f = f - f.mean()
        elif detrend == 'linear' and not norm == 'rms':
            s,i,r,p,std = linregress(t,f)
            f = f - (s * t + i)

        T = N / float(N - 1) * (t.max() - t.min())
        nu_Nyq = N / 2. / T # nyquist frequency


        # apply the window function and get 2d array
        w = window_function(t.astype(np.float), T, type = window)

        #nu = k / T
        #ff, nn = np.meshgrid(f * w,
        #                        nu, indexing = 'ij')
        #tt = np.meshgrid(t, nu, indexing = 'ij')[0]
        # up to the normalization, the periodogram is equal to the fourier 
        # transform (np.rfft) if no window and no detrending are used. 
        #Pk = (ff * np.cos(2. * np.pi * nn * tt)).sum(axis = 0) ** 2. + \
        #        (ff * np.sin(2. * np.pi * nn * tt)).sum(axis = 0) ** 2.

        # performing the fft is much faster with build in functions:
        # check if there's enough memory, if not split the calculation
        if len(f.shape) == 3 and f.nbytes * 5 > psutil.virtual_memory().free: # check if there's enough memory
            step = 2500
            for ij, isim in enumerate(np.arange(step, f.shape[1] + step, step)):
                Pk_ij = np.fft.rfft((f * w)[:,isim - step:isim,:], axis = axis)
                if not ij:
                    Pk = Pk_ij
                else:
                    Pk = np.hstack([Pk,Pk_ij])
                    logging.debug("Array too large, calculating: {0}".format((ij, Pk.shape)))
        else:
            Pk = np.fft.rfft(f * w, axis = axis)

        Pk = np.real(Pk * np.conjugate(Pk))
        nu = np.fft.rfftfreq(f.shape[axis], d = 1. / nu_Nyq / 2.)

        if norm == 'var':
            Pk /= nu_Nyq * N
        elif norm == 'rms':
            Pk /= nu_Nyq * N * f.mean() ** 2.

        return Pk, nu

    def calc_ti_split(self, t,f, tgap, interp_tstep_func):
        """Get interpolated light curve for light curve piece split at gaps larger than tgap"""
        dt = np.diff(self._t) # gaps between observation points
        idx = np.nonzero(dt > tgap)[0]
        mult = 10 if self._lctype == 'average' else 2
        step = 1000
        if not len(idx):
            return [],[]
        tall, fall = [],[]
        for ii, ix in enumerate(idx): # loop over time gaps 
            if not ii:
                x = t[:ix + 1]
                y = f[...,:ix + 1]
            elif ii == idx.size - 1:
                x = t[idx[ii-1]+1:]
                y = f[...,idx[ii-1]+1:]
            else:
                x = t[idx[ii-1]+1:ix + 1]
                y = f[...,idx[ii-1]+1:ix + 1]
            if x.size > 1: # more than one data point in gap required
                # interpolate on time and flux on linear grid
                if len(y.shape) == 3 and y.nbytes * mult > psutil.virtual_memory().free: # check if there's enough memory
                    for ij, isim in enumerate(np.arange(step, y.shape[1] + step, step)):

                        ti,fij,dti = interp_lc(x,y[:,isim - step:isim,:],
                            interp_tstep_func(np.diff(x)),
                            interpolation = self._interpolation, 
                            lctype = self._lctype, axis = -1) 
                        if not ij:
                            fi = fij
                        else:
                            fi = np.hstack([fi,fij])
                            logging.debug("Array too large, calculating: {0}".format((ij, fi.shape)))
                else:
                    ti,fi,dti = interp_lc(x,y,
                            interp_tstep_func(np.diff(x)),
                            interpolation = self._interpolation, 
                            lctype = self._lctype, axis = -1) 
                if ti.size > 1:
                    logging.info("Segment has {0:n} data points," \
                        " using {1:n} interpolated points".format(x.size, ti.size))
                    tall.append(ti)
                    fall.append(fi)
        return tall, fall

    def periodogram_split(self, t, f, tgap, 
        window = 'hanning', detrend = 'none', norm = 'var', axis = -1, 
        nubins = None, interp_tstep_func = lambda x: np.median(x)):
        """
        Calculate the Periodogram, split the light curves for 
        long observation gaps

        Parameters
        ----------
        t: `~numpy.ndarray`
            times of the series (bin centers)
        f: `~numpy.ndarray`
            values measured at times t
        tgap: float
            maximum value of gaps above which light curve will be split

        nubins: `~numpy.ndarray` or int, optional
            bins in frequency space or number of bins in log space
        detrend: str, optional
            detrend the light curve.
            Options are 'none', 'constant', 'linear'
        window: str, optional
            window function applied to light curve
            Options are 'none', 'hanning', 'hann', and 'rect'
        rawlc: bool, optional
            if true, calculate periodogram of raw lc, 
            if false, use interpolated light curve
        norm: str, optional
            normalization of periodogram. Options are:
            'var': integrated periodogram equal to variance, 
                same as Max-Morbeck et al. 2014
            'rms': integrated periodogram equal to rms, 
                same as Emmanoupoulous et al. 2013
        intperp_tstep_func: function
            Function to determine time binning for interpolated light curve. 
            Will be applied to time intervals of light curve and has to return 
            a float scalar.
            By default, it is the median of the time intervals of the light curve. 


        Returns
        -------
        Frequency bins, log averaged periodogram, log variance (in log10) of periodogram
        """
        dt = np.diff(self._t) # gaps between observation points
        idx = np.nonzero(dt > tgap)[0]

        step = 1000 # step size for large simulated arrays
        mult = 10 if self._lctype == 'average' else 2
        tall, fall = [],[] # lists that will hold the times and fluxes for each segment
        if not len(idx): # no gaps larger than tgap
            logging.info("No gaps found, calculating full periodogram")
            if len(f.shape) == 3 and f.nbytes * mult > psutil.virtual_memory().free: # check if there's enough memory
                for ij, isim in enumerate(np.arange(step, f.shape[1] + step, step)):
                    ti,fij,dti = interp_lc(t,f[:,isim - step:isim,:],
                        interp_tstep_func(np.diff(t)),
                        interpolation = self._interpolation, 
                        lctype = self._lctype, axis = -1) 
                    if not ij:
                        fi = fij
                    else:
                        fi = np.hstack([fi,fij])
                        logging.debug("Array too large, calculating: {0}".format((ij, fi.shape)))
            else:
                ti,fi,dti = interp_lc(t,f,
                            interp_tstep_func(np.diff(t)),
                            interpolation = self._interpolation, 
                            lctype = self._lctype, axis = -1) 
            if ti.size > 1:
                logging.info("Segment has {0:n} data points," \
                    " using {1:n} interpolated points".format(t.size, ti.size))
                tall.append(ti)
                fall.append(fi)
        else:
            logging.info("{0:n} gaps found, calculating periodogram for each segment".format(idx.size))
            tall, fall = self.calc_ti_split(t,f, tgap, interp_tstep_func)
            # and add one step where interpolation is over max 
            # gap size between data points
            if False:
                if len(f.shape) == 3 and f.nbytes > psutil.virtual_memory().free: # check if there's enough memory
                    for ij, isim in enumerate(np.arange(step, f.shape[1] + step, step)):
                        ti,fij,dti = interp_lc(t,f[:,isim - step:isim,:],
                            np.max(np.diff(t)),
                            interpolation = self._interpolation, 
                            lctype = self._lctype, axis = -1) 
                        if not ij:
                            fi = fij
                        else:
                            fi = np.hstack([fi,fij])
                            logging.debug("Array too large, calculating: {0}".format((ij, fi.shape)))
                else:
                    ti,fi,dti = interp_lc(t,f,np.max(np.diff(t)),
                            interpolation = self._interpolation, 
                            lctype = self._lctype, axis = -1) 
                if ti.size > 1:
                    logging.info("Full LC with max(dt) interpolation has {0:n} data points," \
                        " using {1:n} interpolated points".format(t.size, ti.size))
                    tall.append(ti)
                    fall.append(fi)


        pall, nall = [],[]

        logging.info("Calculating Periodogram")
        for i,fi in enumerate(fall):
            pi,ni = self.periodogram(tall[i],fi, window = window, detrend = detrend, norm = norm, axis = -1)
            pall.append(pi)
            nall.append(ni)


        logging.info("Calculating log average")
        t1 = time.time()

        # nu bins in log space
        n = np.concatenate(nall, axis = 0)
        dn = np.diff(n)
        if nubins is None or type(nubins) == int:
            nubins = np.logspace(np.log10(n[1] - dn[0]/2.), 
                np.log10(n[-1] + dn[0]/2.), 31 if nubins is None else nubins)

        # bin the periodogram in logspace
        # in case of data
        if len(pall[0].shape) == 1: # data not sim
            p = np.concatenate(pall, axis = 0)
            r = binned_statistic(n,p, bins = nubins, statistic = logmean)
            plogmean = 10.** r.statistic
            r = binned_statistic(n,p, bins = nubins, statistic = logvar)
            logplogvar = r.statistic

        # in case of simulations
        else:
            if len(pall[0].shape) == 3:
                psim = np.dstack(pall) # sim 
                plogmean = np.zeros((psim.shape[0], psim.shape[1], nubins.size - 1))
                logplogvar = np.zeros((psim.shape[0], psim.shape[1], nubins.size - 1))
                for i, p in enumerate(psim):
                    r = binned_statistic(n,p, bins = nubins, statistic = logmean)
                    plogmean[i] = 10.** r.statistic
                    r = binned_statistic(n,p, bins = nubins, statistic = logvar)
                    logplogvar[i] = r.statistic
            elif len(pall[0].shape) == 2:
                psim = np.vstack(pall) # sim 
                r = binned_statistic(n,psim, bins = nubins, statistic = logmean)
                plogmean = 10.** r.statistic
                r = binned_statistic(n,psim, bins = nubins, statistic = logvar)
                logplogvar = r.statistic

        logging.info("Done, it took {0:.2f} s".format(time.time() - t1))
        return nubins, plogmean, logplogvar

    def sim(self,  model, nextend = 1,
            N = None , dt = None ,
            generate_complex=False,
            random_state=None,
            maxiter = 100,
            use_data_gaps = True,
            rescale = 'variance', **pars):
        """
        Generate a power-law light curve
        This uses the method from Timmer & Koenig [1]_
        Adopted from astroML, 
        https://github.com/astroML/astroML/blob/master/astroML/time_series/generate.py

        Parameters
        ----------
        model: function
            intrinsic spectrum
        N : integer or None, optional
            Number of equal-spaced time steps to generate. 
            If None, it will be calculated from obs. light curve (default). 
        dt : float of None, optional
            Spacing between time-steps
            If None, 0.1 * interp_tstep will be used (default). 
        nextend: int, optional
            extend the time of the light curve by this factor. 
            In post-processing, simulated light curve will be split up
            in nextend light curves. Default is 1.
        generate_complex : boolean (optional)
            if True, generate a complex time series rather than a real time series
        random_state : None, int, or np.random.RandomState instance (optional)
            random seed or random number generator
        rescale: str, optional
            Determine how the light curve will be rescaled. 
            Options are 'none', 'variance', 'variance-post', 'EM13'. Default is 'variance'
        use_data_gaps: bool, optional
            if true (default), apply same gaps and interpolation to simulation 
            as you did on the data
        maxiter: int, optional
            Determines the maximum iterations allowed for EM13 
            algorithm. Default: 100. Will be multiplied with nextend.
        pars: dict
            Additional parameters for the model function


        Returns
        -------
        x : ndarray
            the simulated light curves

        References
        ----------
        .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
        """
        random_state = check_random_state(random_state)

        tstep = np.median(np.diff(self._t))
        if dt is None:
            dt = 0.1 * tstep
        dt = float(dt)

        if N == None:
            N = np.ceil((self._t.max() - self._t.min() + \
                        tstep) / dt)

        N = int(N * nextend)

        logging.info("time step for simulation: {0:.2f}" \
            " over total range of {1:.2f}; data range: {2:.2f}".format(
            dt, N * dt, self.t.max() - self.t.min()))

        Npos = int(N / 2)
        Nneg = int((N - 1) / 2)
        domega = (2 * np.pi / dt / N)

        if generate_complex:
            omega = domega * np.fft.ifftshift(np.arange(N) - int(N / 2))
        else:
            omega = domega * np.arange(Npos + 1)

        x_fft = np.zeros(len(omega), dtype=complex)
        x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
        x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

        # rescale the Fourier components with the model
        #x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
        #x_fft[1:] *= (1. / np.sqrt(2))
        x_fft[1:] *= np.sqrt(0.5 * model(omega[1:], **pars))

        # by symmetry, the Nyquist frequency is real if x is real
        if (not generate_complex) and (N % 2 == 0):
            x_fft.imag[-1] = 0

        if generate_complex:
            x = np.fft.ifft(x_fft)
        else:
            x = np.fft.irfft(x_fft, N)

        # rescale the variance
        #if rescale == 'variance':
            #x = var_normalization(x, axis = -1) * np.sqrt(self._var_data - self._esq_mean)

        # ==== now do some post processing ==== #
        if nextend > 1:
            x = x.reshape(nextend, N / nextend)
        else:
            x = x[np.newaxis,:]

        # rescale the variance
        if rescale == 'variance':
            x = var_normalization(x, axis = -1) * np.sqrt(self._var_data - self._esq_mean)

        # reshape the array either by taking 
        # the average or simply taking the data points 
        # so that resulting sampling is equal to 
        # sampling of interploated light curve
        step = int(round(tstep / dt, 3))
        if self._lctype == 'average':
            x = x.reshape(x.shape[0],
                step, # axis to average over
                x.shape[1] / step).mean(axis = 1)

        elif self._lctype == 'density':
            x = x[:,::step]
            if x.shape[1] > N / nextend / step:
                logging.warning("Shape mismatch: x.shape = {0}, N / nextend: {1}" \
                ", using N / nextend shape".format(x.shape, N / nextend / step))
                x = x[:,:N / nextend / step]

        # make sure that simulated light curve has same 
        # length as interpolated observed lc
        assert x.shape[1] == N / nextend / step

        # rescale the variance
        if rescale == 'variance-post':
            x = var_normalization(x, axis = -1) * np.sqrt(self._var_data - self._esq_mean)
           
        if use_data_gaps:
            # now pretend that the simulated light curve has the same gaps 
            # as the observed light curve and perform the exact same interpolation
            # as you did with the data 

            # get the mask for the interpolated light curve 
            # that gives the time values of the interpolated light curve 
            # that are in the original light curve
            # note that self._dti's should all be the same
            ti = np.arange(self._t.min(), self._t.max() + tstep, tstep )
            if ti.size > x.shape[-1]:
                ti = ti[:x.shape[-1]]
            dti = np.diff(ti)

            logging.debug("ti shape: {0}".format(ti.shape))
            logging.debug("x shape: {0}".format(x.shape))
            logging.debug("t shape: {0}".format(self._t.shape))
            
            tibins = np.concatenate([ti - dti[0] / 2., [ti.max() + dti[0] / 2.]])
            m_t_in_ti = np.digitize(self._t, bins = tibins)
            # this mask starts at 1, 0 is underflow
            # since underflow not possible here, subtract 1 to get indices
            m_t_in_ti -= 1
            # suppress overflow
            m_t_in_ti[ m_t_in_ti >= x.shape[-1]] = \
                np.ones( np.sum(m_t_in_ti >= x.shape[-1]) ) * int(x.shape[-1] - 1)

            # an array for indexing
            idx0arr = np.array([np.ones(x.shape[-1], dtype = np.int) * i \
                        for i in range(self._t.shape[0])])
            ii, mm = np.meshgrid(range(x.shape[0]), m_t_in_ti, indexing = 'ij')

            logging.debug("ii, mm shapes: {0}".format((ii.shape, mm.shape)))
            logging.debug("m_t_in_ti: {0}".format(m_t_in_ti))
            logging.debug("m_t_in_ti shape: {0}".format(m_t_in_ti.shape))

            # this adds the gaps
            xsim = x[ii, mm]
            logging.debug("xsim shape: {0}".format(xsim.shape))
            assert xsim.shape[-1] == self._t.size
            #xdata = x[ii, mm]
            #xsim = np.zeros([nextend, self._ti.size])
            # now we interpolate as we would for the data
            #for i,xi in enumerate(xdata):
            #    ts, xs, dts = self.__interp_lc(self._t, xi,
            #                    self._interp_tstep,
            #                    self._interpolation, self._lctype)
            #    xsim[i] = xs
        else:
            xsim = x


        # use PDF and PSD matching from Emmanoupoulous et al. 2013
        if rescale == 'EM13': 
            # Calculate the FFT for the TK generated LC:
            # this is the "norm" light curve and DFT in EM2013
            xnorm_fft = np.fft.rfft(xsim, axis = 1)
            #xnorm_fft = np.fft.fft(xsim, axis = 1)

            # Draw a random sample from the observed (interpolated)
            # flux distribution, this is the "sim,1" or "sim,k" light curve
            # and DFT in EM2013
            # adopted from https://github.com/samconnolly/DELightcurveSimulation/blob/master/DELCgen.py
            chances = self._pdf / float(self._pdf.sum())
            sample = np.random.choice(chances.size,
                        xsim.shape,
                        p=chances, replace = True)
            # draw from step-wise PDF
            xsim0 = np.random.uniform(self._pdf_edges[:-1][sample],
                                        self._pdf_edges[1:][sample])

            k = 0
            xsimk = np.array([1])
            xsimk_m1 = np.array([-1]) # k-1 step 

            # an array for indexing, needed later
            idx0arr = np.array([np.ones(xsim0.shape[1], dtype = np.int) * i \
                                    for i in range(xsim0.shape[0])])
            while k < maxiter * nextend and np.array_equal(xsimk,xsimk_m1) == False:

                xsimk_m1 = xsimk

                if not k:
                    xsimk = xsim0
                else:
                    xsimk = xsimk_adj
                # calculate its FFT
                xsimk_fft = np.fft.rfft(xsimk, axis = 1)
                #xsimk_fft = np.fft.fft(xsimk, axis = 1)
                # replace the coefficients
                xsimk_adj_fft = np.absolute(xnorm_fft) * np.exp(1j * np.angle(xsimk_fft))
                # inverse Fourier transfom
                xsimk_adj = np.fft.irfft(xsimk_adj_fft, axis = 1, n = xsim.shape[1])
                #xsimk_adj = np.fft.ifft(xsimk_adj_fft, axis = 1)
                # amplitude adjustment
                idxsort_simk_adj = np.argsort(xsimk_adj, axis = 1)
                idxsort_simk = np.argsort(xsimk, axis = 1)
                xsimk_adj[idx0arr,idxsort_simk_adj] = xsimk[idx0arr,idxsort_simk]

                k += 1
            xsim = xsimk
            if k >= maxiter * nextend:
                logging.warning("EM13 loop did not converge!")

        return xsim


    def add_sim_unc(self,xsim,unctype='data'):
        """
        Generate uncertainties and them to simulated light curves

        Parameters
        ----------
        xsim: `~numpy.ndarray`
            simulated light curves

        unctype: str
            Method to add uncertainties to light curve.
            Options are:
            'data' = randomly draw uncertainties from data and add with gaussian (default); 
            'datasort' = randomly draw uncertainties from data and add with gaussian,
                use largest uncertainties for largest flux values; 
            'linear' = get the uncerainties from linear interpolation of flux 
            'linear-gauss' = get the uncerainties from linear interpolation of flux and draw from 
            Gaussian centered on flux value
            vs uncertainty from data and add as gaussian term

        Returns
        -------
        simulated light curves with uncertainties as array with same shape as input
        """
        if xsim.shape[-1] == self._t.size:
            replace = False
        else:
            replace == True
        logging.info("Using replacement: {0}".format(replace))

        if unctype == 'data' or unctype == 'datafraction':
            if 'fraction' in unctype:
                if replace:
                    sigma_sim = (self._e / self._f)[np.random.choice(self._e.size,size = xsim.shape, replace = replace)] 
                else: 
                    sigma_sim  = np.array([np.random.permutation(self._e / self._f) \
                                    for i in range(np.product(xsim.shape[:-1]))]).reshape(xsim.shape)
                sigma_sim = np.abs(sigma_sim * xsim)
            else:
                if replace:
                    sigma_sim = self._e[np.random.choice(self._e.size,size = xsim.shape, replace = replace)] 
                else: 
                    sigma_sim  = np.array([np.random.permutation(self._e) \
                        for i in range(np.product(xsim.shape[:-1]))]).reshape(xsim.shape)
            esim = norm.rvs(loc = np.zeros(sigma_sim.shape), 
                            scale = sigma_sim)

            xsim += esim


        elif unctype == 'datasort' or unctype == 'datasortfraction':
            if 'fraction' in unctype:
                sigma_sim = (self._e / self._f)[np.random.choice(self._e.size,size = xsim.shape, replace = replace)] 
                sigma_sim = np.abs(sigma_sim * xsim)
            else:
                sigma_sim = self._e[np.random.choice(self._e.size,size = xsim.shape, replace = replace)] 
            # an array for indexing
            idx0arr = np.array([np.ones(sigma_sim.shape[1], dtype = np.int) * i \
                                for i in range(sigma_sim.shape[0])])
            idxsort = np.argsort(xsim, axis = 1)
            idssort = np.argsort(sigma_sim, axis = 1)
            # points with largest flux get largest error
            sigma_sim[idx0arr,idxsort] = sigma_sim[idx0arr,idssort] 

            esim = norm.rvs(loc = np.zeros(sigma_sim.shape), 
                        scale = sigma_sim)

            xsim += esim

        elif unctype == 'linear':
            # shift to zero
            offset = xsim.min(axis = 1)
            offset[offset > 0] = np.zeros(np.sum(offset > 0))
            xsim = (xsim.T - offset + self._f.min()).T
            sigma_sim = self._interp_unc_lowess(xsim)

            esim = norm.rvs(loc = np.zeros(sigma_sim.shape), 
                        scale = sigma_sim)

            xsim += esim
        elif unctype == 'linear-gauss':
            #sigma_sim = xsim ** self._sle * 10.**self._y0e
            sigma_sim = self._interp_unc_lowess(xsim)
            #xsim = norm.rvs(loc = xsim, 
                        #scale = sigma_sim)
            # use truncated gaussian so that there are no entries < 0
            xsim = truncnorm.rvs(a = (0. - xsim) / sigma_sim, b = np.inf,
                        loc = xsim, 
                        scale = sigma_sim)
        else:
            raise ValueError("Unknown type of unctype chosen")

        return xsim

def lccf(ta,a,tb,b, bins = None):
    """
    Calculate the local cross correlation functin (LCCF)
    for two time series

    Parameters
    ----------
    ta: `~numpy.ndarray`
        n-dim array with times of time series a
    a: `~numpy.ndarray`
        n-dim array with values at times ta of time series a
    tb: `~numpy.ndarray`
        m-dim array with times of time series b
    b: `~numpy.ndarray`
        m-dim array with values at times ta of time series b
    bins: `~numpy.ndarray`
        k-dim array with bin edges for time lags, optional

    Returns
    -------
    tuple with k-dim bin edges for time lags and k-1 dim array 
    with lccf values

    Notes 
    -----
    See e.g. Max-Moerbeck et al. 2014, arXiv:1408.6265

    See also https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/35608701
    """

    if type(bins) == type(None):
        taumax = np.min([ta.max() - ta.min(),
                        tb.max() - tb.min()]) / 2.
        bins = np.linspace(-taumax, taumax, 200)
    else:
        taumax = np.max(np.abs(bins))

    if not np.all(np.equal(ta.shape, a.shape)):
        raise TypeError("ta and a must be of the same shape")
    if not np.all(np.equal(tb.shape, b.shape)):
        raise TypeError("tb and b must be of the same shape")

    # array with indices for time series a
    i = np.arange(a.size)
    # array with indices for time series b
    j = np.arange(b.size)
    # (i*j, 2) array with all combinations of i and j
    # see the stackoverflow link above
    logging.debug("{0} {1}".format(i.shape, j.shape))
    ij = np.array(np.meshgrid(i,j)).T.reshape(-1,2)

    # time series a leading b
    tau = ta[ij[:,0]] - tb[ij[:,1]]
    m = np.abs(tau) <= taumax
    ab = np.array([a[ij[m,0]], b[ij[m,1]]])

    # build a 2d mask with dim (bins, ij), 
    # which contains for each bin the tau values 
    # that fall into that bin
    binnum = np.digitize(tau[m], bins = bins) - 1 # bin number for each tau value

    #bbnn,nn = np.meshgrid(np.arange(bins.size - 1), binnum, indexing = 'ij')
    #aa = np.meshgrid(np.arange(bins.size - 1), a[ij[m,0]], indexing = 'ij')[1]
    #bb = np.meshgrid(np.arange(bins.size - 1), b[ij[m,0]], indexing = 'ij')[1]
    #mask = bbnn == nn

    # build masked arrays
    #am = np.ma.array(aa, mask = ~mask)
    #bm = np.ma.array(bb, mask = ~mask)

    # calculate the lccf
    #lccf = ((am.T - am.mean(axis = -1)) * (bm.T - bm.mean(axis = -1)) / \
    #            np.sqrt(am.var(axis = -1) * bm.var(axis = -1)) / mask.sum(axis = -1)).sum(axis = 0)

    #return lccf.data[~lccf.mask], bins
    result = np.zeros(bins.size - 1)
    for i,b in enumerate(bins[:-1]):
        mask = binnum == i 
        if mask.sum() >= 2 and ab[0][mask].var() > 0. and ab[1][mask].var() > 0.:
            result[i] = ((ab[0][mask] - ab[0][mask].mean()) * (ab[1][mask] - ab[1][mask].mean()) / \
                        np.sqrt(ab[0][mask].var() * ab[1][mask].var())).sum() / mask.sum()
    return result, bins

def lccf_unc(ta,a,ea,tb,b,eb, bins = None, npools = 4, nsim = 100):
    """
    Generage lccf from random sample selection 
    and flux randomization following 
    Peterson et al. (1998), see alse Max-Moerbeck et al. (2014) and Kiehlmann (2015)

    Parameters
    ----------
    ta: `~numpy.ndarray`
        n-dim array with times of time series a
    a: `~numpy.ndarray`
        n-dim array with values at times ta of time series a
    ea: `~numpy.ndarray`
        n-dim array with uncertainties at times ta of time series a
    tb: `~numpy.ndarray`
        m-dim array with times of time series b
    b: `~numpy.ndarray`
        m-dim array with values at times ta of time series b
    eb: `~numpy.ndarray`
        m-dim array with uncertainties at times tb of time series b
    bins: `~numpy.ndarray`
        k-dim array with bin edges for time lags, optional
    npools: int
        number of pools for multiprocessing, optional (default = 4)
    nsim: int
        number of bootstrapped light curves

    Returns
    -------
    `~numpy.ndarray` of size nsim with peak values of time lag
    """
    # draw indices from flat distribution, i.e., with replacement
    xa = np.random.randint(a.size, size = (nsim, a.size))
    xb = np.random.randint(b.size, size = (nsim, b.size))

    # draw random numbers from Gaussian distribution 
    # to add them the to flux
    easim = norm.rvs(loc = np.zeros(xa.shape), 
                            scale = ea[xa])
    ebsim = norm.rvs(loc = np.zeros(xb.shape), 
                            scale = eb[xb])
    # remove duplicates
    xa = [np.unique(x) for x in xa]
    xb = [np.unique(x) for x in xb]
    # generate fluxes and times
    fa = [a[xa[i]] + easim[i,np.unique(xa[i], return_index = True)[1]] for i in range(nsim)]
    fb = [b[xb[i]] + ebsim[i,np.unique(xb[i], return_index = True)[1]] for i in range(nsim)]
    tasim = [ta[xa[i]] for i in range(nsim)]
    tbsim = [tb[xb[i]] for i in range(nsim)]

    # run the LCCF for each LC
    def run_lccfunc(i):
        global tasim, tbsim, fa, fb, bins
        lccf_sim, bsim = lccf(tasim[i], fa[i],
                            tbsim[i], fb[i],
                            bins = bins)
        return lccf_sim
    logging.info("Running LCCF on {0:n} random subsamples".format(nsim))
    t1 = time.time()
    pool_size = np.min([npools, cpu_count])
    jobs = list(range(nsim))

    tau_peak = []
    for i in jobs:
        l, b = lccf(tasim[i], fa[i],
                            tbsim[i], fb[i],
                            bins = bins)
        cen = 0.5 * (b[1:] + b[:-1])
        tau_peak.append(cen[np.argmax(l)])
    #pool = Pool(processes=pool_size,
    #        maxtasksperchild=None)
    #lccf_sim = pool.map(run_lccfunc, jobs)
    #pool.close() # no more tasks
    #pool.join()  # wrap up current tasks
    tau_peak = np.array(tau_peak)
    logging.info("Finished, it took {0:.2f} s".format(time.time() - t1))
    return tau_peak 
