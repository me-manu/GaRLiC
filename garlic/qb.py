"""
Function to estimate the quiescent level of a light curve.
"""

import numpy as np 
from scipy.special import gammaln

def pks(z):
    """Compute KS distribution, see numerical recipes Eq. 6.14.56/57"""
    if z < 0.:
        raise ValueError("bad z in KSdist")
    if z == 0.:
        return 0.;
    if z < 1.18:
        y = np.exp(-1.23370055013616983/(z * z))
        return 2.25675833419102515*np.sqrt(-np.log(y)) \
                    *(y + np.power(y,9) + np.power(y,25) + np.power(y,49))
    else:
        x = np.exp(-2.*z*z)
        return 1. - 2.*(x - np.power(x,4) + np.power(x,9));

def qks(z):
    """Compute complemement of KS distribution, see numerical recipes Eq. 6.14.56/57"""
    if z < 0.:
        raise ValueError("bad z in KSdist")
    if z == 0.:
        return 1.
    if z < 1.18: 
        return 1.-pks(z);
    x = np.exp(-2.*z*z)
    return 2.*(x - np.power(x,4) + np.power(x,9))

def wolpert(h1, h2, axis = 0):
    """
    Calculate the ratio of posteriors using the algorithm of Wolpert (1995), 
    see his Eq. (11)

    Parameters
    ----------
    h1: array-like
        first histogram of data 
    h2: array-like
        second histogram of data 
    axis: int or None (optional) 
        Used when stack of histograms is provided as 2d array.
        In this case, axis specifies the axis of the histograms
        (vs the axis of the different cases). Default: 0

    Returns 
    -------
    log10 of posterior ratio of Pr(same) / Pr(different)
    """
    if not np.all(np.equal(h1.shape, h2.shape)):
        raise ValueError("Histrogams must have same shape!")
    n1 = h1.sum(axis = axis)
    n2 = h2.sum(axis = axis)
    num_bins = h1.shape[axis]
    # log of eq (10) divided by first factor of eq. (9) 
    # [gammaln is ln of the gamma function;
    #  remember n! = gamma( n + 1) ]
    log_common = gammaln(n1 + num_bins) + gammaln(n2 + num_bins) -\
        gammaln(num_bins) - gammaln(n1 + n2 + num_bins) 
    # log of second factor in eq. (9)
    log_hist = gammaln(h1 + h2 + 1) - gammaln(h1 + 1) - gammaln(h2 + 1) 
    # Combine and convert to base-10 log
    log_odds_ratio = np.log10(np.exp(1.)) * (log_common + np.sum(log_hist, axis = axis))
    return log_odds_ratio


def find_quiet(flux_vec, weight_vec = None,
    stat = 'ks', num_grid = 0, num_bins = 200, fudge_factor = 0.1):
    """
    Estimate the background level in flaring time series.
    Assumption: the low-end tail of the flux distribution is dominated
    by the fluxes defining the quiet background level.
    The term "distribution" here means the distribution of the fluxes
    in the range assumed for the "low-end tail" of the distributon
    of all the fluxes.

    Parameters
    ----------
    flux_vec: array-like
        array of flux values
    weight_vec: array-like (optional)
        array with weights for fluxes (same shape as flux_vec)
    stat: str (optional)
        statistic used to optimize test. Options are
        'ks' : komlogorov smirnov test, 
        'sumsq': sum of squared differences (Jeff's matlab code)
        'sumsqpy': a python version of Jeff's matlab code
    num_grid: int (optional)
        size of flux array tested for the peak of the symmetric 
        quiescent flux. If zero, use the measured flux values
    num_bins: int (optional)
        number of bins for high and low flux arrays
    fudge_factor: float (optional)
        factor to avoid some trouble just above the flux minimum. 
        Only used when num_grid > 0. 
    """
    if weight_vec is None: 
        weight_vec = np.ones_like(flux_vec)

    id_sort = np.argsort(flux_vec)
    flux_vec = np.sort(flux_vec)
    weight_vec = weight_vec[id_sort]
    flux_mean = flux_vec.mean() # or median

    x = np.linspace(0.,1.,num_bins)

    flux_min_use = flux_vec[0] + fudge_factor * ( flux_mean - flux_vec[0] )

    if num_grid:
        # Certainly QB is between the minimum and the mean flux!
        flux_grid = np.linspace(flux_min_use, flux_mean, num_grid)
        #flux_grid = np.logspace(np.log10(flux_min_use), np.log10(flux_mean), num_grid)
    else:
        # Certainly QB is between the minimum and the mean flux!
        flux_grid = flux_vec[np.where(flux_vec >= flux_min_use)[0][0]:np.where(flux_vec <= flux_mean)[0][-1]]

    flux_asym = np.ones_like(flux_grid) * np.nan

    #for ii_peak in range(buff, len(flux_vec) - buff):
    for ii_peak in range(flux_grid.size):
        #flux_peak = flux_vec[ii_peak]
        flux_peak = flux_grid[ii_peak]
        dt_low = flux_peak - flux_vec[0]
        flux_high = flux_peak + dt_low # equal intervals on each side of peak

        flux_lo_vec = np.linspace(flux_vec[0], flux_peak, num_bins)
        flux_hi_vec = np.linspace(flux_peak, flux_high, num_bins)

        # compute CDF
        if stat == 'sumsq' or stat == 'wolpert_jeff':
            cdf_hi = np.zeros(flux_hi_vec.size - 1)
            cdf_lo = np.zeros(flux_lo_vec.size - 1)

            for ii_bin in range(num_bins - 1):
                id_lo = flux_vec <= flux_lo_vec[ii_bin]
                id_hi = (flux_vec >= flux_hi_vec[ii_bin]) & \
                        (flux_vec <= flux_high)
                cdf_hi[ii_bin] = np.sum(weight_vec[id_hi])
                cdf_lo[ii_bin] = np.sum(weight_vec[id_lo])

            cdf_hi = cdf_hi[::-1] # reverse the high cdf

            if stat == 'sumsq':
                cdf_hi /= cdf_hi[-1]
                cdf_lo /= cdf_lo[-1]

                flux_asym[ii_peak] = np.sum((cdf_hi - cdf_lo) ** 2.)
            else:
                flux_asym[ii_peak] = -1. * wolpert(cdf_lo / cdf_lo.sum(), cdf_hi / cdf_hi.sum()) 

        else:
            idlo = flux_vec < flux_peak
            idhi= (flux_vec >= flux_peak) & (flux_vec <= flux_high)

            cdf_lo = np.concatenate([[0.],np.cumsum(weight_vec[idlo])])
            cdf_hi = np.concatenate([[0.],np.cumsum(weight_vec[idhi][::-1])])

            cdf_lo /= cdf_lo[-1]
            cdf_hi /= cdf_hi[-1]

            # convert fluxes below and above peak to array between 0 and 1
            xlo = np.concatenate([[0.],(flux_vec[idlo] - flux_vec[idlo][0])/ dt_low])
            xhi = (1. - (flux_vec[idhi] - flux_vec[idhi][0])/ dt_low)[::-1]
            xhi = np.concatenate([[0.],xhi])

            # get the bin number where of fine sampled x array 
            bins_hi = np.digitize(x = x, bins = xhi) - 1
            bins_lo = np.digitize(x = x, bins = xlo) - 1
            if stat == 'sumsqpy':
                flux_asym[ii_peak] = np.sum((cdf_hi[bins_hi] - cdf_lo[bins_lo])**2.)

            elif stat == 'ks':
                d = np.max(np.abs(cdf_hi[bins_hi] - cdf_lo[bins_lo]))
                ne = idhi.sum()* idlo.sum() / (idhi.sum() + idlo.sum())
                flux_asym[ii_peak] = qks((np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne))*d)

            elif stat == 'wolpert':
                # use wolpert algorithm with CDF normalized to integral
                chi = cdf_hi[bins_hi] * cdf_hi[-1] / cdf_hi[bins_hi].sum()
                clo = cdf_lo[bins_lo]* cdf_lo[-1] / cdf_lo[bins_lo].sum()
                flux_asym[ii_peak] = -1. * wolpert(clo,chi)

    if stat == 'ks':
        id_best = np.argmax(flux_asym[np.isfinite(flux_asym)])
    else:
        id_best = np.argmin(flux_asym[np.isfinite(flux_asym)])
    flux_peak = flux_grid[np.isfinite(flux_asym)][id_best]
    flux_high = flux_peak + dt_low
    flux_quiet = flux_vec[flux_vec <= flux_high]
    return flux_asym, flux_quiet, flux_peak, flux_vec, flux_grid
    

