from pytao import Tao
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import re
import io
from os import path,environ
import pandas as pd

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.statistics import resample_particles


#New imports for plotMod()
import pmd_beamphysics
from matplotlib.gridspec import GridSpec
from copy import copy


def plotMod(particle_group, key1='t', key2='p', 
                  bins=None,
                  *,
                  xlim=None,
                  ylim=None,
                  tex=True,
                  nice=True,
                  **kwargs):
    """
    Derived from openPMD-beamphysics marginal_plot()
    """    

    plt.close('all')
    
    CMAP0 = copy(plt.get_cmap('viridis'))
    CMAP0.set_under(CMAP0(0))  # set under-color to the lowest colormap color
    CMAP1 = copy(plt.get_cmap('plasma'))

    plt.ioff()
    
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    # Scale to nice units and get the factor, unit prefix
    x = particle_group[key1]
    y = particle_group[key2]
    
    # Form nice arrays
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(y, nice=nice, lim=ylim)
    
    w = particle_group['weight']
    
    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2
    
    labelx = pmd_beamphysics.labels.mathlabel(key1, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(key2, units=uy, tex=tex)

    fig = plt.figure(**kwargs)
    gs = GridSpec(4,4)
    
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])

    # Set the joint plot background color to match the bottom end of the colormap
    ax_joint.set_facecolor(CMAP0(0))
    
    # Plot the hexbin
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)
    
    # Top histogram
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    if u1 == 's':
        _, hist_prefix = pmd_beamphysics.units.nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:   
        ax_marg_x.set_ylabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{ux}'))

    # Side histogram
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{uy}'))
    
    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)
    
    if xlim:
        ax_joint.set_xlim(xmin/f1, xmax/f1)      
        ax_marg_x.set_xlim(xmin/f1, xmax/f1)
        
    if ylim:
        ax_joint.set_ylim(ymin/f2, ymax/f2)     
        ax_marg_y.set_ylim(ymin/f2, ymax/f2)
    
    return fig
 



def slicePlotMod(particle_group, 
               *keys,
               n_slice=40,
               slice_key=None,
               xlim=None,
               ylim=None,
               tex=True,
               nice=True,
               **kwargs):
    """
    Derived from openPMD-beamphysics slice_plot()
    """    

    #NMM new
    plt.close('all')
    
    #NMM new!
    plt.ioff()

    
    # Allow a single key
    #if isinstance(keys, str):
    # 
    #     keys = (keys, )
    
    if slice_key is None:
        if particle_group.in_t_coordinates:
            slice_key = 'z'
        else:
            slice_key = 't'  
            
    # Special case for delta_
    if slice_key.startswith('delta_'):
        slice_key = slice_key[6:]
        has_delta_prefix = True
    else:
        has_delta_prefix = False
    
    # Get all data
    x_key = 'mean_'+slice_key
    slice_dat = particle_group.slice_statistics(*keys, n_slice=n_slice, slice_key=slice_key)
    slice_dat['density'] = slice_dat['charge']/ slice_dat['ptp_'+slice_key]
    y2_key = 'density'

    # X-axis
    x = slice_dat['mean_'+slice_key]  
    if has_delta_prefix:
        x -= particle_group['mean_'+slice_key]
        slice_key = 'delta_'+slice_key # restore        
        
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    ux = p1+str(particle_group.units(slice_key))
    
    # Y-axis
    
    # Units check
    ulist = [particle_group.units(k).unitSymbol for k in keys]
    uy = ulist[0]
    if not all([u==uy for u in ulist] ):
        raise ValueError(f'Incompatible units: {ulist}')
    
    ymin = max([slice_dat[k].min() for k in keys])
    ymax = max([slice_dat[k].max() for k in keys])
    
    _, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(np.array([ymin, ymax]), nice=nice, lim=ylim)
    uy = p2 + uy
        
    # Form Figure
    fig, ax = plt.subplots(**kwargs)
    
    # Main curves  
    if len(keys) == 1:
        color = 'black'
    else:
        color = None
    
    for k in keys:
        label = pmd_beamphysics.labels.mathlabel(k, units=uy, tex=tex)
        ax.plot(x, slice_dat[k]/f2, label=label, color=color)
    if len(keys) > 1:
        ax.legend()      

    # Density on r.h.s
    y2, _, prey2, _, _ = pmd_beamphysics.units.plottable_array(slice_dat[y2_key], nice=nice, lim=None)
    
    # Convert to Amps if possible
    y2_units = f'C/{particle_group.units(x_key)}'
    if y2_units == 'C/s':
        y2_units = 'A'
    y2_units = prey2+y2_units 
    
    # Labels
    labelx = pmd_beamphysics.labels.mathlabel(slice_key, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(*keys, units=uy, tex=tex)    
    labely2 = pmd_beamphysics.labels.mathlabel(y2_key, units=y2_units, tex=tex)        
    
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)


    # rhs plot
    ax2 = ax.twinx()
    ax2.set_ylabel(labely2)
    ax2.fill_between(x, 0, y2, color='black', alpha = 0.2)  
    ax2.set_ylim(0, None)
    
    # Actual plot limits, considering scaling
    if xlim:
        ax.set_xlim( xmin/f1, xmax/f1) 
    if ylim:
        ax.set_ylim( ymin/f2, ymax/f2)              

    return fig