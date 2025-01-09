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



########
#The following floorplan plots and support functions are adapted from functions from D. Cesar
########

colorlist=['#982649', '#6BCAE2', '#72bda3', '#ed6a5a', '#533a71']
colorlist2=['#E2836A','#6a7ee2','#74e26a']

def floorplan_sorter(ele):
    """
    ele is an element of a pandas dictionary made from the bmad floorplan (made to look like an elegant floorplan from my "elegant_helpers" file). This sorter parses the columns to make a "patch" for plotting purposes. See floorplan_patches().
    """
    if ele['ds']==0:
        ele['ds']=0.05
    s0=float(ele['s'])-float(ele['ds'])
    x=float(ele['X'])*2-0.5
    
    if (re.search('DRIF',ele.ElementType.upper())!=None)|(ele.ElementType.upper()=='MARK'):
        patchColor=None
        patch=None
    elif (re.search('BEND',ele.ElementType.upper())!=None)|(re.search('KICK',ele.ElementType.upper())!=None):
        patchColor='r'
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif re.search('XL',ele.ElementName.upper())!=None:
        patchColor=colorlist2[1]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'QUAD' in ele.ElementType.upper():
        patchColor=colorlist[2]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'SEXT' in ele.ElementType.upper():
        patchColor="#204337"
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'RFCW' in ele.ElementType.upper() or 'CAV' in ele.ElementType.upper():
        string_lst=['L3_10_50','L3_10_25','L2_10_25','L2_10_50','X1_Xband','L1_10_25','L1_9_25','L1_9_50'] #nonzero voltage
        if re.findall(r"(?=("+'|'.join(string_lst)+r"))",ele.ElementName)!=None:
            patchColor="#CD7F32"
            patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=0.5,color=patchColor,lw=0,ls=None)
    elif re.search('^UM',ele.ElementName)!=None:
        patchColor=colorlist[1]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    else:
        patchColor=None
        patch=None
        
    return patch

def floorplan_patches(floorplan,zbounds=None):
    """
    This function returns a list of patches to be plotted (patches) and a list of patches for the legend (leg_patches). If zbounds=[zmin,zmax] is given then the plot is restricted to the bounds. 
    
    Useage:
    
    fp=SDDS(0)
    fp.load(LCLS2scS.flr)
    __,floorplan=sdds2pd(fp)
    patches,leg_patches=flooplan_patches(floorplan,[3425,3750])
    """
    if zbounds==None:
        zbounds=[flooplan['s'].min(),flooplan['s'].max()]
    sFloor=floorplan.s.astype(dtype=float);
    sFloor=sFloor.values
    ii=0;
    patches=[];
    for index, ele in (floorplan.iloc[(sFloor>zbounds[0])&(sFloor<zbounds[1])]).iterrows():
        patches.append(floorplan_sorter(ele))
    
    quad_patch = mpatches.Patch(color=colorlist[2], label='Quad')
    sext_patch = mpatches.Patch(color="#204337", label='Linac')
    bend_patch = mpatches.Patch(color='red', label='Bend')
    leg_patches=[quad_patch,sext_patch,bend_patch];
    return patches,leg_patches


def floorplan_plot_partial(ax_fp,floorplan,zmin=0,zmax=2000):  
    """
    This function plots "patches" for basic elements in the lattice. This can help identify what you're looking at in a "z-plot".
    """
    
    patches,leg_patches=floorplan_patches(floorplan,[zmin,zmax])

    for p in patches:
        if p!=None:
            ax_fp.add_patch(p)

    ax_fp.plot((zmin,zmax),(0,0),'k',alpha=0.0)
    ax_fp.tick_params(axis='x',direction='out',length=15,width=6,color='k',bottom=True)
    plt.yticks([])
    #ax_fp.set_ylim([-3,1])
    ax_fp.set_xlim([zmin,zmax])
    return ax_fp
    
def format_longitudinal_plot(fig, floorplan):
    """
    This function helps format a "z-plot" by providing axes for the main plot and for the a floorplan_plot_partial. It also plots the floorplan.
    """
    outer_grid=fig.add_gridspec(5,1,hspace=0)
    ax=fig.add_subplot(outer_grid[0:4,:])
    ax_fp=fig.add_subplot(outer_grid[4,:], sharex = ax)
    floorplan_plot_partial(ax_fp, floorplan)
    plt.sca(ax)
    
    return ax, ax_fp 

def floorplanPlot(
    tao,
    zmin = 13,
    zmax = 1020,
    ymin = 0.1,
    ymax = 250
):
    elements=tao.lat_ele_list();

    
    floorplan=pd.read_csv(
        io.StringIO('\n'.join(tao.show('lat -all -floor_coords -tracking_elements')[3:-5])), 
        sep="[\s\n]+",
        engine='python',
        names=['Index','ElementName','ElementType','s','ds','X','Y','Z','Theta','Phi','Psi'])
    floorplan.drop(0,inplace=True)
    
    #Get twiss functions
    tao.cmd('set global lattice_calc_on = T')
    s=np.array([tao.lat_list(x,'ele.s')[0] for x in floorplan.Index])
    x=np.array([tao.lat_list(x,'orbit.floor.x')[0] for x in floorplan.Index])
    beta_y=np.array([tao.lat_list(x,'ele.a.beta')[0] for x in floorplan.Index])
    beta_x=np.array([tao.lat_list(x,'ele.b.beta')[0] for x in floorplan.Index])
    etot=np.array([tao.lat_list(x,'ele.e_tot')[0] for x in floorplan.Index])
    eta_y=np.array([tao.lat_list(x,'ele.y.eta')[0] for x in floorplan.Index])
    eta_x=np.array([tao.lat_list(x,'ele.x.eta')[0] for x in floorplan.Index])
    
    fig = plt.figure(num=1,figsize=[3.375*5,3.375*2])
    fig.clf()
    ax,ax_fp=format_longitudinal_plot(fig, floorplan)
    
    ax.semilogy(s,beta_x,label='beta b')
    ax.semilogy(s,beta_y,label='beta a')
    plt.legend(loc=2)
    ax.set_ylim([ymin, ymax])
    ax_r=ax.twinx()
    ax_r.plot(s,eta_x*1e3,'C0--',label='eta b')
    ax_r.plot(s,eta_y*1e3,'C1--',label='eta a')
    plt.legend(loc=1)
    
    
    ax.set_facecolor('w')
    
    ax.set_xlabel('Z [m]',fontsize=14)
    ax.set_ylabel(r'$\beta$ [m]',fontsize=14)
    ax_r.set_ylabel(r'$\eta$ [mm]',fontsize=14)
    
    ax.set_xlim([zmin,zmax])
    ax_fp.set_ylim([-1,3])
    
    plt.show()
    #fig.savefig('beamline',transparent=False,bbox_inches='tight', dpi=300)