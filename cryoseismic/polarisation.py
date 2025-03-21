import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from obspy.core import UTCDateTime
import tqdm
import os
import glob
import pandas as pd
from iqvis import spatial_analysis as sa
from iqvis import event_calculation as ec
from scipy.optimize import minimize

sta_lta = False

local_path = '/Users/jmagyar/Documents/SorsdalData'
cloud_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes'

c_path = os.path.join(cloud_path,'sorsdal_catalogues')
stat_path = os.path.join(local_path,'stations/sorsdal_stations.xml')
w_path = os.path.join(local_path,'waveforms')
p_path = cloud_path
class_path = c_path


inv = inventory.read_inventory(stat_path,level='response')
ref_inv = inv.select(station='REF??')
bbs_inv = inv.select(station='BBS??')


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,16)

chunk = do.SeismicChunk(t1,t2)


#grid here doesn't matter, just need the centre to get the backazimuth from the stations.
backazimuth_grid = np.linspace(0,2*np.pi,101)[:-1]
radial_grid = np.logspace(-5,4,50,base=2)
slowness_grid = np.linspace(0.1,0.8,20)

grid, mfp_centre = sa.radial_grid(backazimuth_grid,radial_grid,slowness_grid,ref_inv)

chunk = do.SeismicChunk(t1,t2)

backazimuth_grid = np.linspace(0,2*np.pi,51)[:-1]
radial_grid = np.logspace(-5,4,50,base=2)

baz, _ = sa.polarisation_grid(backazimuth_grid,radial_grid,bbs_inv,centre=mfp_centre)

for daychunk in chunk:
    if sta_lta:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        filename = os.path.join(c_path,'polarisation_localisation_opt__' + daychunk.str_name + '.csv')
        pol_cat = event_cat.attributes.drop(labels='group',axis=1)
    
    else:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path,templates=True)
        filename = os.path.join(c_path,'template_polarisation_localisation_opt__' + daychunk.str_name + '.csv')
        pol_cat = event_cat.attributes.drop(labels=['group','similarity'],axis=1)


    
    for event in tqdm.tqdm(event_cat,total=event_cat.N):

        event.attach_waveforms(bbs_inv,w_path,buffer=2,length=4)
        event.filter('bandpass',freqmin=3,freqmax=10)
        event.context('polarisation')

        event.phase_shift()
        event.geometry(baz,mfp_centre)
        event.rotate()

        R_corr, total_R_corr = event.correlation()

        ind = np.argmax(total_R_corr)
        ind = np.unravel_index(ind,total_R_corr.shape)

        baz_loc = backazimuth_grid[ind[0]]
        rad_loc = radial_grid[ind[1]]

        pol_cat.at[event.event_id,'baz'] = baz_loc
        pol_cat.at[event.event_id,'rad'] = rad_loc
        pol_cat.at[event.event_id,'corr'] = total_R_corr[ind[0],ind[1]]

        result = minimize(event.opt_objective,(baz_loc,rad_loc),method='Nelder-Mead',bounds=[(-np.inf,np.inf),(radial_grid[0],radial_grid[-1])])
        baz_opt, rad_opt = result.x
        baz_opt = baz_opt % (2*np.pi)
        corr_opt = -1.0 * result.fun

        pol_cat.at[event.event_id,'baz_opt'] = baz_opt
        pol_cat.at[event.event_id,'rad_opt'] = rad_opt
        pol_cat.at[event.event_id,'corr_opt'] = corr_opt

        pol_cat.at[event.event_id,'N'] = R_corr.shape[-1]

    
    pol_cat.to_csv(filename)
