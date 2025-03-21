import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from iqvis import spatial_analysis as sa
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import obspy
from obspy.signal.filter import envelope
from obspy.geodetics import degrees2kilometers
import multiprocessing as mp
import tqdm
import os
import glob
import pandas as pd
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
ref_inv = inv.select(station='REF??',channel='??Z')
bbs06 = inv.select(station='BBS06')

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,1)

chunk = do.SeismicChunk(t1,t2)
event_cat = do.EventCatalogue(t1,t2,c_path)

backazimuth_grid = np.linspace(0,2*np.pi,51)[:-1]
radial_grid = np.logspace(-5,4,50,base=2)
velocity_grid = np.linspace(1.4,2.3,50)
slowness_grid = 1 / velocity_grid

grid, _ = sa.radial_grid(backazimuth_grid,radial_grid,slowness_grid,ref_inv)


"""
MATCHED FIELD PROCESSING COMPUTATIONS
"""

for daychunk in chunk:
    if sta_lta:
        day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        mfp = day_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'mfp_opt__' + daychunk.str_name + '.csv')
    else:
        day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path,templates=True)
        mfp = day_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'template_mfp_opt__' + daychunk.str_name + '.csv')
    
    for event in tqdm.tqdm(day_cat,total=day_cat.N):

        event.attach_waveforms(ref_inv,w_path,buffer=2,length=4,extra=0)
        event.context('beamforming')

        event.fourier(frqlow=3,frqhigh=10)
        event.geometry(grid)
        B = event.coherence()

        ind = np.argmax(B)
        ind = np.unravel_index(ind,B.shape)

        baz = backazimuth_grid[ind[0]]
        rad = radial_grid[ind[1]]
        slow = slowness_grid[ind[2]]

        #save these grid search results
        mfp.at[event.event_id,'baz'] = baz
        mfp.at[event.event_id,'rad'] = rad
        mfp.at[event.event_id,'slow'] = slow
        mfp.at[event.event_id,'vel'] = 1/slow

        #now optimise for a variable slowness to try and improve the result...
        result = minimize(event.opt_objective,(baz,rad,slow),method='Nelder-Mead',bounds=[(-np.inf,np.inf),(radial_grid[0],radial_grid[-1]),(slowness_grid[-1],slowness_grid[0])])
        baz_opt, rad_opt, slow_opt = result.x
        baz_opt = baz_opt % (2*np.pi)

        mfp.at[event.event_id,'baz_opt'] = baz_opt
        mfp.at[event.event_id,'rad_opt'] = rad_opt
        mfp.at[event.event_id,'slow_opt'] = slow_opt
        mfp.at[event.event_id,'vel_opt'] = 1/slow_opt


    #to .csv with the MFP results.
    mfp.to_csv(filename)
