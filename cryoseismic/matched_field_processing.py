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


if __name__ == '__main__':

    all_events = False
    group_events = False
    fixed_slowness = True


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

    backazimuth_grid = np.linspace(0,2*np.pi,101)[:-1]
    radial_grid = np.logspace(-5,4,50,base=2)
    slowness_grid = np.linspace(0.1,0.8,20)

    grid, _ = sa.radial_grid(backazimuth_grid,radial_grid,slowness_grid,ref_inv)


    """
    MATCHED FIELD PROCESSING COMPUTATIONS
    """

    if all_events:
        for daychunk in chunk:
            day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)

            mfp = day_cat.attributes.drop(labels='group',axis=1)
            filename = os.path.join(cloud_path,'mfp__' + daychunk.str_name + '.csv')

            
            for event in tqdm.tqdm(day_cat,total=day_cat.N):

                event.attach_waveforms(ref_inv,w_path,buffer=2,length=4,extra=10)
                #event.decimate(10)

                event.context('beamforming')

                #compute the cross spectral densities for the event
        
                beampower_grid = event.vertical_csd_frq_loop(grid,frqlow=3,frqhigh=50,remove_response=True,normalise=False,frq_step=1)
                #beampower_grid = event.parallelised_vertical_csd(grid,frqlow=3,frqhigh=50,remove_response=True,normalise=False,frq_step=1)

                ind = np.argmax(beampower_grid)
                ind = np.unravel_index(ind,beampower_grid.shape)

                mfp.at[event.event_id,'baz'] = np.rad2deg(backazimuth_grid[ind[0]])
                mfp.at[event.event_id,'rad'] = radial_grid[ind[1]]
                mfp.at[event.event_id,'slow'] = slowness_grid[ind[2]]

            #to .csv with the MFP results.
            mfp.to_csv(filename)

    if fixed_slowness:
        backazimuth_grid = np.linspace(0,2*np.pi,101)[:-1]
        radial_grid = np.logspace(-5,4,50,base=2)
        slowness_grid = np.array([0.5789473684210530])

        grid, _ = sa.radial_grid(backazimuth_grid,radial_grid,slowness_grid,ref_inv)
        for daychunk in chunk:
            day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)

            mfp = day_cat.attributes.drop(labels='group',axis=1)
            filename = os.path.join(c_path,'mfp_fixed_s__' + daychunk.str_name + '.csv')

            
            for event in tqdm.tqdm(day_cat,total=day_cat.N):

                event.attach_waveforms(ref_inv,w_path,buffer=2,length=4,extra=10)
                #event.decimate(10)

                event.context('beamforming')

                #compute the cross spectral densities for the event
        
                beampower_grid = event.vertical_csd_frq_loop(grid,frqlow=3,frqhigh=50,remove_response=True,normalise=False,frq_step=1)
                #beampower_grid = event.parallelised_vertical_csd(grid,frqlow=3,frqhigh=50,remove_response=True,normalise=False,frq_step=1)

                ind = np.argmax(beampower_grid)
                ind = np.unravel_index(ind,beampower_grid.shape)

                mfp.at[event.event_id,'baz'] = np.rad2deg(backazimuth_grid[ind[0]])
                mfp.at[event.event_id,'rad'] = radial_grid[ind[1]]
                mfp.at[event.event_id,'slow'] = slowness_grid[ind[2]]

            #to .csv with the MFP results.
            mfp.to_csv(filename)
    
    if group_events:
        """Do the full computationally expensive MFP, but only on the events that were classified to save computational time."""

        xcorr_files = glob.glob(os.path.join(c_path,'xcorr*'))

        xcorr_files.sort()

        xcorr = pd.concat([pd.read_csv(file,index_col=0) for file in xcorr_files],ignore_index=False)

        threshold_dict = {'1':0.8,
                '2':0.75,
                '3':0.65,
                '4':0.75,
                '5':0.75,
                '6':0.75,
                '7':0.65,
                '8':0.65}


        for i, row in xcorr.iterrows():
            for temp_name, xc in row.items():
                group = temp_name[:-1]
                thresh = threshold_dict[group]
                row.at[temp_name] = xc / thresh
            max_col = row.argmax()
            val = row.iloc[max_col]
            temp = xcorr.columns[max_col]
            xcorr.at[i,'max_xcorr'] = val
            if val > 1.0:
                xcorr.at[i,'template'] = temp
                xcorr.at[i,'group'] = temp[:-1]
            else:
                xcorr.at[i,'template'] = '0' #index for unmatched signals.
                xcorr.at[i,'group'] = '0'

        event_cat.add_classification(xcorr.reset_index())

        temp_groups = event_cat.group_split()

        for group, group_cat in temp_groups.items():
            if int(group) in [1,2,3,4,5,6,7,8]:

                print('Matched Field Processing for group',group)
                filename = os.path.join(cloud_path,'mfp__group__' + group + '.csv')
                mfp = group_cat.attributes.drop(labels='group',axis=1)

                for event in tqdm.tqdm(group_cat,total=group_cat.N):
                    event.attach_waveforms(ref_inv,w_path,buffer=2,length=4,extra=10)
                    event.context('beamforming')

                    beampower_grid = event.vertical_csd_frq_loop(grid,frqlow=3,frqhigh=50,remove_response=True,normalise=False,frq_step=1)

                    ind = np.argmax(beampower_grid)
                    ind = np.unravel_index(ind,beampower_grid.shape)

                    mfp.at[event.event_id,'baz'] = np.rad2deg(backazimuth_grid[ind[0]])
                    mfp.at[event.event_id,'rad'] = radial_grid[ind[1]]
                    mfp.at[event.event_id,'slow'] = slowness_grid[ind[2]]

                #to .csv with the MFP results.
                mfp.to_csv(filename)