
from iqvis import data_objects as do
from iqvis import event_calculation as ec
import numpy as np
import pandas as pd
import os
import tqdm


from obspy.core import UTCDateTime
from obspy.core.inventory import inventory
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from obspy.signal.array_analysis import array_processing
from obspy.core.util import AttribDict

from scipy.signal import hilbert



beamforming = False
polarisation = True

"""
CHUNK SET-UP
"""
t1 = UTCDateTime(2018,1,5)
t2 = UTCDateTime(2018,1,15)

chunk = do.SeismicChunk(t1,t2)


c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/bbs'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/processed'
ref_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed'
class_path = '/Users/jaredmagyar/Documents/SorsdalData/classification'
plot_path = '/Users/jaredmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/event_plots'


inv_bbs = inventory.read_inventory(stat_path,level='response').select(station='BBS??',channel='???')
inv_ref = inventory.read_inventory(stat_path,level='response').select(station='REF??',channel='???')

event_cat = do.EventCatalogue(t1,t2,c_path)

if beamforming:

    for daychunk in chunk:

        day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        filename = os.path.join(c_path,'beamforming__' + daychunk.str_name + '.csv')
        print('Beamforming for',daychunk.starttime,'to',daychunk.endtime)

        beam_cat = day_cat.attributes
        #beam_cat.set_index('event_id',inplace=True)

        for event in tqdm.tqdm(day_cat,total=day_cat.N):
            event.attach_waveforms(inv_ref,ref_path,buffer=2,length=4)
            #event.decimate(20) #only for reftek
            
            stream = event.get_data_window().split()

            for tr in stream:
                network, station, location, channel = tr.id.split('.')
                tr.stats.coordinates = AttribDict({
                'latitude': event.inv.select(station=station)[0][0].latitude,
                'elevation': event.inv.select(station=station)[0][0].elevation/1000,
                'longitude': event.inv.select(station=station)[0][0].longitude})
                

            stime = event.data_window[0]
            etime = event.data_window[1]
            win_length = (etime - stime)

            kwargs = dict(
                # slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=-1.5, slm_x=1.5, sll_y=-1.5, slm_y=1.5, sl_s=0.03,
                # sliding window properties
                win_len=win_length, win_frac=1.0, 
                # frequency properties
                frqlow=1.0, frqhigh=20.0, prewhiten=0,
                # restrict output
                semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
                stime=stime, etime=etime
            )
            out = array_processing(stream, **kwargs)
            t, rel_power, abs_power, baz, slow = out.T
            baz[baz < 0.0] += 360
            max_ind = np.argmax(abs_power)
            baz = baz[max_ind]
            slow = slow[max_ind]

            beam_cat.at[event.event_id,'baz'] = baz
            beam_cat.at[event.event_id,'slow'] = slow

        #beam_cat.reset_index(inplace=True)
        beam_cat.to_csv(filename)


if polarisation:
    for daychunk in chunk:

        day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        filename = os.path.join(c_path,'polarisation__' + daychunk.str_name + '.csv')
        print('Polarisation for',daychunk.starttime,'to',daychunk.endtime)

        pol_cat = day_cat.attributes
        #pol_cat.set_index('event_id',inplace=True)

        for event in tqdm.tqdm(day_cat,total=day_cat.N):
            event.attach_waveforms(inv_bbs,w_path,buffer=2,length=4)
            event.filter('highpass',freq=1)
            for network in inv_bbs:
                for station in network:
                    backazimuth_grid = np.linspace(0,360,73)
                    xcorr_rayleigh = np.zeros_like(backazimuth_grid)
                    xcorr_pwave = np.zeros_like(backazimuth_grid)
                    xcorr_swave = np.zeros_like(backazimuth_grid)

                    for i, backazimuth in enumerate(backazimuth_grid):
                        stream = event.get_data_window().select(station=station.code).copy()
                        stream.rotate(method='NE->RT',back_azimuth=backazimuth)

                        radial = stream.select(component='R')[0].data
                        vertical = stream.select(component='Z')[0].data
                        transverse = stream.select(component='T')[0].data
                        
                        analytical_signal = hilbert(radial)
                        shifted = np.real(np.abs(analytical_signal) * np.exp((np.angle(analytical_signal) + 0.5 * np.pi) * 1j))
                        
                        xcorr_rayleigh[i] = np.dot(vertical,shifted) / np.sqrt(np.dot(vertical,vertical)*np.dot(shifted,shifted))
                        xcorr_pwave[i] = np.dot(vertical,radial) / np.sqrt(np.dot(vertical,vertical)*np.dot(radial,radial))
                        xcorr_swave[i] = np.dot(vertical,transverse) / np.sqrt(np.dot(vertical,vertical)*np.dot(transverse,transverse))


                    max_rayleigh = np.argmax(xcorr_rayleigh)
                    backaz_rayleigh = backazimuth_grid[max_rayleigh]
                    corr_rayleigh = np.max(xcorr_rayleigh)

                    max_pwave = np.argmax(xcorr_pwave)
                    backaz_pwave = backazimuth_grid[max_pwave]
                    corr_pwave = np.max(xcorr_pwave)

                    max_swave = np.argmax(xcorr_swave)
                    backaz_swave = backazimuth_grid[max_swave]
                    corr_swave = np.max(xcorr_swave)

                    pol_cat.at[event.event_id,'baz_r_'+station.code] = backaz_rayleigh
                    pol_cat.at[event.event_id,'baz_p_'+station.code] = backaz_pwave
                    pol_cat.at[event.event_id,'baz_s_'+station.code] = backaz_swave

                    pol_cat.at[event.event_id,'corr_r_'+station.code] = corr_rayleigh
                    pol_cat.at[event.event_id,'corr_p_'+station.code] = corr_pwave
                    pol_cat.at[event.event_id,'corr_s_'+station.code] = corr_swave

        #pol_cat.reset_index(inplace=True)
        pol_cat.to_csv(filename)