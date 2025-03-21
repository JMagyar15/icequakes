import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from obspy.core import UTCDateTime
import tqdm
import os
import glob
import pandas as pd
from iqvis import spatial_analysis as sa

local_path = '/Users/jmagyar/Documents/SorsdalData'
cloud_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes'

sta_lta = False

c_path = os.path.join(cloud_path,'sorsdal_catalogues')
stat_path = os.path.join(local_path,'stations/sorsdal_stations.xml')
w_path = os.path.join(local_path,'waveforms')
p_path = cloud_path
class_path = c_path

inv = inventory.read_inventory(stat_path,level='response')
ref_inv = inv.select(station='REF??')
bbs_inv = inv.select(station='BBS??')


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)

chunk = do.SeismicChunk(t1,t2)


for daychunk in chunk:

    if sta_lta:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        att_cat = event_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'waveform_attributes__' + daychunk.str_name + '.csv')
    else:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path,templates=True)
        att_cat = event_cat.attributes.drop(labels=['group','similarity'],axis=1)
        filename = os.path.join(c_path,'template_waveform_attributes__' + daychunk.str_name + '.csv')

    for event in tqdm.tqdm(event_cat,total=event_cat.N):

        event.attach_waveforms(bbs_inv,w_path,buffer=2,length=4)
        event.filter('highpass',freq=1)
        event.remove_sensitivity()
        event.context('spectral')
        event.get_power_spectrum()
        window = event.get_data_window()
        dt = window[0].stats.delta

        amps = []
        freqs = []
        mean_freqs = []
        energies = []
        for net in event.inv:
            for sta in net:
                sta_stream = window.select(network=net.code,station=sta.code)
                flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                max_amp = np.max(amp)
                amps.append(max_amp)

                energy = np.sum(amp**2) * dt
                energies.append(energy)

                psd = event.psds[sta.code].Z
                cumulative_psd = np.cumsum(psd) 
                cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                i50 = np.abs(cumulative_psd - 0.50).argmin()

                central_freq = event.psds[sta.code].f[i50]
                freqs.append(central_freq)

                psd /= np.sum(psd)
                weighted_freq = np.sum(psd * event.psds[sta.code].f)
                mean_freqs.append(weighted_freq)
        
        ind = np.nanargmax(amps)
        max_amp = amps[ind]
        freq = freqs[ind]
        energy = energies[ind]
        mean_freq = mean_freqs[ind]

        att_cat.at[event.event_id,'amplitude'] = max_amp
        att_cat.at[event.event_id,'med_freq'] = freq
        att_cat.at[event.event_id,'duration'] = event.duration
        att_cat.at[event.event_id,'energy'] = energy
        att_cat.at[event.event_id,'mean_freq'] = mean_freq

    att_cat.to_csv(filename)