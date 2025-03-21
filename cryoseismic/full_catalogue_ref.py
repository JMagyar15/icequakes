"""
Make a full broadband event catalogue for the entire Sorsdal deployment. Use the new Event and Chunk objects as code wrappers
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
from obspy.core.inventory import inventory
import os
from iqvis.event_calculation import WaveformAttributes, SpectralAttributes
import pandas as pd


detect = True
comp_att = False

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,8) #for now, just test with a couple of days to check all is working

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/ref'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed'


for path in [c_path]:
    if not os.path.exists(path):
        os.mkdir(path)

inv = inventory.read_inventory(stat_path,level='response').select(station='REF??')
chunk = do.SeismicChunk(t1,t2,time_offset=7)


if detect:
    chunk.context('detect')
    print('Loading and attaching waveforms...')
    chunk.attach_waveforms(inv,w_path)

    chunk.stream = chunk.stream.split()

    rows = []
    for tr in chunk.stream:
        network, station, location, channel = tr.id.split('.')
        row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
        row = pd.DataFrame(data=row,index=[tr.id])
        rows.append(row)
    avail = pd.concat(rows,ignore_index=True)

    avail.to_csv('ref_stream_availability.csv')


    chunk.stream.decimate(10)

    chunk.stream.merge(fill_value=None)

    print('Detecting events...')
    chunk.detect_events(c_path,freq=1,trigger_type='multistalta',sta=0.2,lta=4,delta_sta=20,delta_lta=20,epsilon=1.3,thr_on=4,thr_off=3,thr_coincidence_sum=4,avg_wave_speed=1.5,thr_event_join=2.0) #needs to be at least 2 to avoid double event IDs - need better way to do this in future...

    

    chunk.stream = None


event_cat = do.EventCatalogue(t1,t2,c_path) #get the event catalogue

if comp_att:
    event_cat.compute_attributes(inv,c_path,w_path,WaveformAttributes,SpectralAttributes)