"""
Make a full broadband event catalogue for the entire Sorsdal deployment. Use the new Event and Chunk objects as code wrappers
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
from obspy.core.inventory import inventory
import os

t1 = UTCDateTime(2017,12,31)
t2 = UTCDateTime(2018,2,16) #for now, just test with a couple of days to check all is working

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/redo_bbs'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/processed'


for path in [c_path]:
    if not os.path.exists(path):
        os.mkdir(path)

inv = inventory.read_inventory(stat_path,level='response').select(station='BBS??')
chunk = do.SeismicChunk(t1,t2,time_offset=7)

avail_rows = []

for daychunk in chunk(24*60*60):
    daychunk.attach_waveforms(inv,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
    daychunk.filter('highpass',freq=1)
    daychunk.context('detect')
    daychunk.detect_events(c_path,trigger_type='multistalta',sta=0.2,lta=4,delta_sta=20,delta_lta=20,epsilon=1.3,thr_on=4,thr_off=3,thr_coincidence_sum=2,avg_wave_speed=1.5,thr_event_join=2.0) #needs to be at least 2 to avoid double event IDs - need better way to do this in future...

# split_stream = daychunk.stream.slice(daychunk.starttime,daychunk.endtime).split()

# for tr in split_stream:
#     network, station, location, channel = tr.id.split('.')
#     row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
#     row = pd.DataFrame(data=row,index=[tr.id])
#     avail_rows.append(row)

# avail = pd.concat(avail_rows,ignore_index=True)
# avail.to_csv('stream_availability.csv')