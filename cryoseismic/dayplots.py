"""
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
from iqvis import dayplot_backend as db
from obspy.core.inventory import inventory
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm


"""
CHUNK AND PATH SET-UP
"""

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,1,30)
time_offset = 7
chunk = do.SeismicChunk(t1,t2,time_offset=time_offset)

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed' #! note that this has been changed to unprocessed waveforms. Filtering with dentrending should hopefully fix this.


if not os.path.exists(c_path):
    os.mkdir(c_path)

bbs_path = os.path.join(c_path,'bbs')
ref_path = os.path.join(c_path,'ref')
ref_un_path = os.path.join(c_path,'ref_un')

if not os.path.exists(bbs_path):
    os.mkdir(bbs_path)

if not os.path.exists(ref_path):
    os.mkdir(ref_path)

if not os.path.exists(ref_un_path):
    os.mkdir(ref_un_path)

"""
STATION SELECTION
"""

total_inv = inventory.read_inventory(stat_path,level='response')
drop = ['REF01','REF03','REF08']

inv = total_inv.copy()
for station in drop:
    inv = inv.remove(station=station)

inv_bbs = inv.select(station='BBS??')
inv_ref = inv.select(station='REF??')

"""
DAYPLOT PARAMETERS
"""

dpi = 100
width = 8
height = 10
mins_per_row = 10
page_length = 6*60*60

fig = plt.figure(figsize=(width,height))
bbs_dayplot = db.StackPlot(fig,mins_per_row=mins_per_row,time_offset=time_offset)
ref_dayplot = db.StackPlot(fig,mins_per_row=mins_per_row,time_offset=time_offset)

"""
EVENT DETECTION
"""
for daychunk in chunk(24*60*60):
    daychunk.attach_waveforms(inv_bbs,w_path,buffer=60*60)
    daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)

    for page_chunk in daychunk(6*60*60):
        bbs_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,bbs_path)
        bbs_cat.events['group'] = 2
        ref_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_path)
        ref_cat.events['group'] = 7
        ref_un_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_un_path)
        ref_un_cat.events['group'] = 8

        combined_cat = (bbs_cat + ref_cat) + ref_un_cat

        bbs_dayplot.process_chunk(page_chunk)
        bbs_dayplot.add_events(page_chunk,combined_cat)

    daychunk.attach_waveforms(inv_ref,w_path,buffer=60*60)
    daychunk.decimate(10)
    daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)

    for page_chunk in daychunk(6*60*60):
        bbs_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,bbs_path)
        bbs_cat.events['group'] = 2
        ref_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_path)
        ref_cat.events['group'] = 7
        ref_un_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_un_path)
        ref_un_cat.events['group'] = 8

        combined_cat = (bbs_cat + ref_cat) + ref_un_cat

        ref_dayplot.process_chunk(page_chunk)
        ref_dayplot.add_events(page_chunk,combined_cat)

bbs_dayplot.normalise_traces()
ref_dayplot.normalise_traces()

for tr_id in inv_bbs.get_contents()['channels']:
    #make a pdf document for this channel
    filename = os.path.join(bbs_path,'dayplot__' + tr_id + '__' + chunk.str_name + '.pdf')
    p = PdfPages(filename) #set up the object used to save each channel as page in a pdf
    for page_chunk in chunk(page_length):
        fig, ax = bbs_dayplot.make_page(tr_id,page_chunk)
        fig.savefig(p,format='pdf')
    p.close()

for tr_id in inv_ref.get_contents()['channels']:
    #make a pdf document for this channel
    filename = os.path.join(ref_path,'dayplot__' + tr_id + '__' + chunk.str_name + '.pdf')
    p = PdfPages(filename) #set up the object used to save each channel as page in a pdf
    for page_chunk in chunk(page_length):
        fig, ax = ref_dayplot.make_page(tr_id,page_chunk)
        fig.savefig(p,format='pdf')
    p.close()