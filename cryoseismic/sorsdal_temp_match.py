import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from obspy.core import UTCDateTime
import os
import pandas as pd

threshold = 'low'

local_path = '/Users/jmagyar/Documents/SorsdalData'
cloud_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes'

c_path = os.path.join(cloud_path,'sorsdal_catalogues')
stat_path = os.path.join(local_path,'stations/sorsdal_stations.xml')
w_path = os.path.join(local_path,'waveforms')
p_path = cloud_path
class_path = c_path

inv = inventory.read_inventory(stat_path,level='response')
bbs_inv = inv.select(station='BBS??',channel='??Z')


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,16)
chunk = do.SeismicChunk(t1,t2)

event_cat = do.EventCatalogue(t1,t2,c_path)


all_templates = {'1a':'20180108T003015Z',
             '1b':'20180114T234955Z',
             '1c':'20180120T132740Z',
             '1d':'20180108T061313Z',
             '2a':'20180108T021558Z',
             '2b':'20180114T165412Z',
             '2c':'20180108T012013Z',
             '2d':'20180114T184018Z',
             '3a':'20180120T084849Z',
             '3b':'20180108T013417Z',
             '3c':'20180114T210649Z',
             '3d':'20180120T081146Z',
             '4a':'20180108T013024Z',
             '4b':'20180120T144129Z',
             '4c':'20180114T171343Z',
             '4d':'20180108T012122Z',
             '5a':'20180108T020737Z',
             '5b':'20180108T020121Z',
             '5c':'20180108T025047Z',
             '5d':'20180108T031250Z',
             '6a':'20180108T023943Z',
             '6b':'20180114T163252Z',
             '6c':'20180108T064747Z',
             '6d':'20180108T010129Z',
             '7a':'20180108T065407Z',
             '7b':'20180114T213023Z',
             '7c':'20180114T212327Z',
             '7d':'20180108T041343Z',
             '8a':'20180108T030040Z',
             '8b':'20180108T075115Z',
             '8c':'20180114T160953Z',
             '8d':'20180120T125805Z'
             }

low_threshold_dict = {'1':0.8,
                '2':0.75,
                '3':0.65,
                '4':0.75,
                '5':0.75,
                '6':0.75,
                '7':0.65,
                '8':0.65}

high_threshold_dict = {'1':0.9,
                '2':0.8,
                '3':0.7,
                '4':0.8,
                '5':0.8,
                '6':0.8,
                '7':0.7,
                '8':0.7}

full_thresholds = {}
full_templates = {}

if threshold == 'high':
    threshold_dict = high_threshold_dict
    save_path = os.path.join(c_path,'high_threshold')

else:
    threshold_dict = low_threshold_dict
    save_path = os.path.join(c_path,'low_threshold')


if not os.path.exists(save_path):
    os.mkdir(save_path)


for temp_name, temp_id in all_templates.items():
    template = event_cat.select_event(temp_id)
    template.attach_waveforms(bbs_inv,w_path,buffer=2,length=4,extra=10)
    template.filter('highpass',freq=2)
    #template.remove_response(pre_filt=[1,2,45,50])

    full_templates[temp_name] = template
    full_thresholds[temp_name] = threshold_dict[temp_name[0]]


for daychunk in chunk:
    print('Template Matching: ' + daychunk.str_name)
    daychunk.attach_waveforms(bbs_inv,w_path,buffer=60*60)
    #daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)   
    daychunk.filter('highpass',freq=2)
    daychunk.context('detect')

    daychunk.template_catalogue(save_path,full_templates,full_thresholds)
