from iqvis import data_objects as do
from iqvis import event_calculation as ec
import numpy as np
import pandas as pd
import os
import tqdm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import glob

from obspy.core import UTCDateTime
from obspy.core.inventory import inventory

from matplotlib.backends.backend_pdf import PdfPages


plot_templates = True
template_match = False
plot_groups = False

"""
CHUNK SET-UP
"""
t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)

chunk = do.SeismicChunk(t1,t2)


c_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/sorsdal_catalogues'
stat_path = '/Users/jmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jmagyar/Documents/SorsdalData/waveforms'
plot_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/event_plots'


inv_bbs = inventory.read_inventory(stat_path,level='response').select(station='BBS??',channel='???')

event_cat = do.EventCatalogue(t1,t2,c_path)


#choose a template event based off dendrograms

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

threshold_dict = {'1':0.8,
                '2':0.75,
                '3':0.65,
                '4':0.75,
                '5':0.75,
                '6':0.75,
                '7':0.65,
                '8':0.65}

c = ['maroon','mediumvioletred','red','darkorange','forestgreen','darkcyan','darkslateblue','darkviolet']

if plot_templates:
    from matplotlib import rc
    import matplotlib.font_manager as fm
    rc('text', usetex=True)
    rc('font', size=8)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Optima']})

    one_col = 3.35 #inches
    two_col = 7.0 #inches
    
    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(one_col,one_col*1.5),gridspec_kw={'hspace':0.1},layout='constrained')
    fig2, axes2 = plt.subplots(nrows=2,ncols=4,figsize=(two_col,two_col*0.5),gridspec_kw={'hspace':0.1},layout='constrained')
    axes = axes.flatten(order='F')
    axes2 = axes2.flatten(order='C')

    for ii, group in enumerate(threshold_dict):
        ax = axes[ii]
        ax2 = axes2[ii]

        ax.set_axis_off()
        ax2.set_axis_off()

        ax.annotate(group,(0,1),xycoords='axes fraction',weight='black')
        #ax.annotate('threshold: ' + str(threshold_dict[group]),(1,0),xycoords='axes fraction',ha='right',va='top')

        ax2.annotate(group,(0,1),xycoords='axes fraction',weight='black')
        #ax2.annotate('threshold: ' + str(threshold_dict[group]),(1,0),xycoords='axes fraction',ha='right',va='top')

        left = mcolors.to_rgb('white')
        middle = mcolors.to_rgb(c[ii])
        right = mcolors.to_rgb('black')
        custom_cm = LinearSegmentedColormap.from_list('custom',[left,middle,right])


        sublist = []
        for key in all_templates:
            if key[:-1] == group:
                sublist.append(key)

        templates = {k:all_templates[k] for k in sublist}


        xcorr_mat = np.zeros((len(templates),len(templates)))
        shift_mat = np.zeros((len(templates),len(templates)))

        for i, event_id_1 in enumerate(templates.values()):
            event_1 = event_cat.select_event(event_id_1)
            event_1.attach_waveforms(inv_bbs,w_path,buffer=2,length=4,extra=10)
            event_1.filter('highpass',freq=1)

            for j, event_id_2 in enumerate(templates.values()):
                event_2 = event_cat.select_event(event_id_2)
                event_2.attach_waveforms(inv_bbs,w_path,buffer=2,length=4,extra=10)
                event_2.filter('highpass',freq=1)

                shift, xc = ec.cross_correlate(event_1,event_2)
                xcorr_mat[i,j] = xc
                shift_mat[i,j] = shift

        i = 0
        for name, template in templates.items():
            frac = (i+2) / (len(templates)+2)

            event = event_cat.select_event(template)
            
            event.attach_waveforms(inv_bbs,w_path,buffer=2,length=4,extra=10)
            event.filter('highpass',freq=1)

            max_energy = 0.0
            for tr in event.get_data_window().select(component='Z'):
                energy = np.sum(tr.data**2)
                if energy > max_energy:
                    max_energy = energy
                    max_tr = tr
            
            net, sta, loc, ch = max_tr.id.split('.')
            max_tr2 = event.stream.select(network=net,station=sta,channel=ch)[0]

            if i > 0:
                shift += shift_mat[i,i-1] / 100
            else:
                shift = 0.0

                
            shift_window = [event.data_window[0] + shift, event.data_window[1] + shift]
            #tr = max_tr.trim(*shift_window,pad=True,fill_value=None)
            tr = max_tr2.trim(*shift_window,pad=True,fill_value=None)

            offset = 10*i

            ax.plot(tr.times(),tr.data/np.abs(tr.data).max()*4 + offset,lw=0.5,color=custom_cm(frac))
            ax2.plot(tr.times(),tr.data/np.abs(tr.data).max()*4 + offset,lw=0.4,color=custom_cm(frac))

            i += 1


    fig.savefig(os.path.join(plot_path,'templates.pdf'),format='pdf')
    fig.savefig(os.path.join(plot_path,'templates.eps'),bbox_inches='tight')

    fig2.savefig(os.path.join(plot_path,'templates_landscape.pdf'),format='pdf')
    fig2.savefig(os.path.join(plot_path,'templates_landscape.eps'),bbox_inches='tight')


if template_match:
    for daychunk in chunk:

        day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        filename = os.path.join(c_path,'xcorr__' + daychunk.str_name + '.csv')

        xcorr = day_cat.attributes.drop(labels='group',axis=1)
        #xcorr.set_index('event_id',inplace=True)
        
        for i, event in tqdm.tqdm(enumerate(day_cat),total=day_cat.N):
            event.attach_waveforms(inv_bbs,w_path,buffer=2,length=4)
            event.filter('highpass',freq=1)
            candidate_tr = event.template_trace()

            for template_name, event_id in all_templates.items():
                template = event_cat.select_event(event_id)
                template.attach_waveforms(inv_bbs,w_path,buffer=2,length=4)
                template.filter('highpass',freq=1)
                template_tr = template.template_trace()

                shift, xc = ec.xcorr_data(template_tr,candidate_tr)
                xcorr.at[event.event_id,template_name] = xc

        xcorr.to_csv(filename)


if plot_groups:

    inv = inventory.read_inventory(stat_path,level='response')
    inv = inv.select(station='BBS??')
    
    xcorr_files = glob.glob(os.path.join(c_path,'xcorr*'))

    xcorr_files.sort()

    xcorr = pd.concat([pd.read_csv(file,index_col=0) for file in xcorr_files],ignore_index=False)


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
            filename = os.path.join(plot_path,'temp_group__'+str(group)+'.pdf')
            p = PdfPages(filename)
            print('Plotting group',str(group))

            i = 0
            for event in tqdm.tqdm(group_cat,total=group_cat.N):
                event.attach_waveforms(inv,w_path,buffer=2,length=4)
                event.filter('highpass',freq=1)
                event.context('spectral')
                event.get_spectrograms(32,24)
                event.get_power_spectrum()
                event.context('plot')
                fig = event.new_plotting(plt.figure(figsize=[10,6],layout='compressed'),components=['Z'],spectrogram='right')
                fig.savefig(p,format='pdf')
                plt.close(fig)

                i += 1
                #if i > 100:
                #    break

            p.close()