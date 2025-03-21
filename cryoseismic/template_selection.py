from obspy.core import UTCDateTime
from iqvis import data_objects as do
import matplotlib.pyplot as plt
from obspy.core.inventory import inventory
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster import hierarchy
from scipy.spatial import distance
import numpy as np
from iqvis import event_calculation as ec
import tqdm

compute_xcorr = False
plot_dendrograms = True
plot_segment_analysis = False

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/bbs'
class_path = '/Users/jaredmagyar/Documents/SorsdalData/classification'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
plot_path = '/Users/jaredmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/event_plots'
env_path = '/Users/jaredmagyar/Documents/SorsdalData/environmental'

c = ['maroon','mediumvioletred','red','darkorange','forestgreen','darkcyan','darkslateblue','darkviolet']

threshold_dict = {'1':0.7,
                '2':0.7,
                '3':0.5,
                '4':0.55,
                '5':0.5,
                '6':0.65,
                '7':0.55,
                '8':0.55}


if not os.path.exists(plot_path):
    os.mkdir(plot_path)

starttime = UTCDateTime(2018,1,8)
endtime = UTCDateTime(2018,2,15)
chunk = do.SeismicChunk(starttime,endtime)

event_cat = do.EventCatalogue(starttime,endtime,c_path)

inv = inventory.read_inventory(stat_path,level='response')
inv = inv.select(station='BBS??')
long = inv[0][0][0].longitude

class_cat = pd.read_csv(os.path.join(class_path,'classified_catalogue.csv'),index_col=0) 


start1 = UTCDateTime(2018,1,8)
end1 = UTCDateTime(2018,1,8,8)

start2 = UTCDateTime(2018,1,20,8)
end2 = UTCDateTime(2018,1,20,16)

start3 = UTCDateTime(2018,1,14,16)
end3 = UTCDateTime(2018,1,15)


inner_cat = do.EventCatalogue(start1,end1,c_path) + do.EventCatalogue(start2,end2,c_path) + do.EventCatalogue(start3,end3,c_path)
outer_cat = do.EventCatalogue(start1,end1,c_path) + do.EventCatalogue(start2,end2,c_path) + do.EventCatalogue(start3,end3,c_path)


inner_cat.add_classification(class_cat)
outer_cat.add_classification(class_cat)


inner_cat_groups = inner_cat.group_split()
outer_cat_groups = outer_cat.group_split()

if compute_xcorr:

    for group in range(1,9):

        print('Cross-correlation clustering group',group)

        i_cat = inner_cat_groups[group]
        o_cat = outer_cat_groups[group]

        cc_mat = np.zeros((i_cat.N,o_cat.N))
        shift_mat = np.zeros((i_cat.N,o_cat.N))

        for i, outer in tqdm.tqdm(enumerate(o_cat),total=o_cat.N):
            outer.attach_waveforms(inv,w_path,buffer=2,length=4)
            outer.filter('highpass',freq=1)
            outer_tr = outer.template_trace()

            for j, inner in enumerate(i_cat):
                if i == j:
                    cc_mat[i,j] = 1.0
                    shift_mat[i,j] = 0
                elif i < j:
                    inner.attach_waveforms(inv,w_path,buffer=2,length=4)
                    inner.filter('highpass',freq=1)
                    inner_tr = inner.template_trace()
                    shift, xcorr = ec.xcorr_data(outer_tr,inner_tr)
                    cc_mat[i,j] = xcorr
                    cc_mat[j,i] = xcorr
                    shift_mat[i,j] = shift
                    shift_mat[j,i] = -shift

        dissimilarity = 1-cc_mat
        dissimilarity = distance.squareform(dissimilarity)

        filename = os.path.join(c_path,'group_' + str(group) + '_xcorr.npz')
        np.savez(filename,xcorr=cc_mat,shift=shift_mat,diss=dissimilarity)

if plot_dendrograms:
    for group in range(1,9):

        o_cat = outer_cat_groups[group]

        filename = os.path.join(c_path,'group_' + str(group) + '_xcorr.npz')

        full = np.load(filename,allow_pickle=True)
        dissimilarity = full['diss']
        shift_mat = full['shift']
    
        threshold = 1 - threshold_dict[str(group)]
        linkage = hierarchy.linkage(dissimilarity, method="average")#,optimal_ordering=True)
        clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

        event_list = o_cat.events.index.to_list()

        fig, [dend_ax,waveform_ax] = plt.subplots(ncols=2,figsize=(8,len(event_list)/4),gridspec_kw={'width_ratios':[1,2],'wspace':0},layout='constrained')

        dend_ax.spines['top'].set_visible(False)
        dend_ax.spines['bottom'].set_visible(False)
        dend_ax.spines['left'].set_visible(False)
        dend_ax.spines['right'].set_visible(False)



        waveform_ax.spines['top'].set_visible(False)
        waveform_ax.spines['bottom'].set_visible(False)
        waveform_ax.spines['left'].set_visible(False)
        waveform_ax.spines['right'].set_visible(False)

        hierarchy.set_link_color_palette(['darkgreen','darkred','slategrey','peru','deeppink','olive'])

        dend_ax.vlines(threshold,ymin=0,ymax=10*len(event_list),color='grey',ls='--',lw=0.8)

        dendrogram = hierarchy.dendrogram(linkage,color_threshold=threshold,orientation='left',ax=dend_ax,labels=event_list,no_labels=True)


        old_index = None
        station_names = []
        for i, event_id in enumerate(dendrogram['ivl']):
            offset = 5 + i * 10

            index = event_list.index(event_id)

            event = o_cat.select_event(event_id)
            event.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
            event.filter('highpass',freq=1)

            max_tr = event.template_trace()
            # max_energy = 0.0
            # for tr in event.get_data_window().select(component='Z'):
            #     energy = np.sum(tr.data**2)
            #     if energy > max_energy:
            #         max_energy = energy
            #         max_tr = tr
            
            net, sta, loc, ch = max_tr.id.split('.')
            station_names.append(sta)
            max_tr2 = event.stream.select(network=net,station=sta,channel=ch)[0]

            if i > 0:
                shift += shift_mat[index,old_index] / 100
                if np.abs(shift) > 3.0:
                    shift = 0.0 #reset shift if it becomes too large
            else:
                shift = 0.0

            old_index = index
                
            shift_window = [event.data_window[0] + shift, event.data_window[1] + shift]
            #tr = max_tr.trim(*shift_window,pad=True,fill_value=None)
            tr = max_tr2.trim(*shift_window,pad=True,fill_value=None)
            waveform_ax.plot(tr.times(),tr.data/tr.data.max()*5 + offset,color=dendrogram['leaves_color_list'][i],lw=0.8)

        waveform_ax.set_ylim((0,len(event_list)*10))
        waveform_ax.set_yticks(np.arange(0,len(event_list)*10,10)+5)
        #waveform_ax.set_yticklabels(station_names)
        waveform_ax.set_yticklabels(dendrogram['ivl'])
        waveform_ax.yaxis.set_ticks_position('right')

        dend_ax.set_xlabel('Dissimilarity')
        waveform_ax.set_xlabel('Relative Time (s)')
        fig.savefig(os.path.join(plot_path,'group_' + str(group) + '_cluster.pdf'),format='pdf')


if plot_segment_analysis:
    #TODO plots for the MFP and waveform attributes for the manually selected groups - not the template matched groups.
    #! then want to compare these plots with the same plots for the automatically assigned groups, hopefully look somewhat similar...

    #? compute the MFP and attribute results elsewhere as will want this for all events in catalogue, just pull the events in the manual groups here and plot those. 
    mfp = pd.read_csv(index_col=0)
    attributes = pd.read_csv(index_col=0)

    for event in inner_cat:
        mfp_event = mfp[event.event_id]
        att_event = attributes[event.event_id]
    pass