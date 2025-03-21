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
from obspy.imaging.util import _set_xaxis_obspy_dates
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import tqdm

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/bbs'
class_path = '/Users/jaredmagyar/Documents/SorsdalData/classification'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
plot_path = '/Users/jaredmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/event_plots'
env_path = '/Users/jaredmagyar/Documents/SorsdalData/environmental'

one_col = 3.35 #inches
two_col = 7.0 #inches

dendrogram = False
plot_groups = False
plot_xcorr = False
xcorr_summary = False
plot_templates = False
plot_cycles = True

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

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

starttime = UTCDateTime(2018,1,1)
endtime = UTCDateTime(2018,2,15)
chunk = do.SeismicChunk(starttime,endtime)

event_cat = do.EventCatalogue(starttime,endtime,c_path)

inv = inventory.read_inventory(stat_path,level='response')
inv = inv.select(station='BBS??')
long = inv[0][0][0].longitude

class_cat = pd.read_csv(os.path.join(class_path,'classified_catalogue.csv'),index_col=0) 

event_cat.add_classification(class_cat)

catalogues = event_cat.group_split()

if plot_groups:

    for group, group_cat in catalogues.items():
        if group in [1,2,3,4,5,6,7,8]:
            filename = os.path.join(plot_path,'group__'+str(group)+'.pdf')
            p = PdfPages(filename)
            print('Plotting group',str(group))

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

            p.close()


if plot_xcorr:
    
    xcorr = []
    beamform = []
    for daychunk in chunk:
        xcorr_name = os.path.join(c_path,'xcorr__' + daychunk.str_name + '.csv')
        beamform_name = os.path.join(c_path,'beamforming__' + daychunk.str_name + '.csv')

        if os.path.isfile(xcorr_name):
            xcorr.append(pd.read_csv(xcorr_name,index_col=0))
        if os.path.isfile(beamform_name):
            beamform.append(pd.read_csv(beamform_name,index_col=0))

    xcorr = pd.concat(xcorr,ignore_index=True)
    beamform = pd.concat(beamform,ignore_index=True)

    xcorr.set_index('event_id',inplace=True)

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

    xcorr.reset_index(inplace=True)

    event_cat.add_classification(xcorr) #assign the groups from the x-correlation to the event catalogue for plotting

    catalogues = event_cat.group_split()


    for group, group_cat in catalogues.items():
        if group != 0:
            print('Plotting group ' + str(group))
            filename = os.path.join(plot_path,'template_group__'+str(group)+'.pdf')
            p = PdfPages(filename)
            

            for event in group_cat:
                event.attach_waveforms(inv,w_path,buffer=2,length=4)
                event.filter('highpass',freq=1)
                event.context('spectral')
                event.get_spectrograms(32,24)
                event.get_power_spectrum()
                event.context('plot')
                fig = event.new_plotting(plt.figure(figsize=[10,6],layout='compressed'),components=['Z'],spectrogram='right')
                fig.savefig(p,format='pdf')

            p.close()


if xcorr_summary:

    #dictionary of the template names and associated event ids.
    import glob

    xcorr_files = glob.glob(os.path.join(c_path,'xcorr*'))
    mfp_files = glob.glob(os.path.join(c_path,'mfp*'))

    xcorr_files.sort()
    mfp_files.sort()

    xcorr = pd.concat([pd.read_csv(file,index_col=0) for file in xcorr_files],ignore_index=False)
    mfp = pd.concat([pd.read_csv(file,index_col=0) for file in mfp_files],ignore_index=False)

    mfp['baz'] *= (np.pi/180) #convert to radians.

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

    #now loop through the groups.
    for ii, group in enumerate(threshold_dict):
        sublist = []
        for key in all_templates:
            if key[:-1] == group:
                sublist.append(key)

        #make a pdf file to save the group summaries to
        filename = os.path.join(plot_path,'group__' + group + '__summary.pdf')
        p = PdfPages(filename)

        templates = {k:all_templates[k] for k in sublist}

        fig = plt.figure(figsize=(8,6),constrained_layout=True)

        gs = fig.add_gridspec(ncols=3,nrows=3,width_ratios=[2,1,1],height_ratios=[1,1,2])

        waveform_ax = fig.add_subplot(gs[:2,0])
        waveform_ax.spines['top'].set_visible(False)
        waveform_ax.spines['bottom'].set_visible(False)
        waveform_ax.spines['left'].set_visible(False)
        waveform_ax.spines['right'].set_visible(False)
        waveform_ax.set_yticks([])
        waveform_ax.set_title('Templates')

        rose_ax = fig.add_subplot(gs[2,0],projection='polar')

        rose_ax.set_theta_zero_location('N',offset=-long)
        rose_ax.set_theta_direction(-1)
        rose_ax.set_xticks([])
        rose_ax.set_yticks([])
        rose_ax.spines['polar'].set_visible(False)
        rose_ax.set_title('Backazimuth')

        event_ax = fig.add_subplot(gs[0,1:])

        event_ax.spines['top'].set_visible(False)
        event_ax.spines['left'].set_visible(False)
        event_ax.spines['right'].set_visible(False)
        event_ax.set_title('Event Times')

        daily_ax = fig.add_subplot(gs[1,1])

        daily_ax.spines['top'].set_visible(False)
        daily_ax.spines['left'].set_visible(False)
        daily_ax.spines['right'].set_visible(False)
        daily_ax.set_title('Diurnal Wrapping')

        tidal_ax = fig.add_subplot(gs[1,2])

        tidal_ax.spines['top'].set_visible(False)
        tidal_ax.spines['left'].set_visible(False)
        tidal_ax.spines['right'].set_visible(False)
        tidal_ax.set_title('Tidal Phase Wrapping')

        slow_ax = fig.add_subplot(gs[2,1:])

        slow_ax.spines['top'].set_visible(False)
        slow_ax.spines['left'].set_visible(False)
        slow_ax.spines['right'].set_visible(False)
        slow_ax.set_title('Slowness')

        fig.suptitle('Group ' + group + ' Summary')

        left = mcolors.to_rgb('white')
        middle = mcolors.to_rgb(c[ii])
        right = mcolors.to_rgb('black')

        custom_cm = LinearSegmentedColormap.from_list('custom',[left,middle,right])

        xcorr_mat = np.zeros((len(templates),len(templates)))
        shift_mat = np.zeros((len(templates),len(templates)))

        for i, event_id_1 in enumerate(templates.values()):
            event_1 = event_cat.select_event(event_id_1)
            event_1.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
            event_1.filter('highpass',freq=1)

            for j, event_id_2 in enumerate(templates.values()):
                event_2 = event_cat.select_event(event_id_2)
                event_2.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
                event_2.filter('highpass',freq=1)

                shift, xc = ec.cross_correlate(event_1,event_2)
                xcorr_mat[i,j] = xc
                shift_mat[i,j] = shift

        i = 0
        for name, template in templates.items():
            frac = (i+2) / (len(templates)+2)

            event = event_cat.select_event(template)
            
            event.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
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

            waveform_ax.plot(tr.times(),tr.data/np.abs(tr.data).max()*4 + offset,lw=0.8,color=custom_cm(frac))

            i += 1


        chunk = do.SeismicChunk(starttime,endtime,time_offset=7)


        """BINNING PARAMETERS"""
        daily_bins = 60*60 
        total_bins = 4*60*60 #hour long bins for entire season
        tide_bins = 20 #width in degrees of bins


        """BIN CONSTRUCTION"""
        total_bin_edges = np.mgrid[chunk.starttime:chunk.endtime:total_bins]
        total_bin_centres = total_bin_edges + total_bins / 2
        total_bin_edges_mt = np.array([time.matplotlib_date for time in total_bin_edges])

        day_start = UTCDateTime(2018,1,1)
        day_end = UTCDateTime(2018,1,2)

        daily_bin_edges = np.mgrid[day_start:day_end:daily_bins]
        daily_bin_centres = daily_bin_edges + daily_bins / 2
        daily_bin_edges_mt = np.array([time.matplotlib_date for time in daily_bin_edges])

        tide_bin_edges = np.mgrid[-180:180:tide_bins]
        tide_bin_centres = tide_bin_edges + tide_bins / 2

        """ENVIRONMENTAL DATA"""
        chunk.context('timeseries')
        chunk.attach_environmental(env_path)
        temp = chunk.env_stream.select(channel='TEM')[0]
        tide = chunk.env_stream.select(channel='TID')[0]
        phase = chunk.env_stream.select(channel='PHA')[0]


        #GET EVENT TIMES IN REQUIRED FORMAT FOR BINNING (TIME OF YEAR, TIME OF DAY, TIDAL PHASE)

        groups = [str(i) for i in range(1,9)]

        total_binned = {}
        daily_binned = {}
        tidal_binned = {}

        for temp_name, temp_id in templates.items():
            group_cat = event_cat.events[event_cat.events.index.isin(xcorr[xcorr['template']==temp_name].index)]
            group_times = group_cat['ref_time'].to_numpy()
            group_times = np.array([UTCDateTime(time) for time in group_times],dtype=UTCDateTime)

            event_times_mt = [time.matplotlib_date for time in group_times]
            daily_event_times_mt = [UTCDateTime(2018,1,1,time.hour,time.minute,time.second).matplotlib_date for time in group_times]
            dt = (group_times - tide.stats.starttime).astype(np.float64)
            event_phase = np.interp(dt,phase.times(),phase.data)

            total_binned[temp_name] = np.bincount(np.digitize(event_times_mt,total_bin_edges_mt)-1,minlength=total_bin_edges_mt.size)

            daily_binned[temp_name] = np.bincount(np.digitize(daily_event_times_mt,daily_bin_edges_mt)-1,minlength=daily_bin_edges_mt.size)
            
            tidal_binned[temp_name] = np.bincount(np.digitize(event_phase,tide_bin_edges)-1,minlength=tide_bin_edges.size)


        t_mt = [time.matplotlib_date for time in total_bin_centres]
        day_mt = [time.matplotlib_date for time in daily_bin_centres]

        total_cumulative = np.zeros_like(total_bin_centres)
        daily_cumulative = np.zeros_like(daily_bin_centres)
        tidal_cumulative = np.zeros_like(tide_bin_centres)


        for i, temp_name in enumerate(templates):
            frac = (i+2) / (len(templates)+2)
            event_ax.bar(t_mt,height=total_binned[temp_name],width=t_mt[1]-t_mt[0],bottom=total_cumulative,align='center',color=custom_cm(frac))
            total_cumulative += total_binned[temp_name]

            daily_ax.bar(day_mt,height=daily_binned[temp_name],width=day_mt[1] - day_mt[0],bottom=daily_cumulative,align='center',color=custom_cm(frac))
            daily_cumulative += daily_binned[temp_name]

            tidal_ax.bar(tide_bin_centres,height=tidal_binned[temp_name],width=tide_bin_centres[1]-tide_bin_centres[0],bottom=tidal_cumulative,align='center',color=custom_cm(frac))
            tidal_cumulative += tidal_binned[temp_name]

        myFmt = DateFormatter("%H:%M")
        daily_ax.xaxis.set_major_formatter(myFmt)
        daily_ax.xaxis.set_major_locator(HourLocator(interval=8))

        _set_xaxis_obspy_dates(event_ax)

        num_bins = 16
        slow_bins = 20

        bin_edges = np.linspace(0,2*np.pi,num_bins+1)
        width = bin_edges[1] - bin_edges[0]

        slow_edges = np.linspace(0,1,slow_bins+1)
        slow_width = slow_edges[1] - slow_edges[0]
        slow_ind = np.arange(slow_bins)

        bin_ind = np.arange(num_bins)
        mfp['baz_bin'] = pd.cut(mfp['baz'],bin_edges,labels=bin_ind) #bin the attribute as a new column in the dataframe
        mfp['slow_bin'] = pd.cut(mfp['slow'],slow_edges,labels=slow_ind)


        cumulative = np.zeros_like(bin_ind)
        slow_cumulative = np.zeros_like(slow_ind)
        i = 0


        for i, temp_name in enumerate(templates):
            
            frac = (i+2) / (len(templates)+2)

            group_mfp = mfp[mfp.index.isin(xcorr[xcorr['template']==temp_name].index)]

            counts = group_mfp['baz_bin'].value_counts().to_dict()
            slow_count = group_mfp['slow_bin'].value_counts().to_dict()

            for key in bin_ind:
                counts.setdefault(key,0)
            
            for key in slow_ind:
                slow_count.setdefault(key,0)

            values = np.array([counts[i] for i in range(len(bin_ind))])
            slow_values = np.array([slow_count[i] for i in range(len(slow_ind))])
            rose_ax.bar(bin_edges[:-1],values,width=width,bottom=cumulative,align='edge',color=custom_cm(frac))
            slow_ax.bar(slow_edges[:-1],slow_values,width=slow_width,bottom=slow_cumulative,align='edge',color=custom_cm(frac))

            cumulative += values
            slow_cumulative += slow_values

        fig.savefig(p,format='pdf')

        #now for the second page, do the polarisation analysis
        
        # fig = plt.figure(figsize=(8,6),constrained_layout=True)

        # gs = fig.add_gridspec(ncols=2,nrows=len(inv))

        # k = 0
        # for net in inv:
        #     for sta in net:

        #         rose_ax = fig.add_subplot(gs[k,0],projection='polar')

        #         rose_ax.set_theta_zero_location('N',offset=-long)
        #         rose_ax.set_theta_direction(-1)
        #         rose_ax.set_xticks([])
        #         rose_ax.set_yticks([])
        #         rose_ax.spines['polar'].set_visible(False)
        #         rose_ax.set_title(sta.code)

        #         corr_ax = fig.add_subplot(gs[k,1])

        #         corr_ax.spines['top'].set_visible(False)
        #         corr_ax.spines['left'].set_visible(False)
        #         corr_ax.spines['right'].set_visible(False)

        #         rose_bins = 16
        #         corr_bins = 20

        #         rose_edges = np.linspace(0,2*np.pi,rose_bins+1)
        #         rose_width = rose_edges[1] - rose_edges[0]
        #         rose_ind = np.arange(rose_bins)

        #         corr_edges = np.linspace(0,1,corr_bins+1)
        #         corr_width = corr_edges[1] - corr_edges[0]
        #         corr_ind = np.arange(corr_bins)

        #         polarisation['binned_baz_r_' + sta.code] = pd.cut(polarisation['baz_r_'+sta.code],rose_edges,labels=rose_ind) #bin the attribute as a new column in the dataframe
        #         polarisation['binned_corr_r_' + sta.code] = pd.cut(polarisation['corr_r_'+sta.code],corr_edges,labels=corr_ind)

        #         rose_cumulative = np.zeros_like(rose_ind)
        #         corr_cumulative = np.zeros_like(corr_ind)

        #         for i, temp_name in enumerate(templates):
        #             frac = (i+1) / (len(templates)+2)

        #             group_pol = polarisation[polarisation['event_id'].isin(xcorr[xcorr['template']==temp_name]['event_id'])]

        #             #group_pol = group_pol[(group_pol['corr_r_'+sta.code] > 0.7)]

        #             rose_count = group_pol['binned_baz_r_'+sta.code].value_counts().to_dict()
        #             corr_count = group_pol['binned_corr_r_'+sta.code].value_counts().to_dict()

        #             for key in rose_ind:
        #                 rose_count.setdefault(key,0)
                    
        #             for key in corr_ind:
        #                 corr_count.setdefault(key,0)

        #             rose_values = np.array([rose_count[i] for i in range(len(rose_ind))])
        #             corr_values = np.array([corr_count[i] for i in range(len(corr_ind))])
        #             rose_ax.bar(rose_edges[:-1],rose_values,width=rose_width,bottom=rose_cumulative,align='edge',color=custom_cm(frac))
        #             corr_ax.bar(corr_edges[:-1],corr_values,width=corr_width,bottom=corr_cumulative,align='edge',color=custom_cm(frac))

        #             rose_cumulative += rose_values
        #             corr_cumulative += corr_values

        #         k += 1

        fig.savefig(p,format='pdf')
        p.close()




"""
DENDROGRAM PLOTTING
"""

if dendrogram:

    inner_cat = do.EventCatalogue(starttime,endtime,c_path)
    outer_cat = do.EventCatalogue(starttime,endtime,c_path)

    inner_cat.add_classification(class_cat)
    outer_cat.add_classification(class_cat)

    inner_cat_groups = inner_cat.group_split()
    outer_cat_groups = outer_cat.group_split()

    for group in range(1,9):

        i_cat = inner_cat_groups[group]
        o_cat = outer_cat_groups[group]

        cc_mat = np.zeros((i_cat.N,o_cat.N))
        shift_mat = np.zeros((i_cat.N,o_cat.N))

        for i, outer in enumerate(o_cat):
            outer.attach_waveforms(inv,w_path,buffer=2,length=4)
            outer.filter('highpass',freq=1)
            for j, inner in enumerate(i_cat):
                if i == j:
                    cc_mat[i,j] = 1.0
                    shift_mat[i,j] = 0
                elif i < j:
                    inner.attach_waveforms(inv,w_path,buffer=2,length=4)
                    inner.filter('highpass',freq=1)
                    shift, xcorr = ec.cross_correlate(outer,inner)
                    cc_mat[i,j] = xcorr
                    cc_mat[j,i] = xcorr
                    shift_mat[i,j] = shift
                    shift_mat[j,i] = -shift

        dissimilarity = 1-cc_mat
        dissimilarity = distance.squareform(dissimilarity)

        threshold = 0.4 #TODO make this variable for the different groups - should be higher for some of the emergent groups to get splitting.
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

            max_energy = 0.0
            for tr in event.get_data_window().select(component='Z'):
                energy = np.sum(tr.data**2)
                if energy > max_energy:
                    max_energy = energy
                    max_tr = tr
            
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


if plot_templates:
    from matplotlib import rc
    import matplotlib.font_manager as fm
    rc('text', usetex=True)
    rc('font', size=8)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Optima']})


    
    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(one_col,one_col*1.5),gridspec_kw={'hspace':0.3})
    axes = axes.flatten(order='F')
    for ii, group in enumerate(threshold_dict):
        ax = axes[ii]
        ax.set_axis_off()

        ax.annotate(group,(0,1),xycoords='axes fraction',weight='black')
        ax.annotate('threshold: ' + str(threshold_dict[group]),(1,0),xycoords='axes fraction',ha='right',va='top')

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
            event_1.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
            event_1.filter('highpass',freq=1)

            for j, event_id_2 in enumerate(templates.values()):
                event_2 = event_cat.select_event(event_id_2)
                event_2.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
                event_2.filter('highpass',freq=1)

                shift, xc = ec.cross_correlate(event_1,event_2)
                xcorr_mat[i,j] = xc
                shift_mat[i,j] = shift

        i = 0
        for name, template in templates.items():
            frac = (i+1) / (len(templates)+2)

            event = event_cat.select_event(template)
            
            event.attach_waveforms(inv,w_path,buffer=2,length=4,extra=10)
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

            i += 1


    fig.savefig(os.path.join(plot_path,'templates.pdf'),format='pdf')
    fig.savefig(os.path.join(plot_path,'templates.eps'),bbox_inches='tight')



if plot_cycles:
    #dictionary of the template names and associated event ids.
    import glob

    xcorr_files = glob.glob(os.path.join(c_path,'xcorr*'))
    xcorr_files.sort()
    xcorr = pd.concat([pd.read_csv(file,index_col=0) for file in xcorr_files],ignore_index=False)


    day_fig, day_ax = plt.subplots(nrows=4,ncols=2,figsize=(one_col,one_col),sharex=True)
    tide_fig, tide_ax = plt.subplots(nrows=4,ncols=2,figsize=(one_col,one_col),sharex=True)

    day_ax = day_ax.flatten(order='F')
    tide_ax = tide_ax.flatten(order='F')

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

    #now loop through the groups.
    for ii, group in enumerate(threshold_dict):
        sublist = []
        for key in all_templates:
            if key[:-1] == group:
                sublist.append(key)

        #make a pdf file to save the group summaries to
        templates = {k:all_templates[k] for k in sublist}

        tide_ax[ii].spines['top'].set_visible(False)
        tide_ax[ii].spines['left'].set_visible(False)
        tide_ax[ii].spines['right'].set_visible(False)
        
        day_ax[ii].spines['top'].set_visible(False)
        day_ax[ii].spines['left'].set_visible(False)
        day_ax[ii].spines['right'].set_visible(False)

        tide_ax[ii].set_yticks([])
        day_ax[ii].set_yticks([])
        
        left = mcolors.to_rgb('white')
        middle = mcolors.to_rgb(c[ii])
        right = mcolors.to_rgb('black')

        custom_cm = LinearSegmentedColormap.from_list('custom',[left,middle,right])

        chunk = do.SeismicChunk(starttime,endtime,time_offset=7)


        """BINNING PARAMETERS"""
        daily_bins = 60*60 
        total_bins = 4*60*60 #hour long bins for entire season
        tide_bins = 15 #width in degrees of bins


        """BIN CONSTRUCTION"""
        total_bin_edges = np.mgrid[chunk.starttime:chunk.endtime:total_bins]
        total_bin_centres = total_bin_edges + total_bins / 2
        total_bin_edges_mt = np.array([time.matplotlib_date for time in total_bin_edges])

        day_start = UTCDateTime(2018,1,1)
        day_end = UTCDateTime(2018,1,2)

        daily_bin_edges = np.mgrid[day_start:day_end:daily_bins]
        daily_bin_centres = daily_bin_edges + daily_bins / 2
        daily_bin_edges_mt = np.array([time.matplotlib_date for time in daily_bin_edges])

        tide_bin_edges = np.mgrid[-180:180:tide_bins]
        tide_bin_centres = tide_bin_edges + tide_bins / 2

        """ENVIRONMENTAL DATA"""
        chunk.context('timeseries')
        chunk.attach_environmental(env_path)
        temp = chunk.env_stream.select(channel='TEM')[0]
        tide = chunk.env_stream.select(channel='TID')[0]
        phase = chunk.env_stream.select(channel='PHA')[0]


        #GET EVENT TIMES IN REQUIRED FORMAT FOR BINNING (TIME OF YEAR, TIME OF DAY, TIDAL PHASE)

        #groups = [str(i) for i in range(1,9)]

        daily_binned = {}
        tidal_binned = {}

        for temp_name, temp_id in templates.items():
            group_cat = event_cat.events[event_cat.events.index.isin(xcorr[xcorr['template']==temp_name].index)]
            group_times = group_cat['ref_time'].to_numpy()
            group_times = np.array([UTCDateTime(time) for time in group_times],dtype=UTCDateTime)

            event_times_mt = [time.matplotlib_date for time in group_times]
            daily_event_times_mt = [UTCDateTime(2018,1,1,time.hour,time.minute,time.second).matplotlib_date for time in group_times]
            dt = (group_times - tide.stats.starttime).astype(np.float64)
            event_phase = np.interp(dt,phase.times(),phase.data)

            daily_binned[temp_name] = np.bincount(np.digitize(daily_event_times_mt,daily_bin_edges_mt)-1,minlength=daily_bin_edges_mt.size)
            tidal_binned[temp_name] = np.bincount(np.digitize(event_phase,tide_bin_edges)-1,minlength=tide_bin_edges.size)


        day_mt = [time.matplotlib_date for time in daily_bin_centres]

        daily_cumulative = np.zeros_like(daily_bin_centres)
        tidal_cumulative = np.zeros_like(tide_bin_centres)


        for i, temp_name in enumerate(templates):
            frac = (i+2) / (len(templates)+2)

            day_ax[ii].bar(day_mt,height=daily_binned[temp_name],width=day_mt[1] - day_mt[0],bottom=daily_cumulative,align='center',color=custom_cm(frac))
            daily_cumulative += daily_binned[temp_name]

            tide_ax[ii].bar(tide_bin_centres,height=tidal_binned[temp_name],width=tide_bin_centres[1]-tide_bin_centres[0],bottom=tidal_cumulative,align='center',color=custom_cm(frac))
            tidal_cumulative += tidal_binned[temp_name]

        myFmt = DateFormatter("%H:%M")
        day_ax[ii].xaxis.set_major_formatter(myFmt)
        day_ax[ii].xaxis.set_major_locator(HourLocator(interval=8))


    day_fig.savefig(os.path.join(plot_path,'day_groups.pdf'),format='pdf')
    tide_fig.savefig(os.path.join(plot_path,'tide_groups.pdf'),format='pdf')