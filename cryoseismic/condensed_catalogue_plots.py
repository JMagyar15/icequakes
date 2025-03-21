from obspy.core import UTCDateTime
from iqvis import data_objects as do
from obspy.core.inventory import inventory
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import pandas as pd
from obspy.imaging.util import _set_xaxis_obspy_dates

from matplotlib import rc
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages


"""
SWITCHES
"""
plot_spec = True
plot_att = False

rc('text', usetex=True)
rc('font', size=8)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Optima']})

one_col = 3.35 #inches
two_col = 7.0 #inches


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15) #for now, just test with a couple of days to check all is working
ref_time = UTCDateTime(2018,1,10) #for finding the response information

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
spec_path = '/Users/jaredmagyar/Documents/SorsdalData/up_spectrograms'
env_path = '/Users/jaredmagyar/Documents/SorsdalData/environmental'
outplots = '/Users/jaredmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes/catalogue_summaries' #TODO want to change this to iCloud when I start doing plots in here...



for path in [c_path,spec_path,outplots]:
    if not os.path.exists(path):
        os.mkdir(path)


chunk = do.SeismicChunk(t1,t2,time_offset=7) #local time is UTC + 07:00

avail = pd.read_csv('stream_availability.csv')
event_cat = do.EventCatalogue(t1,t2,c_path) #get the event catalogue

inv = inventory.read_inventory(stat_path,level='response')


filename = os.path.join(outplots,'condensed_catalogues.pdf')
p = PdfPages(filename) #set up the object used to save each channel as page in a pdf

for code in inv.get_contents()['channels']:
    print('Plotting',code)
    """
    PLOTTING OF THE BINNED SPECTROGRAMS
    """
    full = np.load(os.path.join(spec_path,'full_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
    daily = np.load(os.path.join(spec_path,'diurnal_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
    tidal = np.load(os.path.join(spec_path,'tide_phase_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
    
    resp = inv.get_response(code,ref_time)
    sens = abs(resp.get_evalresp_response_for_frequencies(full['f'][1:]))

    """
    Full chunk plotting
    """

    f_band = full['f'][(full['f']>3)&(full['f']<15)]
    spec_band = full['spec'][(full['f']>3)&(full['f']<15),:]
    med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

    fig = plt.figure(figsize=(two_col,two_col/1.2))

    #make subfigures for total and wrapped catalogues

    cat_fig, wrapped_fig = fig.subfigures(nrows=2)
    cat_fig.suptitle(code,fontsize=16)



    cat_grid_spec = cat_fig.add_gridspec(nrows=4,ncols=2,width_ratios=[1,0.02],height_ratios=[1,0.6,0.4,0.2],wspace=0.02,hspace=0.05)

    start_mt = chunk.starttime.matplotlib_date
    end_mt = chunk.endtime.matplotlib_date

    #firstly deal with the spectrogram + colourbar in the top row.
    spec_ax = cat_fig.add_subplot(cat_grid_spec[0,0])
    spec_cb_ax = cat_fig.add_subplot(cat_grid_spec[0,1])

    t_mt = [time.matplotlib_date for time in full['t']]

    db_spec = 10 * np.log10(full['spec'][1:,:]) - 20 * np.log10(sens)[:,None]

    spec_plot = spec_ax.pcolormesh(t_mt,full['f'][1:],db_spec,cmap='PuBu',vmax=-120,vmin=-200)
    spec_ax.hlines([3,15],xmin=t_mt[0],xmax=t_mt[-1],color='black',ls='--',lw=0.8)
    cb = plt.colorbar(spec_plot,cax=spec_cb_ax,label=r'Amplitude ($m^2/s^2/Hz$) (dB)')

    spec_ax.set_yscale('log')
    spec_ax.set_ylabel('Frequency (Hz)')

    #environmental data and band-integrated median seismic amplitude
    time_series_ax = cat_fig.add_subplot(cat_grid_spec[1,0],sharex=spec_ax)

    tide_ax = plt.twinx(time_series_ax)
    time_series_ax.hlines(0,xmin=start_mt,xmax=end_mt,color='grey',ls='--',lw=0.5)

    tide_ax.plot(t_mt,full['tide'],c='cornflowerblue',lw=0.8)
    tide_ax.set_ylabel('Tide Height (m)')

    time_series_ax.plot(t_mt,full['temp'],c='indianred',lw=0.8)
    time_series_ax.set_ylabel(r'Temperature ($^\circ$C)')

    #histogram of event detections
    hist_ax = cat_fig.add_subplot(cat_grid_spec[2,0],sharex=spec_ax)


    hist_ax.bar(t_mt,height=full['events'],width=t_mt[1]-t_mt[0],bottom=0,align='center',color='grey',alpha=0.5)
    hist_ax.set_ylabel('Event Count')

    amp_ax = plt.twinx(hist_ax)
    amp_ax.plot(t_mt,med_amp,color='black',lw=0.8)
    amp_ax.set_ylim(bottom=0)


    amp_ax.set_ylabel('Amplitude (m/s)')

    #data availability for the broadband stations
    avail_ax = cat_fig.add_subplot(cat_grid_spec[3,0],sharex=spec_ax)

    i = 0
    labels = []
    coords = []

    station_code = code.split('.')[1][0:3]
    if station_code == 'BBS':
        for net in inv.select(station='BBS??',channel='??Z'):
            for sta in net:
                for ch in sta:
                    i -= 1
                    subset = avail[(avail['Network']==net.code) & (avail['Station']==sta.code) & (avail['Channel']==ch.code)]
                    start = np.array([UTCDateTime(time).matplotlib_date for time in subset['Start']])#pd.to_datetime(subset['Start'])# + pd.Timedelta(hours=chunk.time_offset)
                    end = np.array([UTCDateTime(time).matplotlib_date for time in subset['End']])#pd.to_datetime(subset['End']) #+ pd.Timedelta(hours=chunk.time_offset)
                    duration = end - start

                    
                    if sta.code == code.split('.')[1]:
                        color = 'navy'
                    else:
                        color = 'black'
                    avail_ax.barh(i,duration,left=start,color=color)
                    coords.append(i)
                    labels.append(sta.code)

    if station_code == 'REF':
        for net in inv.select(station='REF??',channel='??Z'):
            for sta in net:
                for ch in sta:
                    i -= 1
                    subset = avail[(avail['Network']==net.code) & (avail['Station']==sta.code) & (avail['Channel']==ch.code)]
                    start = np.array([UTCDateTime(time).matplotlib_date for time in subset['Start']])#pd.to_datetime(subset['Start'])# + pd.Timedelta(hours=chunk.time_offset)
                    end = np.array([UTCDateTime(time).matplotlib_date for time in subset['End']])#pd.to_datetime(subset['End']) #+ pd.Timedelta(hours=chunk.time_offset)
                    duration = end - start

                    if sta.code == code.split('.')[1]:
                        color = 'navy'
                    else:
                        color = 'black'
                    avail_ax.barh(i,duration,left=start,color=color)
                    coords.append(i)
                    labels.append(sta.code)

    avail_ax.set_yticks([])
    #avail_ax.set_yticks(coords)
    #avail_ax.set_yticklabels(labels)
    avail_ax.spines['right'].set_visible(False)
    avail_ax.spines['left'].set_visible(False)
    avail_ax.spines['top'].set_visible(False)
    avail_ax.spines['bottom'].set_visible(False)

    avail_ax.set_xlim((start_mt,end_mt))
    _set_xaxis_obspy_dates(avail_ax)
    plt.setp(spec_ax.get_xticklabels(), visible=False)
    plt.setp(time_series_ax.get_xticklabels(), visible=False)
    plt.setp(hist_ax.get_xticklabels(), visible=False)

    sign = '%+i' % chunk.time_offset
    sign = sign[0]
    label = "UTC (Local Time = UTC %s %02i:%02i)" % (sign, abs(chunk.time_offset),(chunk.time_offset % 1 * 60))

    avail_ax.set_xlabel(label)

    """
    DIURNALLY WRAPPED PLOTS
    """
    
    f_band = daily['f'][(daily['f']>3)&(daily['f']<15)]
    spec_band = daily['spec'][(daily['f']>3)&(daily['f']<15),:]
    med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

    wrap_grid_spec = wrapped_fig.add_gridspec(nrows=3,ncols=2,width_ratios=[1,1],height_ratios=[1,0.5,0.5],wspace=0.2,hspace=0.05)

    day_spec_ax = wrapped_fig.add_subplot(wrap_grid_spec[0,0])

    t_mt = [time.matplotlib_date for time in daily['t']]

    db_spec = 10 * np.log10(daily['spec'][1:,:]) - 20 * np.log10(sens)[:,None]

    spec_plot = day_spec_ax.pcolormesh(t_mt,daily['f'][1:],db_spec,cmap='PuBu',vmin=-200,vmax=-120)
    day_spec_ax.hlines([3,15],xmin=t_mt[0],xmax=t_mt[-1],color='black',ls='--',lw=0.8)

    day_spec_ax.set_yscale('log')
    day_spec_ax.set_ylabel('Frequency (Hz)')

    #environmental data and band-integrated median seismic amplitude
    day_time_series_ax = wrapped_fig.add_subplot(wrap_grid_spec[1,0],sharex=day_spec_ax)


    day_time_series_ax.hlines(0,xmin=t_mt[0],xmax=t_mt[-1],color='grey',ls='--')

    day_time_series_ax.plot(t_mt,daily['temp'],c='indianred',lw=0.8)
    day_time_series_ax.set_ylabel(r'Temperature ($^\circ$C)')


    #histogram of event detections
    day_hist_ax = wrapped_fig.add_subplot(wrap_grid_spec[2,0],sharex=day_spec_ax)

    day_hist_ax.bar(t_mt,height=daily['events'],width=t_mt[1]-t_mt[0],bottom=0,align='center',color='grey',alpha=0.5)
    day_hist_ax.set_ylabel('Event Count')

    day_amp_ax = plt.twinx(day_hist_ax)
    day_amp_ax.plot(t_mt,med_amp,color='black',lw=0.8)
    day_amp_ax.set_ylabel('Amplitude (m/s)')
    day_amp_ax.set_ylim(bottom=0)


    plt.setp(day_spec_ax.get_xticklabels(), visible=False)
    plt.setp(day_time_series_ax.get_xticklabels(), visible=False)

    from matplotlib.dates import DateFormatter, HourLocator

    myFmt = DateFormatter("%H:%M:%S")
    day_hist_ax.xaxis.set_major_formatter(myFmt)
    day_hist_ax.xaxis.set_major_locator(HourLocator(interval=8))


    sign = '%+i' % chunk.time_offset
    sign = sign[0]
    label = "UTC (Local Time = UTC %s %02i:%02i)" % (sign, abs(chunk.time_offset),(chunk.time_offset % 1 * 60))

    day_hist_ax.set_xlabel(label)

    """
    TIDALLY WRAPPED PLOTS
    """

    f_band = tidal['f'][(tidal['f']>3)&(tidal['f']<15)]
    spec_band = tidal['spec'][(tidal['f']>3)&(tidal['f']<15),:]
    med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

    #firstly deal with the spectrogram + colourbar in the top row.
    tide_spec_ax = wrapped_fig.add_subplot(wrap_grid_spec[0,1])

    db_spec = 10 * np.log10(tidal['spec'][1:,:]) - 20 * np.log10(sens)[:,None]

    tide_spec_plot = tide_spec_ax.pcolormesh(tidal['phase'],tidal['f'][1:],db_spec,cmap='PuBu',vmin=-200,vmax=-120)
    tide_spec_ax.hlines([1.5,10],xmin=tidal['phase'][0],xmax=tidal['phase'][-1],color='black',ls='--',lw=0.8)

    tide_spec_ax.set_yscale('log')
    #tide_spec_ax.set_ylabel('Frequency (Hz)')

    #environmental data and band-integrated median seismic amplitude
    tide_time_series_ax = wrapped_fig.add_subplot(wrap_grid_spec[1,1],sharex=tide_spec_ax)

    tide_time_series_ax.plot(tidal['phase'],tidal['tide'],c='cornflowerblue',lw=0.8)
    tide_time_series_ax.set_ylabel(r'Tide Height (m)')


    #histogram of event detections
    tide_hist_ax = wrapped_fig.add_subplot(wrap_grid_spec[2,1],sharex=tide_spec_ax)
    tide_hist_ax.bar(tidal['phase'],height=tidal['events'],width=tidal['phase'][1]-tidal['phase'][0],bottom=0,align='center',color='grey',alpha=0.5)
    #tide_hist_ax.set_ylabel('Event Count')


    tide_amp_ax = plt.twinx(tide_hist_ax)
    tide_amp_ax.plot(tidal['phase'],med_amp,color='black',lw=0.8)
    #tide_amp_ax.set_ylabel('Amplitude (m/s)')
    tide_amp_ax.set_ylim(bottom=0)


    plt.setp(tide_spec_ax.get_xticklabels(), visible=False)
    plt.setp(tide_time_series_ax.get_xticklabels(), visible=False)

    fig.savefig(p,format='pdf')

p.close()
