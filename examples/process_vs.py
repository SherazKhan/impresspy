import mne
import pyimpress as pyi
import numpy as np
from mne.decoding import GeneralizationAcrossTime
# from mne.decoding.search_light import _SearchLight, _GeneralizationLight
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
from sklearn.cross_validation import StratifiedKFold
import os.path as op

plt.ion()

#protocol_path = '/autofs/space/megraid_research/MEG/tal/subj_ricardo/170118/'
protocol_path = '/cluster/transcend/MEG/vs/099701/1/'
#subid = 'ricardo'
subid = '099701'
runs = [1,2,3]
raws = []

for index in runs:
    raw_fname = op.join(protocol_path, subid + '_vs_' + str(index) + '_ssss.fif')
    print("Adding " + raw_fname)
    raws.append(mne.io.read_raw_fif(raw_fname, preload=True))

raw = mne.concatenate_raws(raws, preload=True)
# raw_fname = op.join(protocol_path,subid + '_vs_raw.fif')
# raw_fname = op.join(protocol_path,subid + '_vs_2_raw.fif')
# raw = mne.io.read_raw_fif(raw_fname, preload=True)
# raw.crop(0,1046)
# raw.info['bads'] += ['MEG1421','MEG0811', 'MEG1431']

pyi.utils.compute_ecg_eog_proj(raw)
raw.apply_proj()
raw.filter(.5, 40,filter_length='auto',
           l_trans_bandwidth='auto',h_trans_bandwidth='auto')
events = pyi.utils.run_events(raw)[0]
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False)
event_id = dict(np_4=1, np_6=2, np_8=3,
                p_4=4, p_6=5, p_8=6)
tmin, tmax, baseline = -.15, 1,(-.15, 0)
reject=dict(grad=4500e-13, eog=350e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=reject,
                    preload=True)
epochs.resample(150., npad='auto')
epochs.pick_types(meg='grad')
epochs.equalize_event_counts(epochs.event_id)


# folder_path = '/autofs/space/megraid_research/MEG/tal/subj_097201/170711'
# raw = mne.io.read_raw_fif(folder_path+'/097201_vs_2_raw.fif', preload=True)
#
#
# events = mne.find_events(raw, stim_channel='STI101',
#                      consecutive='increasing', mask=255,
#                      mask_type='and',
#                      verbose=True)
# tmin, tmax, baseline = -0.15, 1.0, (-.15, 0)
# event_id = dict(np_4=1, np_6=2, np_8=3, np_10=4,
#                 p_4=5, p_6=6, p_8=7, p_10=8)
# picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False)
# reject=dict(grad=4500e-13, eog=350e-6)
# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
#                     baseline=baseline, reject=reject,
#                     preload=True)
# epochs.resample(150., npad='auto')
#
# gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=12)
# triggers = epochs.events[:, 2]
# np10_vs_p10 = (triggers[np.in1d(triggers, (1, 8))] == 1).astype(int)
# gat.fit(epochs[('np_4', 'p_10')], y=np10_vs_p10)
# gat.score(epochs[('np_4', 'p_10')], y=np10_vs_p10)
# gat.plot_diagonal()



#
#
evokeds = {key:epochs[key].average() for key in event_id.keys()}

np468 = mne.combine_evoked([evokeds['np_4'],evokeds['np_6'],evokeds['np_8']], weights='equal')
f1 = np468.plot_joint([.18,.3,.45,.6],title='Non Pop')
axes = f1.get_axes()
axes[0].set_ylim([-110, 110])


p468 = mne.combine_evoked([evokeds['p_4'],evokeds['p_6'],evokeds['p_8']], weights='equal')
f2 = p468.plot_joint([.18,.3,.45,.6],title='Pop')
axes = f2.get_axes()
axes[0].set_ylim([-110, 110])


triggers = epochs.events[:, 2]
gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=12)
#gat = GeneralizationAcrossTime(predict_mode='mean-prediction', n_jobs=12)


ind = np.in1d(triggers, (4, 5, 6)).astype(int)
gat.fit(epochs[('np_4', 'np_6', 'np_8' ,'p_4', 'p_6', 'p_8')], y=ind)
gat.score(epochs[('np_4', 'np_6', 'np_8' ,'p_4', 'p_6', 'p_8')], y=ind)
gat.plot(vmin=.55,vmax=.7)
gat.plot_diagonal()


###
np8_vs_p4 = (triggers[np.in1d(triggers, (3, 4))] == 4).astype(int)
p8_vs_np4 = (triggers[np.in1d(triggers, (6, 1))] == 1).astype(int)
p8_vs_np8 = (triggers[np.in1d(triggers, (6, 3))] == 3).astype(int)
p6_vs_np6 = (triggers[np.in1d(triggers, (5, 2))] == 2).astype(int)
p4_vs_np4 = (triggers[np.in1d(triggers, (4, 1))] == 1).astype(int)
p8_vs_p6 = (triggers[np.in1d(triggers, (6, 5))] == 5).astype(int)
np8_vs_np6 = (triggers[np.in1d(triggers, (1, 2))] == 2).astype(int)
#
gat.fit(epochs[('p_4', 'np_4')], y=p4_vs_np4)
gat.score(epochs[('p_4', 'np_4')], y=p4_vs_np4)
gat.plot(vmin=.6,vmax=.75,title='p4_vs_np4')
gat.plot_diagonal(title='p4_vs_np4')
plt.ylim(.4,.75)

gat.fit(epochs[('p_6', 'np_6')], y=p6_vs_np6)
gat.score(epochs[('p_6', 'np_6')], y=p6_vs_np6)
gat.plot(vmin=.6,vmax=.75,title='p6_vs_np6')
gat.plot_diagonal(title='p6_vs_np6')
plt.ylim(.4,.75)

gat.fit(epochs[('p_8', 'np_8')], y=p8_vs_np8)
gat.score(epochs[('p_8', 'np_8')], y=p8_vs_np8)
gat.plot(vmin=.6,vmax=.62,title='p8_vs_np8 - p8_vs_np8')
gat.plot_diagonal(title='p8_vs_np8 - p8_vs_np8')
plt.ylim(.4,.75)

##
gat.fit(epochs[('np_8', 'p_4')], y=np8_vs_p4)
gat.score(epochs[('np_8', 'p_4')], y=np8_vs_p4)
gat.plot(vmin=.63,vmax=.73,title='np8_vs_p4')
gat.plot_diagonal(title='np8_vs_p4')
#
# gat.score(epochs[('p_8', 'np_8')], y=p8_vs_np8)
# gat.plot(vmin=.65,vmax=.75,title='np8_vs_p4 - p8_vs_np8')
# gat.plot_diagonal(title='np8_vs_p4 - p8_vs_np8')
#
# gat.score(epochs[('np_8', 'p_4')], y=np8_vs_p4)
# gat.plot(vmin=.65,vmax=.75,title='np8_vs_p4 - np8_vs_p4')
# gat.plot_diagonal(title='np8_vs_p4 - np8_vs_p4')


# ##
# gat.fit(epochs[('p_8', 'np_4')], y=p8_vs_np4)
#
# gat.score(epochs[('p_8', 'np_4')], y=p8_vs_np4)
# gat.plot(vmin=.65,vmax=.75,title='p8_vs_np4 - p8_vs_np4')
# gat.plot_diagonal(title='p8_vs_np4 - p8_vs_np4')
#
# gat.score(epochs[('p_8', 'np_8')], y=p8_vs_np8)
# gat.plot(vmin=.65,vmax=.75,title='p8_vs_np4 - p8_vs_np8')
# gat.plot_diagonal(title='np8_vs_p4 - p8_vs_np8')
#
# gat.score(epochs[('np_8', 'p_4')], y=np8_vs_p4)
# gat.plot(vmin=.65,vmax=.75,title='p8_vs_np4 - np8_vs_p4')
# gat.plot_diagonal(title='np8_vs_p4 - np8_vs_p4')
#


# ##
# gat.fit(epochs[('p_8', 'np_8')], y=p8_vs_np8)
# #
# # gat.score(epochs[('p_8', 'np_4')], y=p8_vs_np4)
# # gat.plot(vmin=.65,vmax=.75,title='p8_vs_np8 - p8_vs_np4')
# # gat.plot_diagonal(title='p8_vs_np8 - p8_vs_np4')
#
# gat.score(epochs[('p_8', 'np_8')], y=p8_vs_np8)
# gat.plot(vmin=.5,vmax=.75,title='p8_vs_np8 - p8_vs_np8')
# gat.plot_diagonal(title='p8_vs_np8 - p8_vs_np8')
#
# gat.score(epochs[('np_8', 'p_4')], y=np8_vs_p4)
# gat.plot(vmin=.65,vmax=.75,title='p8_vs_np8 - np8_vs_p4')
# gat.plot_diagonal(title='p8_vs_np8- np8_vs_p4')

#
#
# ##
# gat.fit(epochs[('p_8', 'p_6')], y=p8_vs_p6)
#
# gat.score(epochs[('p_8', 'p_6')], y=p8_vs_p6)
# gat.plot(vmin=.65,vmax=.75,title='p8_vs_p6 - p8_vs_p6')
# gat.plot_diagonal(title='p8_vs_p6 - p8_vs_p6')
#
#
# ##
# gat.fit(epochs[('np_8', 'np_6')], y=np8_vs_np6)
#
# gat.score(epochs[('np_8', 'np_6')], y=np8_vs_np6)
# gat.plot(vmin=.65,vmax=.75,title='np8_vs_np6 - np8_vs_np6')
# gat.plot_diagonal(title='np8_vs_np6 - np8_vs_np6')