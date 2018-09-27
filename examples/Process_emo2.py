import mne
import os
import os.path as op
import pyimpress as pyi
import numpy as np
from mne.minimum_norm import apply_inverse
import matplotlib.pylab as plt
import pandas as pd

mne.set_log_level('WARNING')
plt.ion()

protocol_path = '/cluster/transcend/emo2/'
erm_protocol_path = '/cluster/transcend/MEG/erm/'
subjects_dir = '/cluster/transcend/MRI/WMA/recons'
csv_fname = op.join(protocol_path,'emo2_info.csv')
info = pd.read_csv(csv_fname)

subject = '011201'
runs = [1, 2, 3]
visit = info.loc[info['subj'].isin(["\'" + subject + "\'"])]['visit'].as_matrix()[0][1:-1]

raws = list()

data_path = op.join(protocol_path,subject,visit)
erm_path = op.join(erm_protocol_path,subject,visit)


for index in runs:
    raw_fname = subject + '_emo2_' + str(index) + '_sss.fif'
    raw_fname = op.join(data_path, raw_fname)
    print("Adding raw files: %s" %raw_fname)
    raws.append(mne.io.read_raw_fif(raw_fname, preload=True))

erm_raw_fname = subject + '_erm_1_sss.fif'
erm_raw_fname = op.join(erm_path, erm_raw_fname)
raw_erm = mne.io.read_raw_fif(erm_raw_fname, preload=True)

trans =  op.join(erm_path, subject + '_1-trans.fif')


raw = mne.concatenate_raws(raws, preload=True)
pyi.utils.compute_ecg_eog_proj(raw)
raw.apply_proj()
raw.filter(.5,35,filter_length='auto',
           l_trans_bandwidth='auto',h_trans_bandwidth='auto')
events = pyi.utils.run_events(raw)[0]
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False)
event_id = dict(angry_faces=1, fearful_faces=2, houses=3,
                neutral_faces=4, flipped_neutral_faces=5)
tmin, tmax, baseline = -.25, 1,(-.25, 0)
reject=dict(grad=4000e-13, eog=350e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=reject,
                    preload=True, add_eeg_ref=False)
epochs.resample(150., npad='auto')
epochs.equalize_event_counts(epochs.event_id)
evokeds = {key:epochs[key].average() for key in event_id.keys()}

contrast = mne.combine_evoked([evokeds['neutral_faces'], - evokeds['houses']], weights='equal')
contrast.plot_joint([.1,.2,.3,.4])

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
reject = dict(grad=4000e-13,  mag=4e-12)
#cov = mne.compute_raw_covariance(raw_erm,reject=reject,picks=picks)

cov =pyi.utils.compute_cov(epochs,reject=reject)

fwd, inv = pyi.utils.run_fwd_inv(raw_fname, subject, cov=cov,
                                 fname_trans=trans,subjects_dir = subjects_dir)

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"
pick_ori = None
stc = apply_inverse(contrast, inv, lambda2, method, pick_ori)
stc.plot(time_viewer=True,hemi='split',
                      views=['lateral','medial'], surface='inflated',initial_time=.2)

stc_faces, stc_houses = (apply_inverse(evokeds['neutral_faces'], inv, lambda2, method, pick_ori),
                         apply_inverse(evokeds['houses'], inv, lambda2, method, pick_ori))

stc_contrast = stc_faces - stc_houses

stc_contrast.plot(time_viewer=True,hemi='split',
                      views=['lateral','medial'], surface='inflated',initial_time=.2)

stc_n = apply_inverse(contrast, inv, lambda2, method, 'normal')
stc_n.plot(time_viewer=True,hemi='split',
                      views=['lateral','medial'], surface='inflated',initial_time=.17)

