import mne
import matplotlib.pylab as plt
import numpy as np
import pyimpress as pyi
import numpy as np
from mne.minimum_norm import apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.time_frequency import tfr_morlet
import os.path as op
plt.ion()
subjects_dir = '/space/megryan/1/users/sheraz/data/lucia/recons'
subject = 'LMV2016-PFOP1'
data_path = '/space/megryan/1/users/sheraz/data/lucia/MEG/LMV2016-PFOP2'
fname_raw =data_path +  '/fix1.fif'
fname_erm= data_path + '/erm.fif'
trans = data_path + '/trans.fif'
runs = [1,2]
raws=[]

#channel_indices = mne.pick_channels_regexp(info['ch_names'], 'MEG *')

for index in runs:
    fname = op.join(data_path, 'grating' + str(index) + '_raw.fif')
    raws.append(mne.io.read_raw_fif(fname, preload=True))

raw = mne.concatenate_raws(raws, preload=True)

raw.filter(1,100,filter_length='auto',
           l_trans_bandwidth='auto',h_trans_bandwidth='auto')


events = pyi.utils.run_events(raw)[0]
bads = pyi.utils.detectBadChannels(raw,zscore=3)
raw.info['bads'] += bads + ['MEG0313', 'MEG0323', 'MEG0543']
pyi.utils.compute_ecg_eog_proj(raw)
raw.apply_proj()


picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)
event_id, tmin, tmax, baseline = dict(grating=1,grating25=2), -.5, 1.5,(-.25, 0)
reject=dict(grad=4900e-11, eog=250e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=reject,
                    preload=True, add_eeg_ref=False)


epochs.resample(450., npad='auto')
epochs.apply_proj()
cov =pyi.utils.compute_cov(epochs,reject=reject)

fwd, inv = pyi.utils.run_fwd_inv(fname_raw, subject, cov=cov,
                                 fname_trans=trans,subjects_dir = subjects_dir)


snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inv, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot(subject, parc='aparc.a2009s',
                                    subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inv['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=True)

# Now we are ready to compute the connectivity in the alpha band. Notice
# from the status messages, how mne-python: 1) reads an epoch from the raw
# file, 2) applies SSP and baseline correction, 3) computes the inverse to
# obtain a source estimate, 4) averages the source estimate to obtain a
# time series for each label, 5) includes the label time series in the
# connectivity computation, and then moves to the next epoch. This
# behaviour is because we are using generators and allows us to
# compute connectivity in computationally efficient manner where the amount
# of memory (RAM) needed is independent from the number of epochs.
fmin = 35
fmax = 70
sfreq = raw.info['sfreq']  # the sampling frequency
con_methods = ['pli', 'wpli2_debiased']
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,tmin=0.05,tmax=1.25,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

# con is a 3D array, get the connectivity for the first (and only) freq. band
# for each method
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c[:, :, 0]

# Now, we visualize the connectivity using a circular graph layout

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig,ax = plot_connectivity_circle(con_res['wpli2_debiased'], label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='All-to-All Connectivity Gamma Gratings '
                               'Condition (w-PLI)')
plt.savefig('circle.png', facecolor='black')

# Plot connectivity for both methods in the same plot
fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')
no_names = [''] * len(label_names)
for ii, method in enumerate(con_methods):
    plot_connectivity_circle(con_res[method], no_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title=method, padding=0, fontsize_colorbar=6,
                             fig=fig, subplot=(1, 2, ii + 1))

plt.show(block=True)


# bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
#                  (12, 30, 'Beta'), (30, 45, 'Gamma'), (47, 80, 'High Gamma')]
# epochs.plot_psd_topomap(ch_type='grad', normalize=True, bands=bands)
power, itc = tfr_morlet(epochs, freqs=np.arange(30,100), n_cycles=7, use_fft=True,
                        return_itc=True, n_jobs=8)
h = power.plot([169], baseline=(-0.2, 0), mode='logratio',tmin=-0.2,tmax=1.2,
               vmin=0.1,vmax=0.4,cmap='jet',colorbar=True)

plt.title('Grating power in V1', fontsize=29)
plt.grid(b=True, which='minor', color='0.65',linestyle='-')
plt.ylabel('Frequency (Hz)',fontsize=24)
plt.xlabel('Time (msec)',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar()