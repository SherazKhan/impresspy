import mne
from mne.chpi import read_head_pos,filter_chpi
import subprocess
from .utils import silentremove, detectBadChannels
from mne.preprocessing import maxwell_filter
import warnings
import os
data_path = '../data'
ctc_fname = data_path + '/ct_sparse.fif'
fine_cal_fname = data_path + '/sss_cal.dat'

class maxfilter:

    def __init__(self, filename, host='dodeca',destination=None):
        self.filename = filename
        self.posfile = filename[:-4] + '.pos'
        self.fname_out = filename[:-4] + '_temp.fif'
        self.fname_sss = filename[:-4] + '_sss.fif'
        self.host=host
        self.destination = destination
        self.ctc_fname = os.path.abspath(ctc_fname)
        self.fine_cal_fname = os.path.abspath(fine_cal_fname)
        self.raw = mne.io.read_raw_fif(self.filename)
        self.maxfilter_applied = False
        self.head_pos = None

    def get_headposition(self):
        COMMAND = ["ssh", "%s" % self.host, 'maxfilter', '-f', self.filename, '-hp', self.posfile, '-headpos', '-o', self.fname_out,
                   '-force']

        p = subprocess.Popen(COMMAND,
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (output, err) = p.communicate()

        if output.split()[-6] == 'successfully':
            head_pos = read_head_pos(self.posfile)
            head_pos[:,0] += self.raw.first_samp/self.raw.info['sfreq']
            silentremove(self.fname_out)
            self.head_pos = head_pos
            return self.head_pos
        else:
            print err
            return None

    def get_headposition_mne(self,t_step_min=0.1, t_step_max=10.,
                              t_window=0.2, dist_limit=0.005, gof_limit=0.98,
                              verbose=None):

        try:
            head_pos = mne.chpi._calculate_chpi_positions(self.raw,t_step_min=t_step_min, t_step_max=t_step_max,
                              t_window=t_window, dist_limit=dist_limit, gof_limit=gof_limit,
                              verbose=verbose)

            head_pos[:, 0] += self.raw.first_samp / self.raw.info['sfreq']
            self.head_pos = head_pos
            return self.head_pos
        except:
            print('CHPI error')
            return None



    def apply_maxfilter(self,st_duration=None,st_correlation=0.98):


        if not self.raw.info['bads']:
            warnings.warn("Detecting bad Channel, Please check manually")
            self.detect_bad_channel()
        if self.head_pos is None:
            self.get_headposition()
        print('Applying maxfilter; This will take time')
        if self.head_pos is not None:
            self.raw_sss = maxwell_filter(self.raw, cross_talk=self.ctc_fname,
                                          calibration=self.fine_cal_fname,head_pos=self.head_pos,
                                          st_duration=st_duration,st_correlation=st_correlation)
        else:
            self.raw_sss = maxwell_filter(self.raw, cross_talk=ctc_fname,
                                          calibration=fine_cal_fname,st_duration=st_duration,
                                          st_correlation=st_correlation)

        self.maxfilter_applied = True
        self.raw_sss.save(self.fname_sss,overwrite=True)
        print('maxfilter done, file saved as {}'.format(self.fname_sss))


    def set_bad_channel(self,badchannels):
        self.raw.info['bads'] += badchannels


    def detect_bad_channel(self,zscore=3):
        badchannels = detectBadChannels(self.raw,zscore=zscore)
        self.set_bad_channel(badchannels)


    def apply_max_move(self,dest_filename):
        self.raw_sss = maxwell_filter(self.raw,
                                      cross_talk=self.ctc_fname, calibration=self.fine_cal_fname,destination=dest_filename)

    def apply_maxfilter_emptyroom(self, st_duration=None):
        if not self.raw.info['bads']:
            warnings.warn("Detecting bad Channel, Please check manually")
            self.detect_bad_channel()
        self.raw_sss = maxwell_filter(self.raw, coord_frame='meg',
                                 regularize=None, cross_talk=self.ctc_fname, calibration=self.fine_cal_fname,origin=(0.,0.013,-0.006))
        self.raw_sss.save(self.fname_sss)



