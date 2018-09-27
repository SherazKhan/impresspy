import mne
from nipype.interfaces.freesurfer import ReconAll
import os.path as op
import subprocess
import os
import shutil

def read_dicom_log_file(dicom_log):
    datafile = file(dicom_log)
    found = False  # this isn't really necessary
    for line in datafile:
        if 'MEMPRAGE' in line and 'RMS' in line:
            # found = True #not necessary
            return line.split('\n')[-2].split(' ')[-1]
    return False  # because you finished the search without finding anything

def read_findsession_log_file(findsession_log):
    datafile = file(findsession_log)
    found = False  # this isn't really necessary
    for line in datafile:
        if 'PATH' in line:
            # found = True #not necessary
            return line.split('\n')[-2].split(' ')[-1]
    return False  # because you finished the search without finding anything



def copy_dicom(dicom_path, dicom_dir, symlinks=False, ignore=None):
    for item in os.listdir(dicom_path):
        s = os.path.join(dicom_path, item)
        d = os.path.join(dicom_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def run_findsession(subject, subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    mri_dir = '/'.join(subjects_dir.split('/')[:-1])

    dicom_dir = op.join(mri_dir,'DICOM',subject)
    if not os.path.exists(dicom_dir):
        os.makedirs(dicom_dir)
    findsession_log = op.join(mri_dir,'DICOM',subject,subject+'-findsession.log')

    COMMAND = ['findsession', '-e', subject]

    p = subprocess.Popen(COMMAND,
                         shell=False,
                         stdout=file(findsession_log, "w"),
                         stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    return findsession_log, dicom_dir


def process_subject_bem(subject, subjects_dir='/cluster/transcend/MRI/WMA/recons', spacing='ico4'):
    try:
        bem_fname = op.join(subjects_dir,subject,'bem', '%s-src.fif' % subject)
        src_fname = op.join(subjects_dir, subject, 'bem', '%s-src.fif' % spacing)
        #headsurf_log = op.join(subjects_dir, subject, 'bem', subject + '_headsurf.log')

        if not os.path.isfile(bem_fname):
            mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True, volume='T1', atlas=True,
                                       gcaatlas=False, preflood=None)
            conductivity = (0.3,)
            model = mne.make_bem_model(subject=subject, ico=4,
                                       conductivity=conductivity,
                                       subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(bem_fname, bem=bem)


        if not os.path.isfile(src_fname):
            src = mne.setup_source_space(subject, spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False)
            mne.write_source_spaces(src_fname, src=src, overwrite=True)
    except Exception as ee:
        error = str(ee)
        print(subject, error)
        pass

    # COMMAND = ['mkheadsurf', '-s', subject, '-sd', subjects_dir]
    #
    # p = subprocess.Popen(COMMAND,
    #                      shell=False,
    #                      stderr=subprocess.PIPE)
    # (output, err) = p.communicate()
    #



def process_subject_head(subject, subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    try:
        my_env = os.environ.copy()
        my_env["SUBJECTS_DIR"] = subjects_dir

        COMMAND = ['mne_make_scalp_surfaces', '--subject', subject]

        p = subprocess.Popen(COMMAND, env=my_env,
                             shell=False,
                             stderr=subprocess.PIPE)
        (output, err) = p.communicate()
    except Exception as ee:
        error = str(ee)
        print(subject, error)
        pass


def process_subject_anatomy(subject, t1, subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    reconall = ReconAll()
    reconall.inputs.subject_id = subject
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = subjects_dir
    reconall.inputs.T1_files = t1
    reconall.run()

def run_dcmunpack(subject, subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    mri_dir = '/'.join(subjects_dir.split('/')[:-1])

    dicom_dir = op.join(mri_dir,'DICOM',subject)
    dicom_log = op.join(mri_dir,'DICOM',subject,subject+'_dicom.log')

    COMMAND = ['dcmunpack', '-src', dicom_dir]

    p = subprocess.Popen(COMMAND,
                         shell=False,
                         stdout=file(dicom_log, "w"),
                         stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    return dicom_log


def run_anat(subject, subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    mne.set_config('SUBJECTS_DIR',subjects_dir)
    findsession_log, dicom_dir = run_findsession(subject, subjects_dir=subjects_dir)
    dicom_path = read_findsession_log_file(findsession_log)
    copy_dicom(dicom_path, dicom_dir)
    dicom_log = run_dcmunpack(subject, subjects_dir=subjects_dir)
    t1 = read_dicom_log_file(dicom_log)
    process_subject_anatomy(subject, t1, subjects_dir='/cluster/transcend/MRI/WMA/recons')
    process_subject_bem(subject, subjects_dir=subjects_dir)


run_anat('0180', subjects_dir = '/cluster/transcend/MRI/WMA/recons')

































subjects_dir = '/autofs/cluster/transcend/sheraz/Dropbox/Median_nerve/freesurfer_reconstruction'
subject = 'msh'

process_subject_bem(subject, spacing='ico5')

process_subject_anatomy('matt', '/autofs/cluster/fusion/data/FreeSurfer/MEMPRAGE_4e_1mm-iso.nii', subjects_dir='/autofs/cluster/fusion/data/FreeSurfer')

run_anat('david2', subjects_dir = '/cluster/transcend/MRI/WMA/recons')

import glob
from pyimpress.utils import ParallelExecutor
from joblib import Parallel, delayed

files = glob.glob('/autofs/cluster/fusion/Sheraz/data/camcan/camcan47/cc700/mri/pipeline/release004/BIDSsep/megraw/sub-*')
subjs = [file.split('/')[-1] for file in files]



n_jobs = 20
aprun = ParallelExecutor(n_jobs=n_jobs)
corr_z = aprun(total=len(subjs))(delayed(process_subject_bem)(subj, '/autofs/cluster/transcend/sheraz/camcan_recons/freesurfer') for subj in subjs)








out = Parallel(n_jobs=30)(
    (delayed(process_subject_bem)(subj, '/autofs/cluster/transcend/sheraz/camcan_recons/freesurfer') for subj in subjs))



out = Parallel(n_jobs=30)(
    (delayed(process_subject_head)(subj, '/autofs/cluster/transcend/sheraz/camcan_recons/freesurfer') for subj in subjs))




import mne
import glob

files = glob.glob('/autofs/cluster/fusion/Sheraz/data/camcan/camcan47/cc700/mri/pipeline/release004/BIDSsep/megraw/sub-*')
subjs = [file.split('/')[-1] for file in files]
subj_done = []

for subj in subjs:
    try:
        print(subj)
        mne.gui.coregistration(subjects_dir='/autofs/cluster/transcend/sheraz/camcan_recons/freesurfer',
                               inst='/autofs/cluster/fusion/Sheraz/data/camcan/camcan47/cc700/mri/pipeline/'
                                    'release004/BIDSsep/megraw/' + subj  + '/meg/rest_raw.fif',
                               subject=subj, verbose='ERROR')
        print(subj+ '-Done')
        subj_done.append(subj)
    except Exception as ee:
        error = str(ee)
        print(subj, error)
        continue
        pass






































