import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import suite2p

from pathlib import Path
import mat73
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
import scipy
import scipy.stats as stats
import scipy.io
import scipy.ndimage

sessions = ['24h_pre', '3h_pre', 'tone_d1', 'tone_d2', 'tone_d3', 'tone_d4', 'tone_d5', 'test']
behaviour_sessions = ['tone_d1', 'tone_d2', 'tone_d3', 'tone_d4', 'tone_d5', 'test']
training_sessions = ['tone_d1', 'tone_d2', 'tone_d3', 'tone_d4', 'tone_d5']
num_tones = np.array([30,30,30,30,30,20])
num_tones_dict = {'tone_d1':30, 'tone_d2':30, 'tone_d3':30, 'tone_d4':30, 'tone_d5':30, 'test':20}


# ~~~~~~~~~~ CONVENIENCE FUNCTIONS ~~~~~~~~~~
  
    
def onset_offset(vect):
    """Returns an array of onsets and offsets for a binary vector"""
    onsets = np.where(np.diff(vect) == 1)[0]
    offsets = np.where(np.diff(vect) == -1)[0]
    
    return onsets, offsets

def onset_offset_with_timeout(vect, timeout):
    """Calculates the onsets and offsets from a binary vector. If a second onset falls within the timeout period, it is not counted """
    i = 0
    onsets = []
    offsets = []
    onsetsearch = True
    while i < len(vect)-1:
        if onsetsearch == True:
            if vect[i] == 0 and vect[i+1] == 1:
                onsets.append(i)
                i += timeout   # Jump forward "timeout" units
                onsetsearch = False
        if onsetsearch == False:
            if np.any(vect[i:i+timeout]) == 0:
                offsets.append(i)
                onsetsearch = True
        i += 1    # If neither case is true, advance by one and try again

    return np.array(onsets), np.array(offsets)

def onoff_to_vect(vect_len, onsets, offsets):
    """Converts a list of onsets and offsets to a vector of 0s and 1s"""

    vect = np.zeros(vect_len)
    windows = list(zip(onsets, offsets))
    for w in windows:
        start, end = w
        vect[start:end] = 1
    return vect


# ~~~~~~~~~~ LICKING BEHAVIOUR CODE ~~~~~~~~~~

def df_to_licks_and_triggers(df, CSplus, num_tones = 20, trigger_length = 10):
    if type(df) == str:
        df = pd.read_csv(df, sep='\t')
    if type(CSplus) == str:
        CSplus = np.loadtxt(CSplus, dtype=int)

    t1 = df['Trigger 1'].values
    t2 = df['Trigger 2'].values
    timestamps = df['Timestamp'].values

    trigger_onset = np.where(np.diff(t2)>2)[0]

    srate = len(timestamps)/timestamps[-1]
    
    trigger_offset = np.round(trigger_onset+(trigger_length*srate)).astype(int)

    #Finding CS+ and CS-
    CSp = np.array(CSplus)-1

    CSm = np.arange(0,num_tones).tolist()
    for i in sorted(CSp, reverse=True):
        del CSm[i]
    CSm = np.array(CSm)

    # CS triggers
    trigrange_csp = np.array([np.arange(trigger_onset[i], trigger_offset[i]).tolist() for i in CSp]).flatten()
    trigrange_csm = np.array([np.arange(trigger_onset[i], trigger_offset[i]).tolist() for i in CSm]).flatten()

    blank_trig = np.zeros(len(t2))

    csp_triggers = blank_trig.copy()
    csp_triggers[trigrange_csp] = 1

    csm_triggers = blank_trig.copy()
    csm_triggers[trigrange_csm] = 1

    csall_triggers = blank_trig.copy()
    csall_triggers[trigrange_csm] = 1
    csall_triggers[trigrange_csp] = 1


    # Computing licks
    threshold = 3.5
    lick_filt = (t1-threshold > 0).astype(int)
    lick_on_off = np.diff(lick_filt)
    lick_index = np.where(lick_on_off > 0)[0][::2]

    licks = blank_trig.copy()
    licks[lick_index] = 1


    #For cancelling out erroneous triggers at each tone onset
    trigger_offset_cancel = np.round(trigger_onset+(100)).astype(int)
    trigrange_cancel = np.array([np.arange(trigger_onset[i], trigger_offset_cancel[i]).tolist() for i in range(len(CSp)+len(CSm))]).flatten()

    cancel_triggers = np.ones(len(t2))
    cancel_triggers[trigrange_cancel] = 0

    licks = cancel_triggers * licks


    #Licks during CS+ and CS-
    csp_licks = np.where(csp_triggers*licks == 1)[0]
    csm_licks = np.where(csm_triggers*licks == 1)[0]

    # Baseline licking
    csno_triggers = (csall_triggers-1)*-1
    cs_baseline_licks = np.where(csno_triggers*licks == 1)[0]
    
    return csp_licks, csp_triggers, csm_licks, csm_triggers, cs_baseline_licks, csno_triggers, licks, csall_triggers, srate, timestamps

def calculate_lick_stats(csp_licks, csp_triggers, csm_licks, csm_triggers, cs_baseline_licks, csno_triggers, licks, csall_triggers, srate):
    lickrate_baseline = len(cs_baseline_licks)/(sum(csno_triggers)/srate)
    lickrate_csp = len(csp_licks)/(sum(csp_triggers)/srate)
    lickrate_csm = len(csm_licks)/(sum(csm_triggers)/srate)
    
    csp_ratio = lickrate_csp/lickrate_baseline
    csm_ratio = lickrate_csm/lickrate_baseline
    
    return lickrate_baseline, lickrate_csp, lickrate_csm, csp_ratio, csm_ratio

def files_to_lick_stats(analog_file, csplus_list):
    df = pd.read_csv(analog_file, sep='\t')

    plicks, ptrigs, mlicks, mtrigs, baselicks, basetrigs, licks, alltrigs, srate, timestamps = df_to_licks_and_triggers(df, csplus_list)

    baseline_lickrate, csp_lickrate, csm_lickrate, csp_ratio, csm_ratio = calculate_lick_stats(plicks, ptrigs, mlicks, mtrigs, baselicks, basetrigs, licks, alltrigs, srate)
    return baseline_lickrate, csp_lickrate, csm_lickrate, csp_ratio, csm_ratio 

def files_to_lick_dict(analog_file, csplus_file, num_tones=20, trigger_length=10):
    df = pd.read_csv(analog_file, sep='\t')
    
    csp_licks, csp_tone_vector, csm_licks, csm_tone_vector, base_licks, base_vector, all_licks, all_vector, srate, timestamps = df_to_licks_and_triggers(df, csplus_file, num_tones, trigger_length)
    baseline_lickrate, csp_lickrate, csm_lickrate, csp_ratio, csm_ratio = calculate_lick_stats(csp_licks, csp_tone_vector, csm_licks, csm_tone_vector, base_licks, base_vector, all_licks, all_vector, srate)
    
    return_dict = {}
    for i in ('csp_licks', 'csp_tone_vector', 'csm_licks', 'csm_tone_vector', 'base_licks', 'base_vector', 'all_licks', 'all_vector', 'srate', 
              'baseline_lickrate', 'csp_lickrate', 'csm_lickrate', 'csp_ratio', 'csm_ratio', 'timestamps'):
        return_dict[i] = eval(i)
        
    return return_dict

def make_downsample_array(orig_events, orig_len, downsample_factor):
    """Downsamples set of triggers and fits them into a new array
    orig_events: array of indices when events occurred
    orig_len: length of the original session
    downsample_factor: factor to reduce sample rate by"""
    
    # Create new blank array at the downsampled length
    ds_vect_len = int(orig_len/downsample_factor)
    ds_vector = np.zeros(ds_vect_len)

    
    ds_event_ixs = (orig_events/downsample_factor).astype(int)
    ds_vector[ds_event_ixs] += 1

    return ds_event_ixs, ds_vector

def make_downsample_array2(orig_events_list, orig_timestamps, target_srate=20):
    """Downsamples a set of triggers based on times from orig_timestamps. Returns a list of downsampled triggers and a vector of events
    orig_events_list: list containing arrays of indices when events occurred.
    orig_timestamps: array of timestamps from original behaviour recording
    target_srate: sample rate (Hz) that you wish to downsample to
    
    Returns event_list, where each line contains two variables (below). One line per item in orig_events_list
    -  ds_event_ixs: list of indices when events occurred in downsampled time
    -  ds_event_vect: downsampled vector of events"""

    target_freq = 1/target_srate
    orig_srate = len(orig_timestamps)/orig_timestamps[-1]
    downsample_factor = orig_srate/target_srate
    
    ticks = np.arange(0, np.ceil(orig_timestamps[-1]), target_freq)  # These are the timestamps for the downsampled array
    #time_ixs = [np.argmin(np.abs(orig_timestamps-t)) for t in tqdm(ticks)]  # This takes too long, so we have to do something else

    # We want the indices where the original timestamp array matches our downsampled array.
    # Since orig_timestamp arrays may not be uniform, we have to look around each tick to find the closest match
    time_ixs = []
    window = 4000 # This determines how many values on either side of our theoretical target to look. Consider making this value larger if encountering errors
    for t_ix, t in enumerate(ticks):
        do_correction = True   # This flag controls whether the index rendered by this loop must be adjusted. If we are starting from a nonzero index, we must correct

        start = int((t_ix+1)*downsample_factor)-window
        if start < 0:   # If our starting value would be below zero, reset to zero. In this case, the resulting index does not need to be corrected
            start = 0
            do_correction = False

        end = int((t_ix+1)*downsample_factor)+window
        if end > len(orig_timestamps):
            end = len(orig_timestamps)

        uncorrected_ix = np.argmin(np.abs(orig_timestamps[start:end] - t))   # The heart of the loop. Find the value in orig_timestamps that is closest to target tick

        if do_correction:
            corrected_ix = (uncorrected_ix-window) + int((t_ix+1)*downsample_factor)
        else:
            corrected_ix = uncorrected_ix
            
        time_ixs.append(corrected_ix)
    
    ds_time = orig_timestamps[time_ixs]   # Apply these new indices to the original timestamp array to generate timestamps for the downsampled array

    # Now we unpack orig_events_list and process each item
    event_list = []
    for event in orig_events_list:
        times_where_event = orig_timestamps[event]   # Find the original timestamps where each event occurred
        ds_event_ixs = [np.argmin(np.abs(ds_time - t)) for t in times_where_event]  # Do the same thing as above, but more compact because this runs fast (relatively few times_where_event)
        # ^ For each timestamp when an event happened, find the closest corresponding timestamp in the downsampled array

        ds_event_vect = np.zeros(len(ds_time))  # Build the vector
        ds_event_vect[ds_event_ixs] = 1
        
        event_list.append([np.array(ds_event_ixs), ds_event_vect])
    
    return event_list

def make_downsample_array2_old(orig_events, orig_len, target_len):
    """Downsamples set of triggers and fits them into a new array
    orig_events: array of indices when events occurred
    orig_len: length of the original (behaviour) session
    target_len: length of the desired output vector (same as imaging session length)"""
    
    # Create new blank array at the downsampled length
    ds_vector = np.zeros(target_len)
    downsample_factor = orig_len / target_len

    
    ds_event_ixs = (orig_events/downsample_factor).astype(int)
    ds_vector[ds_event_ixs] += 1

    return ds_event_ixs, ds_vector

# ~~~~~~~~~~ GEN2 BEHAVIOUR STUFF ~~~~~~~~~~

def df_to_licks_and_triggers_gen2(df):
    """Returns vectors and indices of licks during various experimental phases.
    For use with Gen 2 behaviour code (Dhana's tone_training python scripts)"""
    
    if type(df) == str:
        df = pd.read_csv(df, header=10)[:-2]
    
    srate = len(df) / float(df['time'].values[-1])
    timestamps = df['time'].astype(float).values
    
    threshold = 3.5
    lick_onset, _ = onset_offset((df['analog_lick_trace'] > threshold).astype(int))
    licks = onoff_to_vect(len(df), lick_onset, lick_onset+1)   # Vector of licks, where each lick is a single tick
    
    csp_triggers = df['CS+'].values  #Vector where CS+ is on
    csm_triggers = df['CS-'].values  # Vector where CS- is on
    csall_triggers = csp_triggers + csm_triggers    # Vector where either CS is on
    csno_triggers = (csall_triggers-1)*-1   # Vector where no CS is on

    csp_licks = np.where(csp_triggers*licks == 1)[0]  # List of indices where licks occur during CS+
    csm_licks = np.where(csm_triggers*licks == 1)[0] # List of indices where licks occur during CS-
    cs_baseline_licks = np.where(csno_triggers*licks == 1)[0] # List of indices where licks occur without CS

    return csp_licks, csp_triggers, csm_licks, csm_triggers, cs_baseline_licks, csno_triggers, licks, csall_triggers, srate, timestamps

def get_experimental_params_gen2(df_path):
    """Accepts a path to a behaviour CSV file generated by Gen 2 scripts (Dhana's tone_training scripts).
    Returns a dict with experimental parameters. Note that total licks do not correspond with total licks from df_to_licks_and_triggers_gen2.
    Not sure why this discrepancy exists, but I trust the data from df_to_licks_and_triggers_gen2 more than the data from here"""
    
    initial_params = dict(pd.read_csv(df_path, nrows=9).values)  # Pulls initial rows containing basic parameters
    totals = dict(pd.read_csv(df_path, header=10)[-2:][['time', 'analog_lick_trace']].values)  # Pulls the last two rows, where total licks and rewards were written

    exp_params = {**initial_params, **totals}
    
    return exp_params

def files_to_lick_dict_gen2(csv_file):
    """Accepts path to a Gen2 CSV file (generated using Dhana's tone_training scripts) and returns a dict of relevant values"""
    
    csp_licks, csp_tone_vector, csm_licks, csm_tone_vector, base_licks, base_vector, all_licks, all_vector, srate, timestamps = df_to_licks_and_triggers_gen2(csv_file)
    baseline_lickrate, csp_lickrate, csm_lickrate, csp_ratio, csm_ratio = calculate_lick_stats(csp_licks, csp_tone_vector, csm_licks, csm_tone_vector, base_licks, base_vector, all_licks, all_vector, srate)
    
    return_dict = {}
    for i in ('csp_licks', 'csp_tone_vector', 'csm_licks', 'csm_tone_vector', 'base_licks', 'base_vector', 'all_licks', 'all_vector', 'srate', 
              'baseline_lickrate', 'csp_lickrate', 'csm_lickrate', 'csp_ratio', 'csm_ratio', 'timestamps'):
        return_dict[i] = eval(i)
        
    return return_dict



# ~~~~~~~~~~ PARSING 2P DATA ~~~~~~~~~~.

def moving_average(x, w):
    """Moving average of 1D vector X, window size of W"""
    return np.convolve(x, np.ones(w), 'valid') / w

def load_cellreg_matfile(mat_filepath):
    """Returns the results dict from a loaded output matlab file generated using CellReg"""
    try:
        matfile = mat73.loadmat(mat_filepath)['cell_registered_struct']
    except TypeError:
        matfile = scipy.io.loadmat(mat_filepath)['cell_registered_struct']
    return matfile

def make_s2p_dict(parent_folder_path):
    """Creates a dict of relevant suite2p outputs (ops, stat, iscell, F and F_neu files)"""
    fol = parent_folder_path

    stat_path = fol/'stat.npy'
    iscell_path = fol/'iscell_alex_sort.npy'
    op_path = fol/'ops.npy'
    f_path = fol/'F.npy'
    f_neu_path = fol/'Fneu.npy'


    ops = np.load(op_path, allow_pickle=True).item()
    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)[:,0].astype(bool)
    f = np.load(f_path, allow_pickle=True)
    f_neu = np.load(f_neu_path, allow_pickle=True)

    info_dict = {'ops':ops, 'stat':stat, 'iscell':iscell, 'F':f, 'Fneu':f_neu}
    return info_dict

def glob_find(base_path, glob_pattern, exclude_list=['classifier', 'backup'], verbose=True):
    """Iterates through base_path and subdirectories and returns a list of paths that satisfy glob_pattern"""
    base_path = Path(base_path)
    glob_list = list(base_path.rglob(glob_pattern))
    glob_posix = [i.as_posix() for i in glob_list]
    
    target_paths = [Path(k).parent for k in glob_posix if all(sub not in k for sub in exclude_list)]  # List comp to remove all items in exclude_list
    if verbose == True:
        print(len(target_paths), 'files found')
    return target_paths

def glob_to_s2p_dict(base_path, glob_pattern, exclude_list=['classifier', 'backup'], tqdm_disable=False):
    """Creates a list of suite2p dicts from an input glob pattern and base path location (by combining glob_find and make_s2p_dict)"""
    target_paths = glob_find(base_path, glob_pattern, exclude_list, verbose=False)
    
    # We need to rearrange the results of target paths to match the order that sessions were recorded in
    # This is the complicated but complete solution.  Usually we could just move 'test' from [2] to the end, but what if the folders appear in a different order?
    # Therefore we compare the current order to the correct session order
    sessions = ['24h_pre', '3h_pre', 'tone_d1', 'tone_d2', 'tone_d3', 'tone_d4', 'tone_d5', 'test']
    current_order = [i.parts[-3] for i in target_paths]
    pos = [sessions.index(i) for i in current_order] # Now figure out the position of the current order relative to the correct order
    target_paths_sorted = [y for x,y in sorted(zip(pos, target_paths))]  # This is many steps. Zip the position and path together, sort these on position, then extract corresponding path
    
    s2p_dict_list = []
    for t in tqdm(target_paths_sorted, disable=tqdm_disable):
        d = make_s2p_dict(t)
        s2p_dict_list.append(d)
        
    return s2p_dict_list

def organize_footprints_traces(cell_ids, s2p_dict_list, aligned_footprints):
    """Create arrays for traces, neuropil_traces and footprints in the same shape as cell_id"""
    base_arr = np.ma.zeros(cell_ids.shape, dtype='object')
    base_arr.mask = cell_ids.mask
    traces = base_arr.copy()
    neu = base_arr.copy()
    footprints = base_arr.copy()
    #traces = np.ma.zeros(cell_ids.shape, dtype='object')
    #neu = np.ma.zeros(cell_ids.shape, dtype='object')
    #footprints = np.ma.zeros(cell_ids.shape, dtype='object')

    rows, cols = cell_ids.shape

    for day in range(cols):
        iscell = s2p_dict_list[day]['iscell']
        f = s2p_dict_list[day]['F'][iscell]
        f_neu = s2p_dict_list[day]['Fneu'][iscell]
        
        footprint_array = aligned_footprints[day]

        for cell in range(rows):
            ix = cell_ids[cell, day]
            if ix >=0:
                trace = f[ix]
                fneu = f_neu[ix]
                fp = footprint_array[ix]

                traces[cell, day] = trace
                neu[cell, day] = fneu
                footprints[cell,day] = fp
                
    return footprints, traces, neu

def process_mouse_2p_data(base_path, cage_no, mouse_no, tqdm_disable=False):
    """Given an input mouse folder, returns ID array of cells across days, plus associated footprints and traces
    
    Input: Directories must be structured in a very specific way:
    CellReg data - under parent directory, / 'cellreg_results' / cage_no / mouse_no 
    Suite2p data - under parent directory, / cage_no / mouse_no / [iscell_alex_sort.npy, F.npy, F_neu.npy, ops.npy, stat.npy]
    
    Outputs:
    cell_ids - identity of each day's cell and its position in the multi-day registration matrix (n_total_cells_detected x n_sessions)
    footprints - height x width array of pixels for each spatial footprint. Organized into multiday registration matrix
    traces - activity traces associated with each cell, organized by registration matrix
    neuropil - neuropil trace associated with each cell, organized by registration matrix"""
    
    s2p_path = Path(base_path)/cage_no/mouse_no
    cellreg_path = Path(base_path)/'cellreg_results'/cage_no/mouse_no
    
    # Deal with CellReg data from matlab
    cellreg_matfile_path = list(cellreg_path.rglob('cellRegistered*.mat'))[0]  # This assumes only one cellreg matfile per folder
    matfile = load_cellreg_matfile(cellreg_matfile_path)
    # Get cellreg footprints
    aligned_footprints = matfile['spatial_footprints_corrected']  # Find the aligned footprints

    # More recent versions of Cellreg (at least since 1.5.1) use a different format for saving spatial footprints. Let's check what kind we're dealing with
    if type(aligned_footprints[0][0]) is str:  # Newer cellreg files save a list of path strings instead of arrays directly
        # This is actually pretty stupid, because if we ever move our files from the location they were created, all the referencing breaks.
        # So instead of just taking the file paths from the matfile and reading them, we are going to parse the paths and just extract the filenames.
        # This assumes that the aligned files will always be kept in the same directory as the cellRegistered matfile.
        aligned_fields = []
        for i in aligned_footprints:
            parent_dir = cellreg_matfile_path.parent
            aligned_fname = i[0].split('\\')[-1]
            align_load = scipy.io.loadmat(parent_dir/aligned_fname)['footprint'].squeeze() # For each path provided, go and load that matfile
            aligned_footprints = np.array([i.toarray() for i in align_load]) # Files are saved as [n_cells x 1 sparse matrix per cell] Unpack the sparse matrices and combine them
            aligned_fields.append(np.max(aligned_footprints, axis=0))  # Get a max projection of each cell's footprint to get the whole field
    elif type(aligned_footprints[0][0]) is np.ndarray:
        aligned_footprints = [i[0] for i in aligned_footprints]
        aligned_fields = [np.max(i, axis=0) for i in aligned_footprints]  # Make a nice max projection object for plotting [n_sessions x (lxh array)]
    else:
        print('There is something wrong with the cellreg matfile provided. Check the spatial_footprints_corrected value')
        return
    # Create cell_id matrix
    cell_ids_raw = matfile['cell_to_index_map']
    cell_ids = cell_ids_raw.copy().astype(int)-1
    cell_ids = np.ma.masked_equal(cell_ids, -1)
    
    # Get suite2p data
    s2p_dict_list = glob_to_s2p_dict(s2p_path, 'iscell_alex_sort.npy', tqdm_disable = tqdm_disable)
    footprints, traces, neuropil = organize_footprints_traces(cell_ids, s2p_dict_list, aligned_footprints)
    
    return cell_ids, footprints, traces, neuropil, aligned_fields


def unmask_and_stack(masked_array, flip=True):
    """Returns an array of only unmasked values from input masked_array, formatted into a 2D array (good for pulling just good traces)"""
    unmasked_stack = np.ma.column_stack(masked_array.compressed())
    if flip == True:
        unmasked_stack = unmasked_stack.T
    return unmasked_stack.data

# ~~~~~~~~~~ DATA HOLDER OBJECTS ~~~~~~~~~~

             
"""            
class Behaviour:
    """'Holds information from parsed behaviour files'"""
    def __init__(self, csp_licks, csp_tone_vector, csm_licks, csm_tone_vector, base_licks, base_vector, all_licks, all_vector, srate, baseline_lickrate, csp_lickrate, csm_lickrate, csp_ratio, csm_ratio):
        self.csp_licks = csp_licks
        self.csp_tone_vector = csp_tone_vector
        self.csm_licks = csm_licks
        self.csm_tone_vector = csm_tone_vector
        self.base_licks = base_licks
        self.base_vector = base_vector
        self.all_licks = all_licks
        self.all_vector = all_vector
        self.srate = srate
        self.baseline_lickrate = baseline_lickrate
        self.csp_lickrate = csp_lickrate
        self.csm_lickrate = csm_lickrate
        self.csp_ratio = csp_ratio
        self.csm_ratio = csm_ratio
    def downsample_all(self, target_rate=20):
        """'Creates new downsamples trig arrays and vectors'"""
        orig_len = len(self.all_vector)
        downsample_factor = int(self.srate/target_rate)
        
        self.ds_csp_licks, self.ds_csp_tone_vector = make_downsample_array(self.csp_licks, orig_len, downsample_factor)
        self.ds_csm_licks, self.ds_csm_tone_vector = make_downsample_array(self.csm_licks, orig_len, downsample_factor)
        self.ds_all_licks, self.ds_all_licks_vector = make_downsample_array(np.concatenate([self.csp_licks, self.csm_licks]).sort(), orig_len, downsample_factor)
        self.df_all_tones_vector = self.all_vector[::downsample_factor]
        
"""

@dataclass
class Behaviour:
    """Holds information from parsed behaviour files"""
    csp_licks: np.ndarray
    csp_tone_vector: np.ndarray
    csm_licks: np.ndarray
    csm_tone_vector: np.ndarray
    base_licks: np.ndarray
    base_vector: np.ndarray
    all_licks: np.ndarray
    all_vector: np.ndarray
    srate: np.float64
    baseline_lickrate: np.float64
    csp_lickrate: np.float64
    csm_lickrate: np.float64
    csp_ratio: np.float64
    csm_ratio: np.float64
    timestamps: np.ndarray
        
    def __post_init__(self):
        try:
            self.discrimination_ratio = len(self.csp_licks)/(len(self.csp_licks)+len(self.csm_licks))
        except ZeroDivisionError:
            print('Discrimination ratio could not be calculated, possibly due to no licking')
            self.discrimination_ratio = np.nan
        
    def downsample_all(self, target_srate):
        """Creates new downsampled trig arrays and vectors. Dowsample results in an array of target_len"""
        all_licks = np.concatenate([self.csp_licks, self.csm_licks, self.base_licks])
        all_licks.sort()

        # this is a bit stupid, but downsampling just the starts and ends of the tone vects saves a LOT of time. We will stitch them back into an array later
        all_tone_starts, all_tone_ends = onset_offset(self.all_vector)
        csp_tone_starts, csp_tone_ends = onset_offset(self.csp_tone_vector) 
        csm_tone_starts, csm_tone_ends = onset_offset(self.csm_tone_vector)

        events_to_downsample = [self.csp_licks, self.csm_licks, all_licks, \
            all_tone_starts, all_tone_ends, csp_tone_starts, csp_tone_ends, csm_tone_starts, csm_tone_ends]

        ds_events = make_downsample_array2(events_to_downsample, self.timestamps, target_srate)

        [self.ds_csp_licks, self.ds_csp_lick_vector], \
        [self.ds_csm_licks, self.ds_csm_lick_vector], \
        [self.ds_all_licks, self.ds_all_licks_vector], \
        [ds_all_starts, _], [ds_all_ends, _], [ds_csp_starts, _], [ds_csp_ends, _], [ds_csm_starts, _], [ds_csm_ends, _] = ds_events

        self.ds_all_tones_vector = onoff_to_vect(len(self.ds_csp_lick_vector), ds_all_starts, ds_all_ends)
        self.ds_csp_tone_vector = onoff_to_vect(len(self.ds_csp_lick_vector), ds_csp_starts, ds_csp_ends)
        self.ds_csm_tone_vector = onoff_to_vect(len(self.ds_csp_lick_vector), ds_csm_starts, ds_csm_ends)

        #self.ds_csp_licks, self.ds_csp_lick_vector = make_downsample_array2(self.csp_licks, self.timestamps, target_srate)
        #self.ds_csm_licks, self.ds_csm_lick_vector = make_downsample_array2(self.csm_licks, self.timestamps, target_srate)
        #self.ds_all_licks, self.ds_all_licks_vector = make_downsample_array2(np.concatenate([self.csp_licks, self.csm_licks, self.base_licks]), self.timestamps, target_srate)
        #self.ds_all_licks.sort()
        #self.ds_all_tones_vector = make_downsample_array2(np.where(self.all_vector == 1)[0], self.timestamps, target_srate)
        #self.ds_csp_tone_vector = make_downsample_array2(np.where(self.csp_tone_vector == 1)[0], self.timestamps, target_srate)
        #self.ds_csm_tone_vector = make_downsample_array2(np.where(self.csm_tone_vector == 1)[0], self.timestamps, target_srate)
        
        #self.ds_csp_licks, self.ds_csp_lick_vector = make_downsample_array2(self.csp_licks, orig_len, target_len)
        #self.ds_csm_licks, self.ds_csm_lick_vector = make_downsample_array2(self.csm_licks, orig_len, target_len)
        #self.ds_all_licks, self.ds_all_licks_vector = make_downsample_array2(np.concatenate([self.csp_licks, self.csm_licks, self.base_licks]), orig_len, target_len)
        #self.ds_all_licks.sort()
        #self.ds_all_tones_vector = (scipy.ndimage.interpolation.zoom(self.all_vector, 1/downsample_factor) > 0.5).astype(int)  # Use interpolation.zoom to rescale array, then apply cutoff of 0.5 to ensure we have a nice binary array
        #self.ds_csp_tone_vector = (scipy.ndimage.interpolation.zoom(self.csp_tone_vector, 1/downsample_factor) > 0.5).astype(int)
        #self.ds_csm_tone_vector = (scipy.ndimage.interpolation.zoom(self.csm_tone_vector, 1/downsample_factor) > 0.5).astype(int)

class Day:
    """Contains data parsed out from larger multi-day dataset, plus behaviour data through Behaviour subclass"""
    def __init__(self, name, cell_ids, footprints, traces, neuropil, cage, mouse, spikes):
        self.name = name
        self.cell_ids = cell_ids
        self.footprints = footprints
        self.traces_orig = traces
        self.neuropil = neuropil
        self.cage = cage
        self.mouse = mouse
        self.spikes_orig = spikes
        
        self.present = ~self.cell_ids.mask
        self.num_present = np.sum(self.present)
        self.trace_len = len(unmask_and_stack(self.traces_orig)[0])
        
        # Reshape traces to not be an array of arrays
        self.traces = np.ma.zeros([self.traces_orig.shape[0], self.trace_len])
        self.traces.mask = True
        for i in range(len(self.traces_orig)):
            if self.traces_orig.mask[i] == False:
                self.traces[i] = self.traces_orig[i]
                
        self.spikes = np.ma.zeros([self.spikes_orig.shape[0], self.trace_len])
        self.spikes.mask = True
        for i in range(len(self.spikes_orig)):
            if self.spikes_orig.mask[i] == False:
                self.spikes[i] = self.spikes_orig[i]

    def add_behaviour(self, behaviour_parent_folder, num_tones, trigger_length=7):
        """Pulls behaviour file based on base folder, cage and mouse name"""
        base_path = Path(behaviour_parent_folder)/self.cage/self.mouse
        
        glob_behaviour_pattern = '*{}*{}.txt'.format(self.name, self.mouse)
        glob_csplus_pattern = '*{}*{}_CSplus*'.format(self.name, self.mouse)
        behaviour_path = list(Path(base_path).glob(glob_behaviour_pattern))
        csplus_path = list(Path(base_path).glob(glob_csplus_pattern))
        
        if len(behaviour_path) > 1 or len(csplus_path) > 1:
            print('ERROR: glob has detected too many files. Check filenames of ', behaviour_path)
        if len(behaviour_path) == 0 or len(csplus_path) == 0:  
            print('ERROR: glob did not find any files. Check base path, currently:', base_path)
        else:
            behaviour_path = behaviour_path[0]
            csplus_path = csplus_path[0]

        self.behaviour_dict = files_to_lick_dict(behaviour_path.as_posix(), csplus_path.as_posix(), num_tones, trigger_length)
        self.behaviour = Behaviour(*self.behaviour_dict.values())
    
    def add_behaviour_gen2(self, behaviour_parent_folder):
        """Pulls Gen2 behaviour file (generated by Dhana tone_train scripts) based on base folder, cage and mouse name"""
        base_path = Path(behaviour_parent_folder)/self.cage/self.mouse
        
        glob_behaviour_pattern = f'*{self.name}*{self.mouse}*.csv'
        behaviour_path = list(Path(base_path).glob(glob_behaviour_pattern))
        
        if len(behaviour_path) > 1:
            print('ERROR: glob has detected too many files. Check filenames of ', behaviour_path)
        if len(behaviour_path) == 0:  
            print('ERROR: glob did not find any files. Check base path, currently:', base_path)
        else:
            behaviour_path = behaviour_path[0]

        self.behaviour_dict = files_to_lick_dict_gen2(behaviour_path.as_posix())
        self.behaviour = Behaviour(*self.behaviour_dict.values())

        


class Mouse:
    def __init__(self, base_path, cage_no, mouse_no, tqdm_disable=True):
        self.tqdm_disable = tqdm_disable
        self.cage = cage_no
        self.mouse = mouse_no
        self.id = cage_no + '_' + mouse_no
        self.sessions = ['24h_pre', '3h_pre', 'tone_d1', 'tone_d2', 'tone_d3', 'tone_d4', 'tone_d5', 'test']
        
        self.cell_ids, self.footprints, self.traces, self.neuropil, self.aligned_fields = process_mouse_2p_data(base_path, cage_no, mouse_no, tqdm_disable = self.tqdm_disable)
                
    def smooth_normalize_traces(self, binning_window=10):
        """Replaces self.traces with an updated version where neuropil baseline is subtracted, firing is normalized to max 1, and traces are smoothed with a moving average"""
        self.original_traces = self.traces.copy()
        
        row_len, col_len = self.traces.shape
        for row in range(row_len):
            for col in range(col_len):
                mask = self.traces.mask[row, col]
                if mask == False:
                    self.traces[row,col] = self.traces[row, col] - self.neuropil[row,col]
                    self.traces[row, col] = moving_average(self.traces[row, col], binning_window)   # Apply moving average
                    self.traces[row, col] = self.traces[row, col] - np.min(self.traces[row, col])   # Make minimum zero
                    self.traces[row, col] = self.traces[row, col]/np.max(self.traces[row, col])   # Normalize to maximum
                    
    def get_spikes(self, spike_cutoff_mode = 'global', iqrs_above_med = 2):
        """Binarizes data from traces into 'on' and 'off' with a cutoff defined as some number of interquartile ranges above the median.
        spike_cutoff_mode = 'global': cutoff value is determined from the median and IQR of all traces in self.traces
        spike_cutoff_mode = 'percell': cutoff value is determined from each cell's median and IQR"""
        self.spikes = self.traces.copy()
        self.spikes_cutoff = np.zeros(self.traces.shape)
        
        if spike_cutoff_mode is 'global':
            alltraces = np.concatenate(self.traces[~self.traces.mask])   # Create one long vector of every activity trace
            med = np.median(alltraces)
            iqr = scipy.stats.iqr(alltraces)
            cutoff = med+(iqrs_above_med*iqr)
        
        row_len, col_len = self.traces.shape
        for row in range(row_len):
            for col in range(col_len):
                mask = self.traces.mask[row, col]
                if mask == False:
                    if spike_cutoff_mode == 'percell':
                        med = np.median(self.traces[row, col])
                        iqr = scipy.stats.iqr(self.traces[row,col])
                        cutoff = med + (iqrs_above_med * iqr)
                    self.spikes[row,col] = self.traces[row, col] > cutoff
                    self.spikes_cutoff[row, col] = cutoff

        
    def split_to_days(self):
        """Separates columns of the registered matrix variables (cell_ids, traces, etc.) into separate entries per day"""
        self.days = {}
        for ix, sess in enumerate(self.sessions):
            self.days[sess] = Day(sess, self.cell_ids[:,ix], self.footprints[:,ix], self.traces[:,ix], self.neuropil[:,ix], self.cage, self.mouse, self.spikes[:,ix])
            
    def add_all_behaviour(self, behaviour_parent_folder, behaviour_file_type = 'gen1', num_tones_list = num_tones, trigger_length = 7):
        """Requires split_to_days has been run. Adds behaviour info to all day objects"""
        valid_sess = self.sessions[2:]  # We have to exclude 24h_pre and 3h_pre because they have no behaviour
        for ix, sess in enumerate(tqdm(valid_sess, disable=self.tqdm_disable)):
            if behaviour_file_type == 'gen1':
                self.days[sess].add_behaviour(behaviour_parent_folder, num_tones_list[ix], trigger_length)
            if behaviour_file_type == 'gen2':
                self.days[sess].add_behaviour_gen2(behaviour_parent_folder)



            
    def add_all_downsamples(self, target_srate=20):
        valid_sess = self.sessions[2:]  # We have to exclude 24h_pre and 3h_pre because they have no behaviour
        for ix, sess in enumerate(tqdm(valid_sess, disable=self.tqdm_disable)):
            beh = self.days[sess].behaviour
            beh.downsample_all(target_srate)
            # Shorten behaviour arrays to be the same length as imaging arrays
            imaging_len = len(self.traces[:,ix+2].compressed()[0]) # Pulls the first unmasked entry from the traces array for a given day, and finds length. 
            #Add 2 to compensate for removed sessions in valid sess
            beh.ds_csp_tone_vector = beh.ds_csp_tone_vector[:imaging_len]
            beh.ds_csm_tone_vector = beh.ds_csm_tone_vector[:imaging_len]
            beh.ds_all_licks_vector = beh.ds_all_licks_vector[:imaging_len]
            beh.ds_all_tones_vector = beh.ds_all_tones_vector[:imaging_len]
            
    def process(self, behaviour_parent_folder, behaviour_file_type = 'gen1', num_tones_list = num_tones, binning_window=10, 
    spike_cutoff_mode = 'global', iqrs_above_med=2, trigger_length=7, target_srate = 20):
        self.tqdm_disable=True
        with tqdm(total=5) as pbar:
            self.smooth_normalize_traces(binning_window)
            pbar.update(1)
            self.get_spikes(spike_cutoff_mode, iqrs_above_med)
            pbar.update(1)
            self.split_to_days()
            pbar.update(1)
            self.add_all_behaviour(behaviour_parent_folder, behaviour_file_type, num_tones_list = num_tones_list, trigger_length = trigger_length)
            pbar.update(1)
            self.add_all_downsamples(target_srate)
            pbar.update(1)
        self.tqdm_disable=False
            
            
            
            
            
# ~~~~~~~~~~STOLEN ANDREW FUNCTIONS ~~~~~~~~~~

def on_offsets(vector):
    onsets = []
    offsets = []
    for tt in range(len(vector)):
        if tt + 1 == len(vector):
            break
        if vector[tt] == 0 and vector[tt+1] == 1:
            onsets.append(tt)
        elif vector[tt] == 1 and vector[tt+1] == 0:
            offsets.append(tt)
    return np.array(onsets), np.array(offsets)




# ~~~~~~~~~~ PLOTTING ~~~~~~~~~~.

def sigbars(xy_left, xy_right, height, label_height=0.25, label='*', linewidth=1, color='k', pyplot_obj = plt):
    # Makes a significance bar between xy_left and xy_right
    x_left = xy_left[0]
    x_right = xy_right[0]
    y_left = xy_left[1]
    y_right = xy_right[1]
    
    #Calculate distances
    y_max = np.maximum(y_left, y_right)+height
    x_middle = (x_left+ x_right)/2
    y_label = y_max+label_height  
    
    
    pyplot_obj.vlines(x_left, y_left, y_max, linewidth=1, color=color)
    pyplot_obj.vlines(x_right, y_right, y_max, linewidth=1, color=color)
    pyplot_obj.hlines(y_max, x_left, x_right, linewidth=1, color=color)
    
    pyplot_obj.annotate(label, (x_middle, y_label), ha='center')
    
    
def bar_scatterplot(data_list, join_points=False, scatter_spread=0.02, scatter_color = 'gray', scatter_alpha = 0.5, bar_color = 'tab:blue', bar_capsize=5):
    X = np.arange(len(data_list))

    plt.bar(X, np.nanmean(data_list, axis=1), yerr = stats.sem(data_list, axis=1, nan_policy='omit'), capsize=bar_capsize, color=bar_color, zorder=-1)

    if join_points == True:
        plt.plot(data_list, '-o', color=scatter_color, alpha=scatter_alpha)
    else:
        for i in range(len(data_list)):
            arr = data_list[i]
            scatter_X = np.array([i]*len(arr))
            scatter_X = scatter_X+(np.random.randn(len(scatter_X)) * scatter_spread)
            plt.scatter(scatter_X, arr, color=scatter_color, alpha=scatter_alpha)
            
def bar_scatterplot_unequal(data_list, join_points=False, scatter_spread=0.02, scatter_color = 'gray', scatter_alpha = 0.5, bar_color = 'tab:blue', bar_capsize=5):
    X = np.arange(len(data_list))

    for i in range(len(data_list)):
        plt.bar(X[i], np.nanmean(data_list[i]), yerr = stats.sem(data_list[i], nan_policy='omit'), capsize=bar_capsize, color=bar_color, zorder=-1)

    if join_points == True:
        plt.plot(data_list, '-o', color=scatter_color, alpha=scatter_alpha)
    else:
        for i in range(len(data_list)):
            arr = data_list[i]
            scatter_X = np.array([i]*len(arr))
            scatter_X = scatter_X+(np.random.randn(len(scatter_X)) * scatter_spread)
            plt.scatter(scatter_X, arr, color=scatter_color, alpha=scatter_alpha)




# ~~~~~~~~~~ LAB MEETING ANALYSIS FUNCTIONS ~~~~~~~~~~

def get_block(vect, onsets, offsets, axis=0):
    """Chops up a vect into blocks defined between onset and offset. Returns an N x X array where N=number of blocks and X = block length"""
    
    blocks = []
    for i in range(len(onsets)):
        start = onsets[i]
        end = offsets[i]
        
        if isinstance(vect, np.ma.MaskedArray) == True:
            vect = vect.data
        
        vect_reshape = np.swapaxes(vect, 0, axis)
        window = vect_reshape[start:end]
        window_reshape = np.swapaxes(window, 0, axis)
            
        blocks.append(window_reshape)
    blocks = np.array(blocks)
    return blocks


def get_bin_means(vect, frames_per_bin, axis):
    """Takes an input (multidimensional) vector and computes means for time bins along a given axis. Frames_per_bin controls how many values are binned together"""
    vect_reshape = np.swapaxes(vect, 0, axis)
    bin_starts = np.arange(0, vect.shape[axis], frames_per_bin)
    bin_ends = bin_starts + frames_per_bin


    bin_means = []
    for i in range(len(bin_starts)):
        window = vect_reshape[bin_starts[i]:bin_ends[i]]
        bin_mean = np.mean(window, axis=0)
        bin_means.append(bin_mean)

    bin_means = np.swapaxes(np.array(bin_means), 0, axis)
    return bin_means

def mask_invert(original_masked_array):
    """Inverts the unmasked values of an array while keeping the mask the same"""
    holder = original_masked_array.copy().astype(int)
    holder = holder.filled(-999)
    holder[holder == 0] = 10
    holder[holder == 1] = 0
    holder[holder == 10] = 1
    holder[holder == -999] = 0
    output = np.ma.array(holder).astype(original_masked_array.dtype)
    output.mask = original_masked_array.mask
    output.fill_value = original_masked_array.fill_value
    return output








if __name__ == '__main__':
    #pass
    tst = Mouse('F:/round_3', '684172', 'ms4')
    tst.process(r'F:\round_3\behaviour_results', 'gen2')

    tst2 = Mouse('F:/round_2', '684169', 'ms1')
    tst2.process(r'F:\round_2\behaviour_results')
