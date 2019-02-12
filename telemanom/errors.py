import numpy as np
import pandas as pd 
import more_itertools as mit
import os

from telemanom._globals import Config

config = Config("config.yaml")


def get_errors(y_test, y_hat, anom, smoothed=True):
    """Calculate the difference between predicted telemetry values and actual values, then smooth residuals using
    ewma to encourage identification of sustained errors/anomalies.

    Args:
        y_test (np array): array of test targets corresponding to true values to be predicted at end of each sequence
        y_hat (np array): predicted test values for each timestep in y_test  
        anom (dict): contains anomaly information for a given input stream
        smoothed (bool): If False, return unsmooothed errors (used for assessing quality of predictions)
    Returns:
        e (list): unsmoothed errors (residuals)
        e_s (list): smoothed errors (residuals)
    """

    e = [abs(y_h-y_t[0]) for y_h,y_t in zip(y_hat, y_test)]

    if not smoothed:
        return e

    smoothing_window = int(config.batch_size * config.window_size * config.smoothing_perc)
    if not len(y_hat) == len(y_test):
        raise ValueError("len(y_hat) != len(y_test), can't calculate error: %s (y_hat) , %s (y_test)" %(len(y_hat), len(y_test)))
    
    e_s = list(pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten())

    # for values at beginning < sequence length, just use avg
    if not anom['chan_id'] == 'C-2': #anom occurs early in window (limited data available for channel)
        e_s[:config.l_s] = [np.mean(e_s[:config.l_s*2])]*config.l_s 

    np.save(os.path.join("data", anom['run_id'], "smoothed_errors", anom["chan_id"] + ".npy"), np.array(e_s))

    return e_s


def process_errors(y_test, e_s, logger=None):
    '''Using windows of historical errors (h = batch size * window size), calculate the anomaly
    threshold (epsilon) and group any anomalous error values into continuos sequences. Calculate
    score for each sequence using the max distance from epsilon.
    Args:
        y_test (np array): test targets corresponding to true telemetry values at each timestep t
        e_s (list): smoothed errors (residuals) between y_test and y_hat
        logger (obj): logging object
    Returns:
        E_seq (list of tuples): Start and end indices for each anomaloues sequence
        anom_scores (list): Score for each anomalous sequence
    '''

    window_size = config.window_size
    num_windows = int((y_test.shape[0] - (config.batch_size * window_size)) / config.batch_size)

    # decrease the historical error window size (h) if number of test values is limited
    while num_windows < 0:
        window_size -= 1
        if window_size <= 0:
            window_size = 1
        num_windows = int((y_test.shape[0] - (config.batch_size*window_size)) / config.batch_size)
        if window_size == 1 and num_windows < 0:
            raise ValueError("Batch_size (%s) larger than y_test (len=%s). Adjust in config.yaml." % (config.batch_size, y_test.shape[0]))

    i_anom_full = []  # anomaly indices
    len_y_test = len(y_test)

    for i in range(num_windows + 1):
        start_idx = i * config.batch_size
        if i < num_windows:
            end_idx = start_idx + config.window_size * config.batch_size
        else:
            end_idx = y_test.shape[0]  # for last window, need to cut it short

        window_y_test = y_test[start_idx:end_idx]
        window_e_s = e_s[start_idx:end_idx]

        i_anom, epsilon = get_anomalies(window_e_s, window_y_test, i, i_anom_full, len_y_test)
        i_anom_full.extend([i_a + i * config.batch_size for i_a in i_anom])

    # group anomalous indices into continuous sequences
    i_anom_full = sorted(list(set(i_anom_full)))
    E_seq = group_continuous_seq(i_anom_full)

    # calc anomaly scores based on max distance from epsilon for each sequence
    anom_scores = []
    m = np.mean(e_s)
    s = np.std(e_s)
    for e_seq_start, e_seq_end in E_seq:
        score = max([abs(e_s[x] - epsilon) / (m + s) for x in range(e_seq_start, e_seq_end)])
        anom_scores.append(score)

    return E_seq, anom_scores


def find_epsilon(e_s, e_buffer, std_min=2.5, std_max=12.0, std_step=0.5):
    '''Find the anomaly threshold that maximizes function representing tradeoff between a) number of anomalies
    and anomalous ranges and b) the reduction in mean and st dev if anomalous points are removed from errors
    (see https://arxiv.org/pdf/1802.04431.pdf)
    Args:
        e_s (array): residuals between y_test and y_hat values (smoothes using ewma)
        error_buffer (int): if an anomaly is detected at a point, this is the number of surrounding values
            to add the anomalous range. this promotes grouping of nearby sequences and more intuitive results
        sd_min (float): The min number of standard deviations above the mean to calculate as part of the
            argmax function
        sd_max (float): The max number of standard deviations above the mean to calculate as part of the
            argmax function
        sd_step (float): Step size between sd_min and sd_max
    Returns:
        epsilon (float): the calculated epsilon for the anomaly threshold in number of standard deviations above the
            mean
    '''
    max_metric = 0
    best_z = std_max  # default if no winner found

    mean = np.mean(e_s)
    std = np.std(e_s)
    for z in np.arange(std_min, std_max, std_step):
        epsilon = mean + z * std
        i_anom = set()
        pruned_e_s = []

        for i, e in enumerate(e_s):
            if e <= epsilon:
                pruned_e_s.append(e)
            else:
                buffer_start = max(i - e_buffer, 0)
                buffer_end = min(i + e_buffer + 1, len(e_s))
                i_anom.update(range(buffer_start, buffer_end))

        if len(i_anom) > 0:
            i_anom = sorted(list(i_anom))
            E_seq = group_continuous_seq(i_anom)
            metric = (1 - np.mean(pruned_e_s) / mean + 1 - np.std(pruned_e_s) / std) / (len(E_seq) ** 2 + len(i_anom))
            if metric >= max_metric and len(E_seq) <= 5 and len(i_anom) < (len(e_s) * 0.5):
                max_metric = metric
                best_z = z
    return mean + best_z * std


def compare_to_epsilon(e_s, epsilon, len_y_test, inter_range, chan_std,
                       std, error_buffer, window, i_anom_full):
    '''Compare smoothed error values to epsilon (error threshold) and group consecutive errors together into
    sequences.

    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        epsilon (float): Threshold for errors above which an error is considered anomalous
        len_y_test (int): number of timesteps t in test data
        inter_range (tuple of floats): range between 5th and 95 percentile values of error values
        chan_std (float): standard deviation on test values
        std (float): standard deviation of smoothed errors
        error_buffer (int): number of values surrounding anomalous errors to be included in anomalous sequence
        window (int): Count of number of error windows that have been processed
        i_anom_full (list): list of all previously identified anomalies in test set

    Returns:
        E_seq (list of tuples): contains start and end indices of anomalous ranges
        i_anom (list): indices of errors that are part of an anomlous sequnce
        non_anom_max (float): highest smoothed error value below epsilon
    '''

    i_anom = set()
    E_seq = []
    non_anom_max = 0

    # Don't consider anything in window because scale of errors too small compared to scale of values   
    if not (std > (.05*chan_std) or max(e_s) > (.05 * inter_range)) or not max(e_s) > 0.05:    
        return E_seq, list(i_anom), non_anom_max

    # ignore initial error values until enough history for smoothing, prediction, comparisons
    num_to_ignore = config.l_s * 2
    # if y_test is small, ignore fewer
    if len_y_test < 1800:
        num_to_ignore = 0
    elif len_y_test < 2500:
        num_to_ignore = config.l_s

    for i, e in enumerate(e_s):
        if not e > epsilon or not e > 0.05 * inter_range:
            continue

        if window == 0:
            buffer_start = max(i - error_buffer, num_to_ignore)
            buffer_end = min(i + error_buffer + 1, len(e_s))
        else:
            buffer_start = max(i - error_buffer, len(e_s) - config.batch_size)
            buffer_end = min(i + error_buffer + 1, len(e_s))
        if buffer_end > buffer_start:
            i_anom.update(range(buffer_start, buffer_end))

    # capture max of values below the threshold that weren't previously identified as anomalies
    # (used in filtering process)
    i_anom = sorted(list((i_anom)))
    non_anom_i = [i for i in range(len(e_s)) if i not in i_anom and i+window*config.batch_size not in i_anom_full]
    non_anom_max = max([e_s[i] for i in non_anom_i])

    # group anomalous indices into continuous sequences
    E_seq = group_continuous_seq(i_anom)

    return E_seq, i_anom, non_anom_max


def prune_anoms(E_seq, e_s, non_anom_max, i_anom):
    '''Remove anomalies that don't meet minimum separation from the next closest anomaly or error value

    Args:
        E_seq (list of tuples): contains start and end indices of anomalous ranges
        e_s (list): smoothed errors between y_test and y_hat values
        non_anom_max (float): highest smoothed error value below epsilon
        i_anom (list): indices of errors that are part of an anomlous sequnce

    Returns:
        i_pruned (list): remaining indices of errors that are part of an anomlous sequnces
            after pruning procedure
    '''
    E_seq_max = [max(e_s[start_i:end_i]) for start_i, end_i in E_seq if start_i != end_i]
    e_s_max = E_seq_max.copy()
    e_s_max.sort(reverse=True)

    if non_anom_max and non_anom_max > 0:
        e_s_max.append(non_anom_max)  # for comparing the last actual anomaly to next highest below epsilon

    i_to_remove = []
    p = config.p
    for e_s_max1, e_s_max2 in zip(e_s_max, e_s_max[1:]):
        if (e_s_max1 - e_s_max2) / e_s_max1 < p:
            i_to_remove.append(E_seq_max.index(e_s_max1))
            # p += 0.03 # increase minimum separation by this amount for each step further from max error
        else:
            i_to_remove = []

    E_seq = [val for i, val in enumerate(E_seq) if i not in i_to_remove]

    i_pruned = []
    for i in i_anom:
        for e_seq_start, e_seq_end in E_seq:
            if e_seq_start <= i <= e_seq_end:
                i_pruned.append(i)
                break

    return i_pruned


def get_anomalies(e_s, y_test, window, i_anom_full, len_y_test):
    '''Find anomalous sequences of smoothed error values that are above error threshold (epsilon). Both
    smoothed errors and the inverse of the smoothed errors are evaluated - large dips in errors often
    also indicate anomlies.
    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        y_test (np array): test targets corresponding to true telemetry values at each timestep for given window
        z (float): number of standard deviations above mean corresponding to epsilon
        window (int): number of error windows that have been evaluated
        i_anom_full (list): list of all previously identified anomalies in test set
        len_y_test (int): num total test values available in dataset
    Returns:
        i_anom (list): indices of errors that are part of an anomlous sequnces
    '''

    perc_high, perc_low = np.percentile(y_test, [95, 5])
    inter_range = perc_high - perc_low

    std_window_e_s = np.std(e_s)
    std_window_y_test = np.std(y_test)

    epsilon = find_epsilon(e_s, config.error_buffer)
    # flip around the mean for the inverse
    mean_window_e_s = np.mean(e_s)
    window_e_s_inv = np.array([mean_window_e_s + (mean_window_e_s - e) for e in e_s])
    epsilon_inv = find_epsilon(window_e_s_inv, config.error_buffer)

    E_seq, i_anom, non_anom_max = compare_to_epsilon(e_s, epsilon, len_y_test,
                                                     inter_range, std_window_y_test, std_window_e_s,
                                                     config.error_buffer, window, i_anom_full)

    # find sequences of anomalies using inverted error values (lower than normal errors are also anomalous)
    E_seq_inv, i_anom_inv, inv_non_anom_max = compare_to_epsilon(window_e_s_inv, epsilon_inv,
                                                                 len_y_test, inter_range, std_window_y_test,
                                                                 std_window_e_s, config.error_buffer, window,
                                                                 i_anom_full)

    if len(E_seq) > 0:
        i_anom = prune_anoms(E_seq, e_s, non_anom_max, i_anom)

    if len(E_seq_inv) > 0:
        i_anom_inv = prune_anoms(E_seq_inv, window_e_s_inv, inv_non_anom_max, i_anom_inv)
    i_anom = list(set(i_anom + i_anom_inv))

    return i_anom, epsilon


def evaluate_sequences(E_seq, anom):
    '''Compare identified anomalous sequences with labeled anomalous sequences

    Args:
        E_seq (list of tuples): contains start and end indices of anomalous ranges
        anom (dict): contains anomaly information for a given input stream

    Returns:
        anom (dict): with updated anomaly information (whether identified, scores, etc.)
    '''

    anom["false_positives"] = 0
    anom["false_negatives"] = 0
    anom["true_positives"] = 0
    anom["fp_sequences"] = []
    anom["tp_sequences"] = []
    anom["num_anoms"] = len(anom["anomaly_sequences"])   

    E_seq_test = eval(anom["anomaly_sequences"])

    if len(E_seq) > 0:
        matched_E_seq_test = []

        for e_seq1, e_seq2 in E_seq:
            valid = False

            for i, (a1, a2) in enumerate(E_seq_test):
                if (a1 <= e_seq1 <= a2) or (a1 <= e_seq2 <= a2) or (e_seq1 <= a1 and e_seq2 >= a2) or (a1 <= e_seq1 and a2 >= e_seq2):

                    anom["tp_sequences"].append((e_seq1, e_seq2))
                    valid = True

                    if i not in matched_E_seq_test:
                        anom["true_positives"] += 1
                        matched_E_seq_test.append(i)

            if valid == False:
                anom["false_positives"] += 1
                anom["fp_sequences"].append([e_seq1, e_seq2])

        anom["false_negatives"] += (len(E_seq_test) - len(matched_E_seq_test))
    else:
        anom["false_negatives"] += len(E_seq_test)

    return anom


def group_continuous_seq(i_anom):
    '''Group anomalous indices into countious sequences

    Args:
        i_anom (set or list): contains anomaly indices

    Returns:
        (list of tuples): contains start and end indices of anomalous ranges
    '''

    i_anom = sorted(list(i_anom))
    groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
    return E_seq
