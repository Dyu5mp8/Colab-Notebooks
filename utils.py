import re
import pandas as pd


def contains(pattern, comparison_string):
    """
    Checks if the comparison_string contains the regex pattern, case-insensitive.
    
    Args:
    pattern (str): The regex pattern.
    comparison_string (str): The string to compare against the pattern.
    
    Returns:
    bool: True if there's a match, False otherwise.
    """
    return bool(re.search(pattern, comparison_string, re.IGNORECASE))
    
def calculate_time_metrics(episodes, clinic_regex=None):
    BC_total_time = []
    BC_time_to_prel_report = []
    BC_time_to_arrival = []
    T2_total_time = []
    T2_time_to_prel_report = []
    T2_time_to_arrival = []

    for episode in episodes.values():
        for sample in episode.bc_samples:
            if clinic_regex and not re.search(clinic_regex, sample.clinic, re.IGNORECASE):
                continue
            
            if sample.total_time_to_id is not None:
                BC_total_time.append(sample.total_time_to_id.total_seconds())
            if sample.time_to_prel_id_report is not None:
                BC_time_to_prel_report.append(sample.time_to_prel_id_report.total_seconds())
            if sample.time_to_arrival is not None:
                BC_time_to_arrival.append(sample.time_to_arrival.total_seconds())

                
        if clinic_regex and not re.search(clinic_regex, episode.t2_sample.clinic, re.IGNORECASE):
            continue
            
        if episode.t2_sample.total_time is not None:
            T2_total_time.append(episode.t2_sample.total_time.total_seconds())
        if episode.t2_sample.time_to_prel_report is not None:
            T2_time_to_prel_report.append(episode.t2_sample.time_to_prel_report.total_seconds())
        if episode.t2_sample.time_to_arrival is not None:
            T2_time_to_arrival.append(episode.t2_sample.time_to_arrival.total_seconds())

        



    BC_total_time = [t / 3600 for t in BC_total_time]
    BC_time_to_prel_report = [t / 3600 for t in BC_time_to_prel_report]
    BC_time_to_arrival = [t / 3600 for t in BC_time_to_arrival]
    T2_total_time = [t / 3600 for t in T2_total_time]
    T2_time_to_prel_report = [t / 3600 for t in T2_time_to_prel_report]
    T2_time_to_arrival = [t / 3600 for t in T2_time_to_arrival]
    
    return BC_total_time, BC_time_to_prel_report, BC_time_to_arrival, T2_total_time, T2_time_to_prel_report, T2_time_to_arrival

def calculate_time_metrics_to_df(episodes, clinic_regex=None):
    data = []

    for episode_nr, episode in episodes.items():
        for sample in episode.bc_samples:
            if clinic_regex and not re.search(clinic_regex, sample.clinic, re.IGNORECASE):
                continue

            if sample.total_time_to_id is not None:
                data.append(["BC", sample.total_time_to_id.total_seconds() / 3600, "Total time", episode.id, sample.id])
            if sample.time_to_prel_report is not None:
                data.append(["BC", sample.time_to_prel_id_report.total_seconds() / 3600, "Time to Prel Report", episode.id, sample.id])
            if sample.time_to_arrival is not None:
                data.append(["BC", sample.time_to_arrival.total_seconds() / 3600, "Time to Arrival", episode.id, sample.id])

        if clinic_regex and not re.search(clinic_regex, episode.t2_sample.clinic, re.IGNORECASE):
            continue

        if episode.t2_sample.total_time is not None:
            data.append(["T2", episode.t2_sample.total_time.total_seconds() / 3600, "Total time", episode_nr, episode.t2_sample.id])
        if episode.t2_sample.time_to_prel_report is not None:
            data.append(["T2", episode.t2_sample.time_to_prel_report.total_seconds() / 3600, "Time to Prel Report", episode_nr, episode.t2_sample.id])
        if episode.t2_sample.time_to_arrival is not None:
            data.append(["T2", episode.t2_sample.time_to_arrival.total_seconds() / 3600, "Time to Arrival", episode_nr, episode.t2_sample.id])

    df = pd.DataFrame(data, columns=["Test type", "Time", "Time type", "Episode nr", "Sample ID"])
    
    return df
