from typing import List, Dict, Set
from datetime import datetime
from models import Isolate, Sample, BC_Sample, T2_Sample, Other_Sample, Episode, Patient
from datetime import timedelta

def create_episodes(patients: Dict[str, Patient], bc_window: int = 72, other_sample_window: int = 336, direction: str = 'backward') -> int:
    episode_number = 0
    bc_window_seconds = bc_window * 3600  # Convert hours to seconds
    other_sample_window_seconds = other_sample_window * 3600

    def create_episodes_for_patient(patient: Patient):
        nonlocal episode_number
        samples_with_dates = {k: v for k, v in patient.samples.items() if v.sample_date is not None}
        for sample_id, sample in sorted(samples_with_dates.items(), key=lambda x: x[1].sample_date):
            
            if isinstance(sample, T2_Sample):
                episode = Episode(id=episode_number, t2_sample=sample)
                patient.add_episode(episode)
                for comparing_sample_id, comparing_sample in patient.samples.items():
                    if comparing_sample.sample_date is not None and sample.sample_date is not None:
                        time_difference = (comparing_sample.sample_date - sample.sample_date).total_seconds()
                        
                        if direction == 'backward' and time_difference > 0:
                            continue
                        elif direction == 'forward' and time_difference < 0:
                            continue

                        if isinstance(comparing_sample, BC_Sample) and abs(time_difference) <= bc_window_seconds:
                            episode.add_bc_sample(comparing_sample)
                        elif isinstance(comparing_sample, Other_Sample) and abs(time_difference) <= other_sample_window_seconds:
                            episode.add_other_sample(comparing_sample)
        
                episode_number += 1
                
    for patient_id, patient in patients.items():
        create_episodes_for_patient(patient)
    print(f"Total number of patients: {len(patients)}")
    print(f"Populated patients list with {episode_number} episodes")



def exclude_recurrent_episodes(patients: Dict[str, 'Patient'], t2_gap=7):
    ##Exlude episodes that comes within some days after another T2
    excluded_episodes = {}
    for patient_id, patient in patients.items():
        valid_episodes = {}
        sorted_episodes = sorted(patient.episodes.values(), key=lambda x: x.t2_sample.sample_date)
        last_t2_sample_date = None

        for episode in sorted_episodes:
            if last_t2_sample_date:
                time_difference = episode.t2_sample.sample_date - last_t2_sample_date
                if time_difference > timedelta(days=t2_gap):
                    valid_episodes[episode.id] = episode
                    last_t2_sample_date = episode.t2_sample.sample_date
                else:
                    excluded_episodes[episode.id] = episode
            else:
                valid_episodes[episode.id] = episode
                last_t2_sample_date = episode.t2_sample.sample_date

        patient.episodes = valid_episodes

    print(f"{len(excluded_episodes)} episodes with prior T2 within {t2_gap} days")   
    return excluded_episodes


from datetime import datetime

def filter_episodes_by_date_range(patients, start_date: datetime, end_date: datetime):
    ##If needed, filter episodes by date range if T2 falls in it
    for patient_id, patient in patients.items():
        filtered_episodes = []
        for episode in patient.episodes:
            if start_date <= episode.t2_sample.sample_date <= end_date:
                filtered_episodes.append(episode)
        patient.episodes = filtered_episodes

def exclude_episodes_based_on_condition(patients, condition_func, message):
    excluded_episodes = {}
    for patient_id, patient in patients.items():
        episodes_to_remove = {}
        for episode_id, episode in patient.episodes.items():
            if condition_func(episode):  
                episodes_to_remove[episode_id] = episode
                excluded_episodes[episode_id] = episode

        for episode_id in episodes_to_remove.keys():
            del patient.episodes[episode_id]
    
    print(f"{len(excluded_episodes)} {message}")
    return excluded_episodes

def exclude_invalid_episodes(patients):
    condition = lambda episode: len(episode.t2_sample.invalid_T2) == 6
    message = "episodes with completely invalid T2 results excluded"
    return exclude_episodes_based_on_condition(patients, condition, message)

def exclude_empty_episodes(patients):
    condition = lambda episode: len(episode.bc_samples) == 0
    message = "episodes with no BC samples excluded"
    return exclude_episodes_based_on_condition(patients, condition, message)

def exclude_patients(patients):
    ## Exclude patients without episodes
    excluded_patients = {}
    keys_to_remove = []

    for patient_id, patient in patients.items():
        if not patient.episodes:  # Check if the patient has no episodes
            keys_to_remove.append(patient_id)
            excluded_patients[patient_id] = patient  # Store them in the 'excluded' dictionary

    # Remove patients without episodes from the original dictionary
    for key in keys_to_remove:
        patients.pop(key)
    print(f"{len(excluded_patients)} patients with no valid episodes (T2 sample without BCs) excluded")
    return excluded_patients