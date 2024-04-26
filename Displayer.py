from models import Isolate, Sample, BC_Sample, T2_Sample, Episode, Patient, Other_Sample
from collections import Counter
import numpy as np
from typing import List, Dict, Set
from datetime import datetime
from T2studyPlotter import T2studyPlotter
from scipy import stats
import stat_utils as su
import re
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from utils import calculate_time_metrics
from utils import calculate_time_metrics_to_df
from bactdict import T2_LIST
import pandas as pd

class Displayer:
  ### INIT
    def __init__(self, patients):
        self.patients = patients
        self.samples = {k: v for patient in self.patients.values() for k, v in patient.samples.items()}
        self.episodes = {k: v for patient in self.patients.values() for k, v in patient.episodes.items()}
        
        print(f"Total patients: {len(self.patients)}")
        print(f"Total episodes: {len(self.episodes)}")
        print(f"Total samples: {len(self.samples)} (note that this is all samples, not only within episodes")

    ### get a plotter

    def get_plotter(self):
    
        plotter = T2studyPlotter(self.patients, self.samples, self.episodes)

        return plotter
    
    ### display data for single patient,episode or sample ###
    
    def display_patient_info(self, patient_id):
        if patient_id not in self.patients:
            print(f"No patient with ID: {patient_id}")
            return
        patient = self.patients[patient_id]
        patient.display()
      

    def display_sample_info(self, sample_id):
        sample = self.samples.get(sample_id, None)
        if sample:
            sample.display()
        else:
            print(f"No sample with ID: {sample_id}")

    def display_episode_info(self, episode_id):
        episode = self.episodes.get(episode_id, None)
        if episode:
            episode.display()
        else:
            print(f"No episode with ID: {episode_id}")

    ### long list of all patients
    
    def display_all_patients(self):
        for patient in self.patients.values():
            patient.display_episodes()
    #### display aggregate most common bacteria
    def display_specific_isolates(self, isolate_name: str):
        for episode in self.episodes.values():
            detected_by_t2_names = {isolate.name for isolate in episode.t2_isolates}
            detected_by_bc_names = {isolate.name for isolate in episode.bc_isolates_in_panel}
            if isolate_name in detected_by_t2_names or isolate_name in detected_by_bc_names:
                episode.display()

    def all_samples(self):
        episodes = self.episodes.values()
        samples = [sample for episode in episodes for sample in episode.other_samples]
        return samples

    
    def t2pos_bcneg_other_samples(self):
        # First, filter the episodes
        episodes = [episode for episode in self.episodes.values() if episode.classify(compare_with="BC_IN_PANEL") == "T2_POS_COMPARISON_NEG"]
        samples = [sample for episode in episodes for sample in episode.other_samples]
        return samples

    def t2neg_bcpos_other_samples(self):
        episodes = [episode for episode in self.episodes.values() if episode.classify(compare_with="BC_IN_PANEL") == "T2_NEG_COMPARISON_POS"]
        samples = [sample for episode in episodes for sample in episode.other_samples]
        return samples

            
    def format_interval(interval):
    # Format the left and right edges of the interval to remove decimal points
        if interval.left.is_integer() and interval.right.is_integer():
            return f"[{int(interval.left)}, {int(interval.right)})"
        else:
            # In case they are not integers, you can adjust formatting as needed
            return f"[{interval.left:.2f}, {interval.right:.2f})"



    def process_isolates(self, isolates, counter, condition=lambda x: True):
        for isolate in isolates:
            if condition(isolate):
                counter[isolate.name] += 1
            
    def display_most_common_bacteria(self):
        ap_counter = Counter()
        cc_counter = Counter()
        cont_counter = Counter()
        other_counter = Counter()
    
        for patient in self.patients.values():
            for episode in patient.episodes.values():
                for sample in episode.bc_samples:
                    for isolate in sample.isolates:
                        if isolate.is_contaminant:
                            cont_counter[isolate.name] += 1
                        else:
                            ap_counter[isolate.name] += 1

                for sample in episode.other_samples:  # Assuming you have a list or iterable named other_samples in Episode
                    for isolate in sample.isolates:
                        other_counter[isolate.name] += 1
                                
                # Check if cc_sample and isolates are available
                if episode.t2_sample and episode.t2_sample.isolates:
                    for isolate in episode.t2_sample.isolates:
                        cc_counter[isolate.name] += 1

        
        print("Most Common Bacteria in BC samples")
        for bacteria, count in ap_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")
    
        print("Most Common Bacteria in T2 samples")
        for bacteria, count in cc_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")
    
        print("Most Common Contaminants")
        for bacteria, count in cont_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")

        print("Most Common isolates in other samples")
        for bacteria, count in other_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")


    def display_most_common_bacteria_in_episodes(self, is_t2included = False):
        ap_counter = Counter()
        cc_counter = Counter()
        cont_counter = Counter()
        other_counter = Counter()
    
        condition = (lambda x: x.is_t2included) if is_t2included else (lambda x: True)
    
        for episode in self.episodes.values():
            self.process_isolates(episode.bc_contaminants, cont_counter, condition)
            self.process_isolates(episode.t2_isolates, cc_counter, condition)
            self.process_isolates(episode.bc_isolates, ap_counter, condition)
            self.process_isolates(episode.other_sample_isolates, other_counter, condition)
        
        print("Most Common Bacteria in BC samples")
        for bacteria, count in ap_counter.most_common(40):  # top 15
            print(f"\t{bacteria}: {count}")
    
        print("Most Common Bacteria in T2 samples")
        for bacteria, count in cc_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")
    
        print("Most Common Contaminants")
        for bacteria, count in cont_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")

        print("Most Common isolates in other samples")
        for bacteria, count in other_counter.most_common(15):  # top 15
            print(f"\t{bacteria}: {count}")

    def get_mean_time_between_samples(self):
        time_differences = []  # to store the time differences between T2 and BC samples

        for episode in self.episodes.values():
            t2_time = episode.t2_sample.sample_date  # Replace with your actual attribute name for T2 time
            
            for bc_sample in episode.bc_samples:
                bc_time = bc_sample.sample_date  # Replace with your actual attribute name for BC time
                time_difference = abs((t2_time - bc_time).total_seconds() / 3600.0)  # Converting Timedelta to hours
                time_differences.append(time_difference)

        desc = su.calculate_descriptives(time_differences)
        print(desc)
        

    
#### main descriptive display
    
    def display_aggregate_data(self):
        total_patients = len(self.patients)
        total_episodes = len(self.episodes)
        
        def get_basic_stats():
            age_data = [p.age for p in self.patients.values()]
            print(f"Total no of patients: {total_patients}")
            print("Age")
            print(f"Mean age: {np.mean(age_data)}")
            print(f"Min age: {np.min(age_data)}")
            print(f"Max age: {np.max(age_data)}")
            print(f"Standard deviation of age: {np.std(age_data):.2f}")

        def get_gender_stats():
            gender_data = [p.gender for p in self.patients.values()]
            male_count = gender_data.count(1)
            female_count = gender_data.count(2)
            print("Gender")
            print(f"No. of males: {male_count} ({male_count / len(self.patients) * 100:.2f}%)")
            print(f"No. of females: {female_count} ({female_count / len(self.patients) * 100:.2f}%)")

        def get_episode_stats():
            episode_counter = Counter([len(patient.episodes) for patient in self.patients.values()])
            print("Episodes")
            print(f"Total number of episodes: {total_episodes}")
            for num_episodes, count in sorted(episode_counter.items()):
                print(f"\t{count} patients had {num_episodes} episodes")

        def get_sample_stats():
            total_bc_samples = 0
            total_other_samples = 0
            bc_sample_counter = Counter()
            other_sample_counter = Counter()
            t2_icu_count = 0
            
          
            for episode in self.episodes.values():
                
                total_bc_samples += len(episode.bc_samples)
                total_other_samples += len(episode.other_samples)
                
                bc_sample_counter[len(episode.bc_samples)] += 1
                other_sample_counter[len(episode.other_samples)] += 1
                
                if episode.t2_sample.icu:
                    t2_icu_count += 1

            print("Samples")
            print(f"Total no of T2 samples: {total_episodes}")  # Make sure 'total_episodes' is defined somewhere
            print(f"Total no of BC samples: {total_bc_samples}")
            print(f"Total no of other samples: {total_other_samples}")
            print(f"Episodes where T2 was taken in ICU: {t2_icu_count} ({t2_icu_count / total_episodes * 100:.2f})")
            
            total_above_cutoff = 0
            cutoff = 6
            
            print("Episodes by number of BC samples")
            for num_samples, count in sorted(bc_sample_counter.items()):
                if num_samples <= cutoff:
                    print(f"\t{count} episodes had {num_samples} BC samples")
                else:
                    total_above_cutoff += count
            
            if total_above_cutoff > 0:
                print(f"\t{total_above_cutoff} episodes had more than {cutoff} BC samples")

            total_above_cutoff = 0
        
            print("Episodes by number of other samples")
            for num_samples, count in sorted(other_sample_counter.items()):
                if num_samples <= cutoff:
                    print(f"\t{count} episodes had {num_samples} other samples")
                else:
                    total_above_cutoff += count
            
            if total_above_cutoff > 0:
                print(f"\t{total_above_cutoff} episodes had more than {cutoff} other samples")
            
            

        def get_sample_distribution():
            for sample_type in ['BC', 'Other']:
                if sample_type == 'BC':
                    sample_counts = [len(episode.bc_samples) for patient in self.patients.values() for episode in patient.episodes.values()]
                else:
                    sample_counts = [len(episode.other_samples) for patient in self.patients.values() for episode in patient.episodes.values()]
                
                median = np.median(sample_counts)
                q1 = np.percentile(sample_counts, 25)
                q3 = np.percentile(sample_counts, 75)
                iqr = q3 - q1
        
                print(f"{sample_type} sample median sample count")
                print(f"Median: {median}")
                print(f"Q1: {q1}")
                print(f"Q3: {q3}")
                print(f"IQR: {iqr}")


        print("Aggregate Data:")
        get_basic_stats()
        get_gender_stats()
        get_episode_stats()
        get_sample_stats()
        get_sample_distribution()
    
    def get_other_sample_locales(self, group='all'):
        if group == 'all':
            samples = self.all_samples()
        elif group == 't2pos_bcneg':
            samples = self.t2pos_bcneg_other_samples()
        elif group == 't2neg_bcpos':
            samples = self.t2neg_bcpos_other_samples()
        else:
            print("only 'all', 't2pos_bcneg' or 't2neg_bcpos'")
            return

        locales = [sample.categorized_locale for sample in samples if isinstance(sample, Other_Sample)]
        locale_count = Counter(locales)
        total_samples = sum(locale_count.values())

        if total_samples == 0:
            print("No samples found.")
            return

        sorted_locale_count = locale_count.most_common()
        for locale, count in sorted_locale_count:
            percentage = (count / total_samples) * 100
            print(f"Locale: {locale}, Count: {count} ({percentage:.2f}%)")

        print(f"Total samples: {total_samples}")

    
    def positivity_count(self):
        any_positive=0
        any_positive_in_panel=0
        t2_positive= 0
        bc_positive_in_panel = 0
        bc_positive = 0
        positive_bc_samples = 0
        for episode in self.episodes.values():
            if episode.t2_isolates:
                t2_positive +=1 
            if episode.bc_isolates_in_panel:
                bc_positive_in_panel +=1
            if episode.bc_isolates:
                bc_positive += 1

            if episode.bc_isolates_in_panel or episode.t2_isolates:
                any_positive_in_panel+=1
            if episode.bc_isolates or episode.t2_isolates:
                any_positive+=1
   

            for sample in episode.bc_samples:
                if sample.get_t2panel_isolates():
                    positive_bc_samples += 1


                    
        print("=" * 30)
        print(f"Any positive episodes: {any_positive}")
        print(f"Any positive episodes with bacteria in panel: {any_positive_in_panel}")
        print(f"T2 positive episodes: {t2_positive}")
        print(f"BC positive (in-panel): {bc_positive_in_panel}")
        print(f"BC positive (all relevant): {bc_positive}")
        print("=" * 30)
        print(f"positive BC samples (in-panel): {positive_bc_samples}")


    def polymicrobial_not_only_panel(self):
        poly_episodes = []
    
        # Prepare data for DataFrame
        data = {
            'Episode_ID': [],
            'T2_Isolates': [],
            'BC_Isolates': [],
            'Other_Sample_Isolates': []
        }
    
        for episode in self.episodes.values():
            combined_isolates = episode.t2_isolates | episode.bc_isolates
            if len(combined_isolates) > 1:
                episode.display()
                poly_episodes.append(episode)
    
                # Extract and sort isolate names for T2 and BC
                t2_isolate_names = sorted([isolate.name for isolate in episode.t2_isolates])
                bc_isolate_names = sorted([isolate.name for isolate in episode.bc_isolates])
    
                # Process isolates from other samples
                other_sample_dict = {}
                for sample in episode.other_samples:
                    for isolate in sample.isolates:
                        if isolate.name not in other_sample_dict:
                            other_sample_dict[isolate.name] = set()
                        other_sample_dict[isolate.name].add(sample.categorized_locale)
    
                # Format isolates with their locales
                other_sample_isolates = []
                for isolate, locales in other_sample_dict.items():
                    isolate_str = f"{isolate} ({', '.join(sorted(locales))})"
                    other_sample_isolates.append(isolate_str)
    
                # Append data
                data['Episode_ID'].append(episode.id)
                data['T2_Isolates'].append(', '.join(t2_isolate_names))
                data['BC_Isolates'].append(', '.join(bc_isolate_names))
                data['Other_Sample_Isolates'].append('; '.join(sorted(other_sample_isolates)))
    
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
     
    def episode_classification(self, compare_with):
        episode_classification_counter = Counter()
        valid_categories = ["ALL_BC", "BC_IN_PANEL", "OTHER_IN_PANEL"]
        if compare_with not in valid_categories:
            raise ValueError(f"Invalid comparison category. Choose from {valid_categories}")
        
        for episode in self.episodes.values():
            classification = episode.classify(compare_with)
            episode_classification_counter[classification] += 1

        # Count all episodes where both tests are positive, including subsets
        both_pos = (episode_classification_counter['BOTH_POS_MATCH'] +
                          episode_classification_counter['T2_SUBSET_OF_COMPARISON_SET'] +
                          episode_classification_counter['COMPARISON_SUBSET_OF_T2'])

        # Construct the contingency table for McNemar's test
        # Note: b and c are discordant pairs (one positive, one negative)
        contingency_table = [[both_pos, episode_classification_counter['T2_POS_COMPARISON_NEG']],
                             [episode_classification_counter['T2_NEG_COMPARISON_POS'], episode_classification_counter['ALL_NEG']]]
        print(contingency_table)

        # Perform McNemar's test
        mcnemar_result = mcnemar(contingency_table, exact=False, correction=False)

        print("\nEpisode Classification Summary:")
        print("=" * 30)
        for classification, count in episode_classification_counter.items():
            print(f"{classification.replace('_', ' ').title()}: {count}")
        print("=" * 30)

        # Print McNemar's test result
        print("\nMcNemar's Test Result:")
        print(f"Statistic: {mcnemar_result.statistic}, P-value: {mcnemar_result.pvalue}\n")

   
    def discordant_results(self, compare_with, result):     ## display discordant results, result is the classification  
        valid_categories = ["ALL_BC", "BC_IN_PANEL", "OTHER_IN_PANEL"]
        if compare_with not in valid_categories:
            raise ValueError(f"Invalid comparison category. Choose from {valid_categories}")
        for episode in self.episodes.values():
            classification = episode.classify(compare_with)  # Or however you call it
            if result == classification:
                episode.display()
  
   
    def discordant_check_other_samples(self):
        count_t2_positive_bc_negative_other_negative = 0
        count_t2_positive_bc_negative_other_positive = 0
        count_t2_positive_bc_negative_other_empty = 0
    
        count_bc_positive_t2_negative_other_negative = 0
        count_bc_positive_t2_negative_other_positive = 0
        count_bc_positive_t2_negative_other_empty = 0
        
        count_both_positive_other_negative = 0
        count_both_positive_other_positive = 0
        count_both_positive_other_empty = 0
        
        for episode in self.episodes.values():
            if not episode.other_samples:
                if episode.t2_isolates and not episode.bc_isolates_in_panel:
                    count_t2_positive_bc_negative_other_empty += 1
                elif episode.bc_isolates_in_panel and not episode.t2_isolates:
                    count_bc_positive_t2_negative_other_empty += 1
                elif episode.t2_isolates and episode.bc_isolates_in_panel:
                    count_both_positive_other_empty += 1
            else:
                if episode.t2_isolates and not episode.bc_isolates_in_panel:
                    if episode.t2_isolates & episode.other_sample_isolates_in_panel:
                        count_t2_positive_bc_negative_other_positive += 1
                    else:
                        count_t2_positive_bc_negative_other_negative += 1
                elif not episode.t2_isolates and episode.bc_isolates_in_panel:
                    if episode.bc_isolates_in_panel & episode.other_sample_isolates_in_panel:
                        count_bc_positive_t2_negative_other_positive += 1
                    else:
                        count_bc_positive_t2_negative_other_negative += 1
                elif episode.t2_isolates and episode.bc_isolates_in_panel:
                    if episode.t2_isolates & episode.other_sample_isolates_in_panel:
                        count_both_positive_other_positive += 1
                    else:
                        count_both_positive_other_negative += 1
                        

        bc_positives = count_bc_positive_t2_negative_other_empty + count_bc_positive_t2_negative_other_negative + count_bc_positive_t2_negative_other_positive
        t2_positives = count_t2_positive_bc_negative_other_empty + count_t2_positive_bc_negative_other_negative + count_t2_positive_bc_negative_other_positive
        both_positives = count_both_positive_other_empty + count_both_positive_other_negative + count_both_positive_other_positive
        
        print(f"|------Total T2 positive: {t2_positives}")
        print(f"Total count of episodes where T2 is positive but BC is negative and no other samples were taken: {count_t2_positive_bc_negative_other_empty}")
        print(f"Total count of episodes where T2 is positive, BC is negative, and other samples are positive: {count_t2_positive_bc_negative_other_positive}")
        print(f"Total count of episodes where T2 is positive, BC is negative, and other samples are negative: {count_t2_positive_bc_negative_other_negative}")
        print(f"|------Total BC positive: {bc_positives}") 
        print(f"Total count of episodes where BC is positive but T2 is negative and no other samples were taken: {count_bc_positive_t2_negative_other_empty}")
        print(f"Total count of episodes where BC is positive, T2 is negative, and other samples are positive: {count_bc_positive_t2_negative_other_positive}")
        print(f"Total count of episodes where BC is positive, T2 is negative, and other samples are negative: {count_bc_positive_t2_negative_other_negative}")
        print(f"|------Total both T2 and BC positive: {both_positives}")
        print(f"Total count of episodes where both T2 and BC are positive and no other samples were taken: {count_both_positive_other_empty}")
        print(f"Total count of episodes where both T2 and BC are positive and other samples are negative: {count_both_positive_other_negative}")
        print(f"Total count of episodes where both T2 and BC are positive and other samples are positive: {count_both_positive_other_positive}")

        
        # Call the Chi-Squared function here
        label = "T2 vs BC with Other Samples Positive"
        su.perform_chi2_test(count_t2_positive_bc_negative_other_positive, count_t2_positive_bc_negative_other_positive + count_t2_positive_bc_negative_other_negative, 
        count_bc_positive_t2_negative_other_positive, count_bc_positive_t2_negative_other_positive + count_bc_positive_t2_negative_other_negative, label)

    def calculate_sensitivity_specificity(self, include_other=True):
        TP, FP, TN, FN = 0, 0, 0, 0
    
        for episode in self.episodes.values():
            t2_result = episode.t2_isolates
            bc_result = episode.bc_isolates_in_panel
            other_result = episode.other_sample_isolates_in_panel
    
            combined_bc_other = bc_result.union(other_result)
            comparison = combined_bc_other if include_other else bc_result
    
            if t2_result:
                if t2_result.intersection(comparison):
                    TP += 1  # True Positive: common element
                else:
                    FP += 1  # False Positive: Mismatch in isolates
            else:
                if bc_result:
                    FN += 1  # False Negative: T2 negative, BC positive
                else:
                    TN += 1  # True Negative: All tests negative
    
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    
        print(f"True Positives (TP): {TP}\n"
          f"False Positives (FP): {FP}\n"
          f"True Negatives (TN): {TN}\n"
          f"False Negatives (FN): {FN}\n"
          f"Sensitivity: {sensitivity:.2f}\n"
          f"Specificity: {specificity:.2f}\n"
          f"PPV: {ppv:.2f}\n"
          f"NPV: {npv:.2f}\n"
                                            )  

    def calculate_sensitivity_specificity_bacteria(self, include_other=True):
        
        # Add 'Detected by Both' to the dictionary
        stats = {bacteria: {'T2 Detected': 0, 'BC Detected': 0, 'Detected by Both': 0, 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for bacteria in T2_LIST}
        total_stats = {'T2 Detected': 0, 'BC Detected': 0, 'Detected by Both': 0, 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        
        for episode in self.episodes.values():
            
            detected_by_t2_names = {isolate.name for isolate in episode.t2_isolates}
            if include_other:
                detected_by_comparison = {isolate.name for isolate in episode.bc_isolates_in_panel.union(episode.other_sample_isolates_in_panel)}
            else:
                detected_by_comparison = {isolate.name for isolate in episode.bc_isolates_in_panel}
        
            for bacteria in T2_LIST:
                bacteria_detected_by_bc = bacteria in {isolate.name for isolate in episode.bc_isolates_in_panel}
                bacteria_detected_by_t2 = bacteria in detected_by_t2_names
        
                if bacteria_detected_by_bc:
                    stats[bacteria]['BC Detected'] += 1
                    total_stats['BC Detected'] += 1
                if bacteria_detected_by_t2:
                    stats[bacteria]['T2 Detected'] += 1
                    total_stats['T2 Detected'] += 1
        
                if bacteria_detected_by_t2 and bacteria_detected_by_bc:
                    stats[bacteria]['Detected by Both'] += 1
                    total_stats['Detected by Both'] += 1
        
                if bacteria_detected_by_t2:
                    if bacteria in detected_by_comparison:
                        stats[bacteria]['TP'] += 1
                        total_stats['TP'] += 1
                    else:
                        stats[bacteria]['FP'] += 1
                        total_stats['FP'] += 1
                else:
                    if bacteria_detected_by_bc:
                        stats[bacteria]['FN'] += 1
                        total_stats['FN'] += 1
                    else:
                        stats[bacteria]['TN'] += 1
                        total_stats['TN'] += 1
        
        results = {}
        for bacteria, counts in stats.items():
            T2_detected, BC_detected, detected_by_both, TP, FP, TN, FN = counts.values()
            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            specificity = TN / (TN + FP) if TN + FP > 0 else 0
            PPV = TP / (TP + FP) if TP + FP > 0 else 0
            NPV = TN / (TN + FN) if TN + FN > 0 else 0
            
            results[bacteria] = {
                'T2 Detected': T2_detected, 'BC Detected': BC_detected,
                'Detected by Both': detected_by_both, 'TP': TP, 'FP': FP,
                'TN': TN, 'FN': FN, 'Sensitivity': sensitivity,
                'Specificity': specificity, 'PPV': PPV, 'NPV': NPV
            }
        
        # Add the totals for 'Detected by Both'
        T2_detected_total, BC_detected_total, detected_by_both_total, TP, FP, TN, FN = total_stats.values()
        total_sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
        total_specificity = TN / (TN + FP) if TN + FP > 0 else 0
        total_PPV = TP / (TP + FP) if TP + FP > 0 else 0
        total_NPV = TN / (TN + FN) if TN + FN > 0 else 0
        
        results['Total'] = {
            'T2 Detected': T2_detected_total, 'BC Detected': BC_detected_total,
            'Detected by Both': detected_by_both_total, 'TP': TP, 'FP': FP,
            'TN': TN, 'FN': FN, 'Sensitivity': total_sensitivity,
            'Specificity': total_specificity, 'PPV': total_PPV, 'NPV': total_NPV
        }
        
        # Convert results to DataFrame and display
        df = pd.DataFrame.from_dict(results, orient='index')
        pd.options.display.float_format = '{:,.2f}'.format
        display(df)


    def tat(self, clinic_regex = None): ##get and print turnaround times
        
        BC_total_time, BC_time_to_prel_report, BC_time_to_arrival, T2_total_time, T2_time_to_prel_report, T2_time_to_arrival = calculate_time_metrics(self.episodes, clinic_regex)

        print("Comparative Statistics: BC Samples vs T2 Samples\n")

        comparisons = [("Total Time", BC_total_time, T2_total_time),
                       ("Time to Preliminary Report", BC_time_to_prel_report, T2_time_to_prel_report),
                       ("Time to Arrival", BC_time_to_arrival, T2_time_to_arrival)]

        for title, BC_data, T2_data in comparisons:
            print(f"{title} (BC Count: {len(BC_data)}, T2 Count: {len(T2_data)}):")
            print(f"  - BC Samples: {su.calculate_descriptives(BC_data)}")
            print(f"  - T2 Samples: {su.calculate_descriptives(T2_data)}")
            print(f"  - {su.perform_mann_whitney(BC_data, T2_data)}\n")
                
    
    def tat_differences(self, limit: int, time_to_analyse: str): 
        quick_samples = []
        slow_samples = []
    
        # Assign the appropriate attribute based on time_to_analyse
        if time_to_analyse == "ARRIVAL":
            get_time = lambda episode: episode.t2_sample.time_to_arrival
        elif time_to_analyse == "ANALYSIS":
            get_time = lambda episode: episode.t2_sample.time_to_prel_report
        elif time_to_analyse == "TOTAL":
            get_time = lambda episode: episode.t2_sample.total_time
        else:
            raise ValueError("Invalid value for time_to_analyse")
    
        for episode in self.episodes.values():
            time = get_time(episode)
            if time is not None:
                time_in_seconds = time.total_seconds()
                if time_in_seconds < limit * 3600:
                    quick_samples.append(episode.t2_sample)
                elif time_in_seconds > limit * 3600:
                    slow_samples.append(episode.t2_sample)
     
        weekend_sample_slow = 0
        karolinska_sample_slow = 0
        weekend_sample_quick = 0
        karolinska_sample_quick = 0
        
        # Loop through slow_samples
        for sample in slow_samples:
            print("-----------------------SLOW-----------------------")
            sample.display()
            print(f"Time to arrival: {sample.time_to_arrival}, Time to prel report: {sample.time_to_prel_report}, Total time: {sample.total_time}")
            if sample.arrival_date.weekday() in [4, 5, 6]:
                weekend_sample_slow += 1
            if re.search(r'\bkarolinska h\b', sample.clinic, re.IGNORECASE) and not re.search(r'\bkarolinska s\b', sample.clinic, re.IGNORECASE):
                karolinska_sample_slow += 1
        
        # Loop through quick_samples
        for sample in quick_samples:
            print("-----------------------QUICK-----------------------")
            sample.display()
            print(sample.time_to_arrival)
            print(sample.time_to_prel_report)
            print(sample.total_time)
            if sample.arrival_date.weekday() in [4, 5, 6]:
                weekend_sample_quick += 1
            if re.search(r'\bkarolinska h\b', sample.clinic, re.IGNORECASE) and not re.search(r'\bkarolinska s\b', sample.clinic, re.IGNORECASE):
                karolinska_sample_quick += 1
        
        # Calculate proportions
        total_slow_samples = len(slow_samples)
        total_quick_samples = len(quick_samples)
        
        weekend_proportion_slow = weekend_sample_slow / total_slow_samples if total_slow_samples > 0 else 0
        karolinska_proportion_slow = karolinska_sample_slow / total_slow_samples if total_slow_samples > 0 else 0
        
        weekend_proportion_quick = weekend_sample_quick / total_quick_samples if total_quick_samples > 0 else 0
        karolinska_proportion_quick = karolinska_sample_quick / total_quick_samples if total_quick_samples > 0 else 0

        # Output the counts and proportions
        print(f"Weekend samples in slow_samples: {weekend_sample_slow} ({weekend_proportion_slow:.2f})")
        print(f"Karolinska H samples in slow_samples: {karolinska_sample_slow} ({karolinska_proportion_slow:.2f})")
        print(f"Weekend samples in quick_samples: {weekend_sample_quick} ({weekend_proportion_quick:.2f})")
        print(f"Karolinska H samples in quick_samples: {karolinska_sample_quick} ({karolinska_proportion_quick:.2f})")
        

         # Print or store the chi2 statistics and p-values
        su.perform_chi2_test(weekend_sample_slow, total_slow_samples, weekend_sample_quick, total_quick_samples, "Weekend samples")
        su.perform_chi2_test(karolinska_sample_slow, total_slow_samples, karolinska_sample_quick, total_quick_samples, "Karolinska samples")
        
    def tat_df(self):    
        tat_df = calculate_time_metrics_to_df(self.episodes)
        tat_df.to_excel("till_patrik.xlsx", engine='openpyxl', index=False)
        print(f"Saved times from {len(self.episodes)} episodes to excel")

    def count_polymicrobials(self, switch):
        count = 0
        attribute = ""
    
        if switch == "t2":
            attribute = "t2_isolates"
        elif switch == "bc":
            attribute = "bc_isolates"
        elif switch == "bc_in_panel":
            attribute = "bc_isolates_in_panel"
        else:
            print("wrong input")
            return
    
        for episode in self.episodes.values():
            if len(getattr(episode, attribute)) > 1:
                count += 1
                episode.display()
    
        return count
                    