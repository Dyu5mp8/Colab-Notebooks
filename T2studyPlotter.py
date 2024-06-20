from Plotter import Plotter
import numpy as np
from matplotlib_venn import venn2
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import ceil, floor
import pandas as pd
import seaborn as sns
import re
from utils import calculate_time_metrics
import stat_utils as su
from bactdict import T2_LIST
from matplotlib.colors import ListedColormap
from scipy.stats import mannwhitneyu
from matplotlib.ticker import MaxNLocator

    

class T2studyPlotter(Plotter):
     
    def show_time_between_BC_T2(self):
        time_differences = []  # to store the time differences between T2 and BC samples

        for episode in self.episodes.values():
            t2_time = episode.t2_sample.sample_date  # Replace with your actual attribute name for T2 time
            for bc_sample in episode.bc_samples:
                bc_time = bc_sample.sample_date  # Replace with your actual attribute name for BC time
                time_difference = abs(t2_time - bc_time)  # No more abs(), to keep sign
    
                
                # Convert the Timedelta to minutes (or hours, if you prefer)
                time_difference_in_minutes = time_difference.total_seconds() / 60.0
                
                time_differences.append(time_difference_in_minutes)
                
        print(len(time_differences))
        # Now, let's plot the distribution
        plt.hist(time_differences, bins=20, edgecolor='black')
        plt.title('Distribution of Time Differences Between T2 and BC Samples')
        plt.xlabel('Time Difference (minutes)')
        plt.ylabel('Frequency')
        plt.show()


    def plot_venn(self, compare_with):
        episode_classification_counter = Counter()
        valid_categories = ["ALL_BC", "BC_IN_PANEL", "OTHER_IN_PANEL"]
        if compare_with not in valid_categories:
            raise ValueError(f"Invalid comparison category. Choose from {valid_categories}")
        for episode in self.episodes.values():
            classification = episode.classify(compare_with)  # Or however you call it
            episode_classification_counter[classification] += 1
        # Intersection count
        
        intersection_keys = [
            "BOTH_POS_MATCH",
            "T2_SUBSET_OF_COMPARISON_SET",
            "COMPARISON_SUBSET_OF_T2",
            "BOTH_POS_NO_MATCH"
        ]
        
        intersection_count = sum(episode_classification_counter.get(key, 0) for key in intersection_keys)
    
                # Count of 'posT2_negBC'
        posT2_negBC = episode_classification_counter.get('T2_POS_COMPARISON_NEG', 0)
        
        # Count of 'posBC_negT2'
        posBC_negT2 = episode_classification_counter.get('T2_NEG_COMPARISON_POS', 0)
        
        # Count of 'both_positive'
        both_positive = intersection_count
        total = len(self.episodes)
        both_negative = total - (posBC_negT2 + posT2_negBC + both_positive)

        positives = total - both_negative

        
        plt.figure(figsize=(12, 12))
        
        # Create the Venn diagram again for demonstration
        venn_diagram = venn2(subsets=(posT2_negBC, posBC_negT2, both_positive),
                             set_labels=None,
                             set_colors=('#66c2a5', '#fc8d62'))
        
        # Hatching patterns and color of intersection
        venn_diagram.get_patch_by_id('10').set_hatch('////')
        venn_diagram.get_patch_by_id('01').set_hatch('\\\\\\\\')
        venn_diagram.get_patch_by_id('11').set_hatch('xxxx')
        intersection_color = '#8da0cb'  # Choose the desired color for the intersection
        venn_diagram.get_patch_by_id('11').set_color(intersection_color)
        venn_diagram.get_patch_by_id('11').set_alpha(0.3)  # Optional: set transparency

        

        for idx in ['10', '01', '11']:
            venn_diagram.get_patch_by_id(idx).set_edgecolor('black')
            venn_diagram.get_patch_by_id(idx).set_linewidth(2)
            venn_diagram.get_label_by_id(idx).set_text('')
        
        # Manually place labels and draw straight lines
        label_positions = np.array([[-0.7, 0.5], [0.7, 0.4], [0, -0.6]])
        labels = [f" T2 positive, BC negative: {posT2_negBC} ({(100*posT2_negBC/positives):.1f} %)", f"BC positive, T2 negative: {posBC_negT2} ({(100*posBC_negT2/positives):.1f} %)", f"Both Positive: {both_positive} ({(100*both_positive/positives):.1f} %)"]
        arrow_positions = np.array([[-0.5, 0.05], [0.45, 0.00], [0.05, -0.2]])
        
        # Adding space between arrows and labels: offset the text position slightly in the direction away from arrow
        offset = np.array([[0, 0.03], [0, 0.03], [0, -0.03]])  # Adjust these values as needed
        new_label_positions = label_positions + offset
        
        for label_pos, new_label_pos, label, arrow_pos in zip(label_positions, new_label_positions, labels, arrow_positions):
            plt.text(*new_label_pos, label, fontsize = 16, horizontalalignment='center', verticalalignment='center', color='black')

            plt.annotate('', xy=arrow_pos, xytext=label_pos,
                            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),  # Ensure arrow is black and slightly thicker
                            zorder=3)  # Ensure arrows are on top

        
        # Title and annotations
        plt.title('Comparison of Diagnostic Methods: T2 vs BC', fontsize=16, fontweight='bold', fontname='Arial')
        plt.annotate(f'Total Episodes: {total}', xy=(0.5, -0.43), xycoords='axes fraction', fontsize = 16, ha='center',fontname='Arial')
        plt.annotate(f'Total Positives: {positives}', xy=(0.5, -0.4), xycoords='axes fraction', fontsize = 16, ha='center',fontname='Arial')
        plt.annotate(f'Both Negative: {both_negative}', xy=(0.5, -0.46), xycoords='axes fraction', fontsize = 16, ha='center',fontname='Arial')
        plt.tight_layout()
        plot = plt.gcf()
        plt.show()
        return plot



    def plot_bacteria_occurrences(self):
        # Initialize the counters for each category
        bacteria_counts = {
            'T2_POS_BC_NEG': {bacteria: 0 for bacteria in T2_LIST},
            'T2_NEG_BC_POS': {bacteria: 0 for bacteria in T2_LIST},
            'BOTH_POS': {bacteria: 0 for bacteria in T2_LIST}
        }
    
        # Iterate over episodes
        for episode in self.episodes.values():
            # Classify each episode
            classification = episode.classify(compare_with='BC_IN_PANEL')
            
            # Update the counts for detected bacteria based on the classification
            if classification == 'T2_POS_COMPARISON_NEG':
                detected_bacteria = episode.t2_isolates
                category = 'T2_POS_BC_NEG'
            elif classification == 'T2_NEG_COMPARISON_POS':
                detected_bacteria = episode.bc_isolates_in_panel
                category = 'T2_NEG_BC_POS'
            else: # classification in ['BOTH_POS_MATCH', 'T2_SUBSET_OF_COMPARISON_SET', 'COMPARISON_SUBSET_OF_T2', 'BOTH_POS_NO_MATCH']
                detected_bacteria = episode.t2_isolates.union(episode.bc_isolates_in_panel)
                category = 'BOTH_POS'
            
            for isolate in detected_bacteria:
                if isolate.name in T2_LIST:
                    bacteria_counts[category][isolate.name] += 1
        
        # Plotting
             # Define a colormap to use different colors for each bar
        cmap = ListedColormap(['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])
        figs = []
        # Plotting
        for category, counts in bacteria_counts.items():
            # Skip if there are no occurrences in this category
            if not any(counts.values()):
                continue
            
            bacteria_names = list(counts.keys())
            occurrences = list(counts.values())
            bar_height = 0.1
            fig, ax = plt.subplots(figsize=(6, 3))  # Adjust for a compact figure
            y_positions = np.arange(len(bacteria_names)) * (0.2)  # Adjust the spacing here
            # 'bars' should be the container returned by 'barh'
            bars = ax.barh(y_positions, occurrences, height=bar_height, color=cmap.colors[:len(bacteria_names)], edgecolor='gray')
            

            # Rounding the bars
            for bar in bars:
                bar.set_linewidth(0.5)
                bar.set_edgecolor("black")
    
            ax.set_xlabel('Occurrences', fontsize = 14, fontweight = "bold")
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14) 
            ax.set_yticks(y_positions)
            # Format x-axis with integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures x-ticks are integers
            ax.set_yticklabels(bacteria_names, fontsize = 14, fontweight = "bold")
            ax.invert_yaxis()  # Invert the y-axis so the bars start from the top
            ax.set_xlim(0, max(occurrences) * 1.1)  # Set x-axis limit to give some padding for labels
            plt.grid(False)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        
            figs.append(fig)
            
        return figs

# Call this function from your class instance to plot the occurrences


    def plot_longitudinal(self, positive = True, save = False): 
        
        # Initialize lists
        episode_ids = []
        days = []
        colors = []
        
        # Counters to evenly distribute the lines
        counters = Counter()
        episode_ids_with_data = set()
        y_mapping = {}  # Mapping from episode_id to new y_value
        new_y_value = 0  # Start the new y_values from 0
        
        # The rest of the dummy data generation loop remains the same
        for episode in self.episodes.values():
            if positive:
                should_plot = episode.t2_isolates
            else:
                should_plot = not episode.t2_isolates
                
            if should_plot:
                t2_sample = episode.t2_sample
                cc_day = t2_sample.sample_date
                for sample in episode.bc_samples:
                    total_seconds_diff = (sample.sample_date - cc_day).total_seconds()
                    # Round to the nearest day, but keep it within the bounds [-1, 1], [-2, 2], etc.
                    # Convert total seconds to hours
                    hours_diff = total_seconds_diff / (60 * 60)
                    
                    # Calculate the day band it should belong to
                    day_diff = floor(hours_diff / 24)

                    days.append(day_diff)
                    episode_ids.append(episode.id)

                    # Add this episode_id to the set of episode_ids that have data
                    episode_ids_with_data.add(episode.id)
                    bc_t2panel_isolates = {iso for iso in sample.isolates if iso.is_t2included}                    
                    # Determine the color based on various conditions
                    if bc_t2panel_isolates - t2_sample.isolates:
                        color = 'red'  # Blue
                    elif sample.get_relevant_isolates() - t2_sample.isolates:
                        color = 'orange'  # Orange
                    elif sample.get_relevant_isolates() == t2_sample.isolates:
                        color = 'green'  # Green for Concordant
                    elif t2_sample.isolates - bc_t2panel_isolates:
                        color = 'black'  # Red for Missing Bacteria
                    else:
                        color = 'blue'  # Purple for Other
                                
                    colors.append(color)

        # Filtering unique episode_ids that have data
        unique_episode_ids_with_data = set(episode_ids) & episode_ids_with_data
        
        # Create a mapping from episode_ids to new y-values
        # Create a mapping from episode_ids to new y-values
        episode_id_to_y = {episode_id: index * 0.03 for index, episode_id in enumerate(sorted(unique_episode_ids_with_data))}
        
        
        num_unique_episodes = len(set(episode_ids))
        spacing = 0.01 / num_unique_episodes  # This will give you even spacing between lines based on the number of unique episode_ids
        
        for episode_id in set(episode_ids):
            y_mapping[episode_id] = new_y_value
            new_y_value += spacing
        
        
        # Now you'll use y_mapping to get the new y_value for each episode_id in your plotting code
        for unique_episode in unique_episode_ids_with_data:
            new_y = episode_id_to_y[unique_episode]
            plt.axhline(y=new_y, color='grey', linestyle='--', linewidth=0.5)
            plt.text(-3.25, new_y, str(unique_episode), fontsize=8, verticalalignment='center')
        
        
        
        # Create short horizontal lines for each sample
        for day, episode_id, color in zip(days, episode_ids, colors):
            new_y = y_mapping[episode_id]  # Get the new y_value for this episode_id
            
            # Count how many samples are for this day and episode
            total_for_day = sum((d == day) and (eid == episode_id) for d, eid in zip(days, episode_ids))
            
            # Calculate the space between each line within the day
            space = 1 / (total_for_day + 1)
    
            # Calculate the middle position for the "cell" that represents a 24-hour interval
            middle_x = day + 0.5  # Adding 0.5 so that it's centered in the middle of the cell

        
            # Get the current count for this specific day and episode
            count_for_day = counters[(day, episode_id)]
        
            # Calculate the position for this sample within the day
            start_x = middle_x + (count_for_day - total_for_day / 2) * space
        
            plt.hlines(y=episode_id_to_y[episode_id], xmin=start_x, xmax=start_x + 0.1, color=color, linewidth=3)
            counters[(day, episode_id)] += 1  # Update counter for the next line
            
            # Mark the boundaries for each 24-hour interval around the day
            plt.axvline(x=day, color='grey', linestyle=':', linewidth=0.5)
            plt.axvline(x=day + 1, color='grey', linestyle=':', linewidth=0.5)
            
                    
        # Add vertical line at day 0, extending it to the text label
        y_min = plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.03
        plt.axvline(x=0, color='purple', linestyle='--', linewidth=1, ymin=y_min, ymax=1)
        
        
        # Labels and title with label padding
        
        plt.xlabel('Days from T2 sampling', fontsize=16, labelpad=20)
        
        # Get current axis
        ax = plt.gca()
        
        # Move x-tick labels down
        ax.tick_params(axis='x', which='both', pad=30)
        ax.set_xticks(days)
        
        
        # Add text label under the vertical line at day 0
        ax.text(0.5, -0.04, 'Timepoint where T2 was sampled', fontsize=12, ha='center', va='bottom', transform=ax.transAxes, color ='purple')
        
        # Add horizontal arrows. Adjust the positions and sizes as needed.
        ax.annotate('', xy=(0.35, -0.11), xycoords='axes fraction', xytext=(0.2, -0.11),
                    arrowprops=dict(arrowstyle="<-", lw=1.5))
        ax.annotate('', xy=(0.8, -0.11), xycoords='axes fraction', xytext=(0.65, -0.11),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        
        plt.ylabel('Episode', fontsize = 16, labelpad = 20)
        plt.title('Distribution of BCs in relationship to positive T2 samples', pad = 15, fontsize = 20)
        # Legends
        if positive:
            legend_labels = [
                mpatches.Patch(color='green', label='Concordant results'),
                mpatches.Patch(color='black', label='BC did not detect T2 finding'),
                mpatches.Patch(color='red', label='BC detected additional bacteria within T2 repertoire'),
                mpatches.Patch(color='orange', label='BC detected additional bacteria, not within T2 repertoire'),
            ]

        else:
            legend_labels = [
                mpatches.Patch(color='green', label='BC negative'),
                mpatches.Patch(color='orange', label='BC detected bacteria, not within T2 repertoire'),   
                mpatches.Patch(color='red', label='BC detected bacteria within T2 repertoire'),
            ]
        
        plt.legend(handles=legend_labels, loc='upper right')
        y_max = max(episode_id_to_y.values())
        plt.ylim(-y_max * 0.05, y_max * 1.2)  # Here, 1.1 is to leave 10% extra space at the top.
        ax.set_yticks([])  # This removes the ticks
        ax.set_yticklabels([])  # This removes the tick labels
        # Make the plot larger
        if positive:
            plt.gcf().set_size_inches(15, 10)
        else:
            plt.gcf().set_size_inches(15, 50)
            
        plot = plt.gcf()
        return plot
        
        
        
    def tat_boxplot(self, clinic_regex = None):
        
        BC_total_time, BC_time_to_prel_report, BC_time_to_arrival, T2_total_time, T2_time_to_prel_report, T2_time_to_arrival = calculate_time_metrics(self.episodes, clinic_regex)

        metrics = [
            ('T2_total_time', T2_total_time),
            ('T2_time_to_prel_report', T2_time_to_prel_report),
            ('T2_time_to_arrival', T2_time_to_arrival),
            ('BC_total_time', BC_total_time),
            ('BC_time_to_prel_report', BC_time_to_prel_report),
            ('BC_time_to_arrival', BC_time_to_arrival),
        ]
        
        all_data = []

        titles = {
            'total_time': 'Total turn-around time',
            'time_to_prel_report': 'Time from arrival in laboratory to preliminary report',
            'time_to_arrival': 'Time from sampling to arrival in the laboratory'
        }

        # Loop through metrics and populate the all_data list
        for metric_name, metric_data in metrics:
            type_name = metric_name.split('_')[0]
            metric_type = '_'.join(metric_name.split('_')[1:])
            
            for d in metric_data:
                all_data.append({'Type': type_name, 'Metric': metric_type, 'Time_in_seconds': d})

        # Create a DataFrame from the all_data list
        df = pd.DataFrame(all_data)

        
        # Now plot
        figs = []
        outlier_info = {}

        for metric_type in ['total_time', 'time_to_prel_report', 'time_to_arrival']:
            fig = plt.figure(figsize=(8, 6))
            data_subset = df[df['Metric'] == metric_type]
            ax = sns.boxplot(
            x='Metric', 
            y='Time_in_seconds', 
            hue='Type', 
            dodge=0.2,
            data=data_subset, 
            width=0.4, 
            palette="Set2",
            showfliers=True, 
            fliersize= 3
            )
                   # Using describe() to print statistics for each type within the metric
            grouped_stats = data_subset.groupby('Type')['Time_in_seconds'].describe()
            print(f"\nStats for {metric_type}:")
            print(grouped_stats)


            ax.tick_params(axis='y', labelsize=14) 
            ax.set_xlabel('Method', fontsize=14, fontweight = 'bold', fontname = 'Arial')
            ax.set_xticks([-0.1, 0.1])
            ax.set_xticklabels(['T2','BC'], fontsize = 14, fontweight = 'bold', fontname = 'Arial')
            ax.set_ylabel('Time in hours', fontsize=14, fontweight = 'bold', fontname = 'Arial')
            ax.set_title(titles[metric_type], fontsize=16, fontweight = 'bold', fontname = 'Arial')
            ax.get_legend().remove()

            outliers_count = {}
            for type_category in ['T2', 'BC']:  # Assuming these are the only two categories
                type_data = data_subset[data_subset['Type'] == type_category]['Time_in_seconds']
                Q1 = np.percentile(type_data, 25)
                Q3 = np.percentile(type_data, 75)
                IQR = Q3 - Q1
                outlier_condition = ((type_data < (Q1 - 1.5 * IQR)) | (type_data > (Q3 + 1.5 * IQR)))
                outliers_count[type_category] = np.sum(outlier_condition)
            
            outlier_info[metric_type] = outliers_count
            print(f"Outliers for {metric_type} - T2: {outliers_count['T2']}, BC: {outliers_count['BC']}")
        
            plt.show()
            figs.append(fig)
        return figs
    
    def tat_distributions(self, clinic_regex = None): #Kernel Density Estimation and Histogram for times.
        
        BC_total_time, BC_time_to_prel_report, BC_time_to_arrival, T2_total_time, T2_time_to_prel_report, T2_time_to_arrival = calculate_time_metrics(self.episodes, clinic_regex)

        metrics_BC = [
            ('BC_total_time', BC_total_time),
            ('BC_time_to_prel_report', BC_time_to_prel_report),
            ('BC_time_to_arrival', BC_time_to_arrival),
        ]
        
        metrics_T2 = [
            ('T2_total_time', T2_total_time),
            ('T2_time_to_prel_report', T2_time_to_prel_report),
            ('T2_time_to_arrival', T2_time_to_arrival),
        ]
            
        for (bc_label, bc_data), (t2_label, t2_data) in zip(metrics_BC, metrics_T2):
            plt.figure(figsize=(10, 5))
            sns.kdeplot(bc_data, label=bc_label, color='blue')
            sns.kdeplot(t2_data, label=t2_label, color='green')  
            plt.title(f'Density Plot of {bc_label} and {t2_label}')
            plt.xlabel('Time (hours)')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    def check_distribution(self, test="NORMAL"):
        BC_total_time, BC_time_to_prel_report, BC_time_to_arrival, T2_total_time, T2_time_to_prel_report, T2_time_to_arrival  = calculate_time_metrics(self.episodes)

        metrics_BC = [
            ('BC_total_time', BC_total_time),
            ('BC_time_to_prel_report', BC_time_to_prel_report),
            ('BC_time_to_arrival', BC_time_to_arrival),
        ]
        
        metrics_T2 = [
            ('T2_total_time', T2_total_time),
            ('T2_time_to_prel_report', T2_time_to_prel_report),
            ('T2_time_to_arrival', T2_time_to_arrival),
        ]
     
        for metric_label, metric_data in metrics_BC + metrics_T2:

            if test == "NORMAL":
                su.check_normality(metric_data, metric_label)
            elif test == "EXP":
                su.check_exponential(metric_data, metric_label)
            else:
                raise ValueError("Invalid test type specified")
    

    def plot_tat_by_time(self, sample_type, time_type, result_category: 'positive'):
        
        data = []
    
        
        def add_sample_data(sample, time_type):
            nonlocal data
            sample_date = sample.sample_date
            weekday = sample_date.strftime("%A")  # Get the name of the day
            hour = sample_date.hour
            # Determine duty status
            duty_status = '8-16' if 8 <= hour < 16 else '16-08'
            if time_type == "all":
                turnaround_time = (sample.final_report_date-sample.sample_date).total_seconds() / 3600
            elif sample.time_to_prel_report:
                if time_type == "total":
                    turnaround_time = sample.total_time.total_seconds() / 3600  # Convert to hours
                elif time_type == "arrival":
                    turnaround_time = sample.time_to_arrival.total_seconds() / 3600  # Convert to hours
                elif time_type == "analysis":
                    turnaround_time = sample.time_to_prel_report.total_seconds() / 3600  # Convert to hours
            data.append({'weekday': weekday, 'duty_status': duty_status, 'turnaround_time': turnaround_time})
            #print({'weekday': weekday, 'duty_status': duty_status, 'turnaround_time': turnaround_time}) debug

                
    
        if sample_type == "t2":
            for episode in self.episodes.values():
                sample = episode.t2_sample
                #sample.display()
                if result_category == "positive":   
                    if sample.isolates:
                        add_sample_data(sample, "all")
                       #print("positive")
                elif result_category == "negative":
                    if not sample.isolates:
                        add_sample_data(sample, "all")
                else:
                    add_sample_data(sample, "all")
                
    
        if sample_type == "bc":
            for episode in self.episodes.values():
                for bc_sample in episode.bc_samples:
                    if any(isolate.is_t2included for isolate in bc_sample.isolates):
                        add_sample_data(bc_sample, time_type)
        # Create DataFrame
        df = pd.DataFrame(data)
        #print(df)
        # Aggregate data by weekday and duty status
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        # Group the data
                # Separate weekdays and weekends
        weekdays_df = df[df['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['turnaround_time']
        weekends_df = df[df['weekday'].isin(['Saturday', 'Sunday'])]['turnaround_time']

        # Perform Mann-Whitney U Test
        stat, p = mannwhitneyu(weekdays_df, weekends_df, alternative='two-sided')

        print("Mann-Whitney U test between weekdays and weekends:")
        print("U-statistic:", stat, "P-value:", p)
        print("Weekdays:", weekdays_df.describe())
        print("Weekends:", weekends_df.describe())

        # Separate data by duty status
        duty_8_16 = df[df['duty_status'] == '8-16']['turnaround_time']
        duty_16_08 = df[df['duty_status'] == '16-08']['turnaround_time']

        # Perform Mann-Whitney U Test
        stat, p = mannwhitneyu(duty_8_16, duty_16_08, alternative='two-sided')

        print("Mann-Whitney U test between '8-16' and '16-08' duty statuses:")
        print("U-statistic:", stat, "P-value:", p)
        print("'8-16' duty status:", duty_8_16.describe())
        print("'16-08' duty status:", duty_16_08.describe() )

    
        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        # Create a larger figure to accommodate the plots

        plt.figure(figsize=(10, 5))


        # Create the boxplot
        ax = sns.boxplot(x='weekday', y='turnaround_time', hue='duty_status', data=df, order=weekday_order, palette='Set3', width = 0.4, fliersize=3, dodge=True)

        # Enhance readability
        plt.xticks(rotation=45)
        plt.xlabel('Day of the week', fontsize=12, fontweight = 'bold', fontname = 'Arial')
        plt.ylabel('Turn-around time (hours)', fontsize=12, fontweight = 'bold', fontname = 'Arial')
        plt.title('T2 turn-around time by weekday and time of day', fontsize=14, fontweight = 'bold', fontname = 'Arial')
        plt.legend(title='Time of day', loc='upper right')
        fig = plt.gcf()
        # Display the plot
        plt.show()
        return fig
    
        
