import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, expon, kstest
import matplotlib.pyplot as plt
import seaborn as sns


def perform_mann_whitney(list1, list2):
    # Remove NaN values if any
    clean_list1 = [x for x in list1 if not np.isnan(x)]
    clean_list2 = [x for x in list2 if not np.isnan(x)]
    
    # Perform Mann-Whitney U test
    if clean_list1 and clean_list2:
        _, p_val = mannwhitneyu(clean_list1, clean_list2, alternative='two-sided')
        return f"Mann-Whitney p-value: {p_val}"
    else:
        return "Mann-Whitney test not applicable (one or both lists are empty after removing NaNs)"


def calculate_descriptives(data_list):
    if not data_list:
        return "No data available"
    clean_data_list = [x for x in data_list if not np.isnan(x)]
    if not clean_data_list:
        return "No data available after removing NaNs"
    mean = np.mean(clean_data_list)
    median = np.median(clean_data_list)
    iqr = stats.iqr(clean_data_list)
    sd = np.std(clean_data_list)
    return f"Mean: {mean}, Median: {median}, IQR: {iqr}, SD: {sd}"


def perform_chi2_test(count_of_interest_group1: int, total_count_group1: int, 
                      count_of_interest_group2: int, total_count_group2: int, label: str) -> None:
    """
    Perform a Chi-Square test between two groups and print the results.

    Parameters:
    count_of_interest_group1 (int): Count of the feature of interest in the first group
    total_count_group1 (int): Total count in the first group
    count_of_interest_group2 (int): Count of the feature of interest in the second group
    total_count_group2 (int): Total count in the second group
    label (str): A label to describe what the groups represent

    Returns:
    None: Prints the Chi2 statistics and p-values
    """
    # Create a contingency table
    contingency_table = [[count_of_interest_group1, total_count_group1 - count_of_interest_group1],
                         [count_of_interest_group2, total_count_group2 - count_of_interest_group2]]
    
    # Perform the Chi-Square Test
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
    
    # Print or store the chi2 statistics and p-values
    print(f"Chi2 Stat for {label}: {chi2_stat}, P-value: {p_val}")


# Modified function to check for normality for a given data list and label
    
def check_normality(data, label):
    # Visual Methods
    ## Histogram
    plt.figure()
    plt.hist(data, bins=10, alpha=0.5, label=label)
    plt.legend(loc='upper right')
    plt.title(f'Histogram for {label} Data')
    plt.show()
    
    ## QQ-Plot
    plt.figure()
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'QQ-Plot for {label} Data')
    plt.show()
    
    # Statistical Tests
    ## Shapiro-Wilk Test
    shapiro_result = stats.shapiro(data)
    print(f"Shapiro-Wilk Test for {label} Data: {shapiro_result}")
    
    ## Kolmogorov-Smirnov Test
    ks_result = stats.kstest(data, 'norm')
    print(f"Kolmogorov-Smirnov Test for {label} Data: {ks_result}")
    
    ## Anderson-Darling Test
    anderson_result = stats.anderson(data)
    print(f"Anderson-Darling Test for {label} Data: {anderson_result}")
    
def check_exponential(your_data, label):
    # Fit the data to an exponential distribution
    params = expon.fit(your_data)
    
    # Extract the scale parameter (lambda) from the fit
    _, scale = params
    
    # Generate the x values for plotting the fitted distribution
    x = np.linspace(min(your_data), max(your_data), 1000)
    
    # Generate the y values based on the x values and fitted parameters
    pdf_fitted = expon.pdf(x, loc=0, scale=scale)
    
    # Plot the histogram of your data
    plt.hist(your_data, bins=20, density=True, alpha=0.5, label='Data')
    
    # Plot the fitted exponential distribution
    plt.plot(x, pdf_fitted, 'r-', label=f'Fitted Exponential (scale={scale:.2f})')
    ks_result = kstest(your_data, 'expon', args=(0, scale))

    print(f"Kolmogorov-Smirnov test result: {ks_result}")
    
    plt.legend()
    plt.title(f'Exponential Fit for {label}')
    plt.show()

