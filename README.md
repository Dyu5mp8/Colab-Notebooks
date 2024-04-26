David Yu 2024-0426

# Source code for T2Bact study
Jupyter notebook file can be provided upon request
##Overview
Models, loader function for patients and partitioning into episodes

Patient, episodes are defined, and Isolates are classified in bactdict.py.

## loading

### create list of patients
loads excel file into pandas DataFrame, iterates the rows, to creates a dictionary of of patients that is populated with samples.

### partitioning into episodes. 
Partitions into episodes with prespecified time frames, exclusion criteria etc.

After this data is ready for analysis.

