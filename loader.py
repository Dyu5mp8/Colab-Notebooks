### Loads patients from a data source 
### Important: needs to be rewritten for each new data source
### Plan to implement a more general version of this

import pandas as pd
from models import Isolate, Sample, BC_Sample, T2_Sample, Other_Sample, Episode, Patient
from bactdict import OTHER_SAMPLES

def from_excel(path):
    df = pd.read_excel(path)
    return df


def patients_list(dfs, df_times = None, df_times_bact_id = None):
    
    patients = {}
    
    time_data = {}

    time_id_data = {} ##separate time file for ID for BCs.

    if df_times is not None:
        time_data = pd.concat(df_times).set_index('ProvID').to_dict(orient='index')
    if df_times_bact_id is not None:
        time_id_data = df_times_bact_id.set_index('id').to_dict(orient='index')


    
    def create_patient(row):
        return Patient(
            id=row['ID nyckel'],
            gender=row['Kön'],
            age=row['Ålder']
        )
    
    
    def create_sample(row):
        """
        Create a sample object based on the content of the row.
        
        Args:
            row (dict): A dictionary containing all the necessary fields.
            
        Returns:
            A sample object, either of type BC_Sample, T2_Sample, or Other_sample.
        """
        try:
            sample_series = row['Provserie']
            args = {
                'id': row['ProvID'],
                'clinic': row['Avsändare'],
                'type': row['Undersökning'],
                'material': row['Provmaterial'],
                'locale': row['Specification'],
                'anamnesis': row['AnamnesisText'],
                'sample_date': pd.to_datetime(row['Provtagnig']),
                'arrival_date': row['Ankomst'],
                'prel_report_date': row['Prel. svar'],
                'final_report_date': row['Slutsvar'],
            }
        except KeyError as e:
            raise KeyError(f"Missing key in row: {e}")
    
        if sample_series == 'AP':
            return BC_Sample(**args)
        elif sample_series == 'CC':
            if row['Undersökning'] == 'Molekylär sepsisdiagnostik':
                return T2_Sample(**args)
            else:
                return Other_Sample(**args)
        elif sample_series in OTHER_SAMPLES:
            return Other_Sample(**args)
        else:
            raise ValueError("ERROR: Sample series not included")

    
    for df in dfs:    
        for index, row in df.iterrows():
            patient_id = row['ID nyckel']
            sample_id = row['ProvID']
            if patient_id not in patients:
                patient = create_patient(row)
                patients[patient_id] = patient  
            else:
                patient = patients[patient_id]
        
            if sample_id not in patient.samples:
                sample = create_sample(row)
                
                if time_data and sample_id in time_data:
                    sample.sample_date = pd.to_datetime(time_data[sample_id].get('PrDtm', None))
                    sample.arrival_date = pd.to_datetime(time_data[sample_id].get('Inreg Mikro', None))
                    sample.prel_report_date = pd.to_datetime(time_data[sample_id].get('Prel svarstid (1:a)', None))
                    sample.final_report_date = pd.to_datetime(time_data[sample_id].get('Slutsvarstid (1:a)', None))
            
  
                if time_id_data and sample_id in time_id_data:
                    sample.prel_id_report_date = pd.to_datetime(time_id_data[sample_id].get('timestamp'), errors='coerce')
                    
                
                patient.add_sample(sample)
        
            else:
                sample = patient.samples[sample_id]
                
            
            if isinstance(sample, (BC_Sample, Other_Sample)):
                if not pd.isna(row['Bakterie']):
                    sample.add_isolate(row['Bakterie'])
            elif isinstance(sample, T2_Sample):
                if not pd.isna(row['Analys']): 
                    if pd.isna(row['Bedömning']):
                        sample.add_invalid_T2(row['Analys'])
                    else:
                        sample.add_isolate(row['Analys'], row['Bedömning'])
                
    print(f"Total of {len(patients)} patients")
    return patientsimport pandas as pd
from models import Isolate, Sample, BC_Sample, T2_Sample, Other_Sample, Episode, Patient
from bactdict import OTHER_SAMPLES

def from_excel(path):
    df = pd.read_excel(path)
    return df


def patients_list(dfs, df_times = None, df_times_bact_id = None):
    
    patients = {}
    
    time_data = {}

    time_id_data = {} ##separate time file for ID for BCs.

    if df_times is not None:
        time_data = pd.concat(df_times).set_index('ProvID').to_dict(orient='index')
    if df_times_bact_id is not None:
        time_id_data = df_times_bact_id.set_index('id').to_dict(orient='index')


    
    def create_patient(row):
        return Patient(
            id=row['ID nyckel'],
            gender=row['Kön'],
            age=row['Ålder']
        )
    
    
    def create_sample(row):
        """
        Create a sample object based on the content of the row.
        
        Args:
            row (dict): A dictionary containing all the necessary fields.
            
        Returns:
            A sample object, either of type BC_Sample, T2_Sample, or Other_sample.
        """
        try:
            sample_series = row['Provserie']
            args = {
                'id': row['ProvID'],
                'clinic': row['Avsändare'],
                'type': row['Undersökning'],
                'material': row['Provmaterial'],
                'locale': row['Specification'],
                'anamnesis': row['AnamnesisText'],
                'sample_date': pd.to_datetime(row['Provtagnig']),
                'arrival_date': row['Ankomst'],
                'prel_report_date': row['Prel. svar'],
                'final_report_date': row['Slutsvar'],
            }
        except KeyError as e:
            raise KeyError(f"Missing key in row: {e}")
    
        if sample_series == 'AP':
            return BC_Sample(**args)
        elif sample_series == 'CC':
            if row['Undersökning'] == 'Molekylär sepsisdiagnostik':
                return T2_Sample(**args)
            else:
                return Other_Sample(**args)
        elif sample_series in OTHER_SAMPLES:
            return Other_Sample(**args)
        else:
            raise ValueError("ERROR: Sample series not included")

    
    for df in dfs:    
        for index, row in df.iterrows():
            patient_id = row['ID nyckel']
            sample_id = row['ProvID']
            if patient_id not in patients:
                patient = create_patient(row)
                patients[patient_id] = patient  # Add new patient to the dictionary
            else:
                patient = patients[patient_id]
        
            if sample_id not in patient.samples:
                sample = create_sample(row)
                
                if time_data and sample_id in time_data:
                    sample.sample_date = pd.to_datetime(time_data[sample_id].get('PrDtm', None))
                    sample.arrival_date = pd.to_datetime(time_data[sample_id].get('Inreg Mikro', None))
                    sample.prel_report_date = pd.to_datetime(time_data[sample_id].get('Prel svarstid (1:a)', None))
                    sample.final_report_date = pd.to_datetime(time_data[sample_id].get('Slutsvarstid (1:a)', None))
            
  
                if time_id_data and sample_id in time_id_data:
                    sample.prel_id_report_date = pd.to_datetime(time_id_data[sample_id].get('timestamp'), errors='coerce')
                    
                
                patient.add_sample(sample)
        
            else:
                sample = patient.samples[sample_id]
                
            
            if isinstance(sample, (BC_Sample, Other_Sample)):
                if not pd.isna(row['Bakterie']):
                    sample.add_isolate(row['Bakterie'])
            elif isinstance(sample, T2_Sample):
                if not pd.isna(row['Analys']): 
                    if pd.isna(row['Bedömning']):
                        sample.add_invalid_T2(row['Analys'])
                    else:
                        sample.add_isolate(row['Analys'], row['Bedömning'])
                
    print(f"Total of {len(patients)} patients")
    return patients