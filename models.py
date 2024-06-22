from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Set
from datetime import datetime
from bactdict import CONTAMINANT_LIST, T2_NORMALIZED_NAMES, BC_NORMALIZED_NAMES, T2_LIST, SPECIMEN_CATEGORY_MAP, CANDIDA_LIST

@dataclass
class Isolate:
    name: str
    is_contaminant: bool = field(init=False)  # This field will be set later, so init=False
    is_t2included: bool = field(init=False) 
    
    def __post_init__(self):
        self.is_contaminant = self.name in CONTAMINANT_LIST
        self.is_t2included = self.name in T2_LIST

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Isolate):
            return False
        return CANDIDA_LIST.get(self.name, self.name) == CANDIDA_LIST.get(other.name, other.name)  # Assuming name uniqueness for the purpose of this example
    
    def display(self):
        print(f"\t\t\t |-- Isolate: {self.name}, {self.relevance()}, {self.t2_included()}")
    

    def relevance(self):
        if self.is_contaminant:  
            return "Contaminant"
        else:
            return "Relevant"

    def t2_included(self):
        if self.is_t2included:
            return "In T2 panel"
        else: 
            return "Not in T2 panel"


        
    
                
@dataclass
class Sample:
    id: int
    clinic: str
    type: str
    material: str
    locale: str
    anamnesis: str
    sample_date: datetime 
    arrival_date: datetime 
    prel_report_date: datetime 
    final_report_date: datetime 
    isolates: Set = field(default_factory=set)

    def display(self):
        print(f"\t\t |-- Sample Id {self.id}, date: {self.sample_date}, clinic: {self.clinic}, time to prel report: {self.total_time}")
        for isolate in self.isolates:
            isolate.display()

        
    @property
    def icu(self):
        import re
        if re.search(r'\bIVA\b|\bMIVA\b', self.clinic):
            return True
        elif re.search(r'intensiv', self.clinic, re.IGNORECASE):
            return True
        else:
            return False

    @property
    def time_to_arrival(self):
        if self.isolates and self.arrival_date and self.sample_date:
            return self.arrival_date - self.sample_date
        return None

    @property
    def time_to_prel_report(self):
        if self.isolates and self.prel_report_date and self.arrival_date:
            return self.prel_report_date - self.arrival_date
        return None

    @property
    def total_time(self):
        if self.isolates and self.prel_report_date and self.sample_date:
            return self.prel_report_date - self.sample_date
        return None
        
            
        
        
            
@dataclass
class BC_Sample(Sample):

    prel_id_report_date: datetime = None
    
    def add_isolate(self, name):
            normalized_name = self.normalize_name(name)
            isolate = Isolate(normalized_name)
            self.isolates.add(isolate)
    def get_all_isolates(self):
        return self.isolates

    def get_t2panel_isolates(self):
        isolates = {isolate for isolate in self.isolates if isolate.is_t2included}
        return isolates

    def get_contaminant_isolates(self):
        contaminant_isolates = {isolate for isolate in self.isolates if isolate.is_contaminant}
        return contaminant_isolates
   
    def get_relevant_isolates(self):
        contaminant_isolates = self.get_contaminant_isolates()
        relevant_isolates = self.isolates - contaminant_isolates
        return relevant_isolates

    def normalize_name(self, name):
        return BC_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found

    @property
    def time_to_arrival(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.arrival_date and self.sample_date:
            return self.arrival_date - self.sample_date
        return None
    
    @property
    def time_to_prel_report(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_report_date and self.arrival_date:
            return self.prel_report_date - self.arrival_date
        return None

    @property
    def time_to_prel_id_report(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_id_report_date and self.arrival_date:
            return self.prel_id_report_date - self.arrival_date
        return None
    
    @property
    def total_time(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_report_date and self.sample_date:
            return self.prel_report_date - self.sample_date
        return None

    @property
    def total_time_to_id(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_id_report_date and self.sample_date:
            return self.prel_id_report_date - self.sample_date
        return None



@dataclass
class T2_Sample(Sample):

    invalid_T2: Set = field(default_factory=set)
    
    def add_isolate(self, analysis, positivity):
        
        if positivity == '**POSITIV**':  # If the bacteria is found to be positive
            normalized_name = self.normalize_name(analysis)
            new_isolate = Isolate(normalized_name)
            self.isolates.add(new_isolate)
    
    def display(self):
        super().display()
        if self.invalid_T2:
            print(f"\t\t\t |-- invalid T2 results = {self.invalid_T2}")
        else:
            print("\t\t\t |-- All results valid")
            
    def normalize_name(self, name):
        return T2_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found

    def add_invalid_T2(self, analysis):
        self.invalid_T2.add(analysis)
        
    @property
    def time_to_prel_report(self):
        return self.final_report_date - self.arrival_date if self.isolates else None

    @property
    def time_to_prel_id_report(self):
        return self.final_report_date - self.arrival_date if self.isolates else None
    

    

    @property
    def total_time(self):
        return self.final_report_date - self.sample_date if self.isolates else None


@dataclass
class Other_Sample(Sample):
    
    def add_isolate(self, name):
            normalized_name = self.normalize_name(name)
            isolate = Isolate(normalized_name)
            self.isolates.add(isolate)
    
    def normalize_name(self, name):
        return BC_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found
    
    @property
    def categorized_locale(self):
        return SPECIMEN_CATEGORY_MAP.get(self.material, self.material)

    def display(self):
        super().display()
        print(f"\t\t\t |-- Specimen locale: {self.categorized_locale}")

    
    @property
    def time_to_prel_report(self):
        return None
        
    @property
    def total_time(self):
        return None
    
    
    
@dataclass
class Episode:
    id: int
    t2_sample: T2_Sample
    bc_samples: list = field(default_factory=list)
    other_samples: list = field(default_factory=list)
    ##set of isolates found
    #T2
    t2_isolates: set = field(default_factory=set)
    #BC (all isolates)
    bc_isolates: set = field(default_factory=set)
    #BC (only in panel)
    bc_isolates_in_panel: set = field(default_factory=set)
    #Contaminants
    bc_contaminants: set = field(default_factory=set)
    #Isolates in other samples if in panel
    other_sample_isolates: set = field(default_factory=set)
    other_sample_isolates_in_panel: set = field(default_factory=set)

    def __post_init__(self):
        self.t2_isolates = self.t2_sample.isolates
        
    def add_bc_sample(self, bc_sample):
        self.bc_samples.append(bc_sample)
        relevant_isolates = {isolate for isolate in bc_sample.isolates if not isolate.is_contaminant}
        contaminants = {isolate for isolate in bc_sample.isolates if isolate.is_contaminant}
        in_panel = {isolate for isolate in bc_sample.isolates if isolate.is_t2included}
        self.bc_isolates.update(relevant_isolates)
        self.bc_contaminants.update(contaminants)  
        self.bc_isolates_in_panel.update(in_panel)

    def add_other_sample(self, other_sample):
        self.other_samples.append(other_sample)
        self.other_sample_isolates.update(other_sample.isolates)
        
        in_panel = {isolate for isolate in other_sample.isolates if isolate.is_t2included}
        self.other_sample_isolates_in_panel.update(in_panel)

    def display(self):
        t2_isolate_names = [isolate.name for isolate in self.t2_isolates]
        bc_isolate_names = [isolate.name for isolate in self.bc_isolates]
        other_isolate_names = [isolate.name for isolate in self.other_sample_isolates]
                
        print(f"\033[1m\t |-- Episode # {self.id}, T2 date: {self.t2_sample.sample_date}, classification: {self.classify(compare_with = 'BC_IN_PANEL')}\033[0m")

        if t2_isolate_names:
            print(f"Isolates found in T2 samples: {', '.join(t2_isolate_names)}")
        if bc_isolate_names:
            print(f"Isolates found in BC samples: {', '.join(bc_isolate_names)}")
        if other_isolate_names:
            print(f"Isolates found in other samples: {', '.join(other_isolate_names)}")
            
        
        self.t2_sample.display()
        for sample in self.bc_samples:
            sample.display()
        for sample in self.other_samples:
            sample.display()

    def classify(self, compare_with: str = "ALL_BC") -> str:
        valid_categories = ["ALL_BC", "BC_IN_PANEL", "OTHER_IN_PANEL"]
        
        if compare_with not in valid_categories:
            raise ValueError(f"Invalid comparison category. Choose from {valid_categories}")
        
        t2 = bool(self.t2_isolates)

        # Initialize bc or other depending on the comparison category
        if compare_with == "ALL_BC":
            comparison_set = self.bc_isolates
        elif compare_with == "BC_IN_PANEL":
            comparison_set = self.bc_isolates_in_panel
        elif compare_with == "OTHER_IN_PANEL":
            comparison_set = self.other_sample_isolates_in_panel
        else:
            return "INVALID_COMPARISON_CATEGORY"
        
        comparison_bool = bool(comparison_set)
    
        if t2 and comparison_bool:
            if self.t2_isolates == comparison_set:
                return "BOTH_POS_MATCH"
            elif self.t2_isolates.issubset(comparison_set):
                return "T2_SUBSET_OF_COMPARISON_SET"
            elif comparison_set.issubset(self.t2_isolates):
                return "COMPARISON_SUBSET_OF_T2"
            else:
                return "BOTH_POS_NO_MATCH"
        elif t2 and not comparison_bool:
            return "T2_POS_COMPARISON_NEG"
        elif not t2 and comparison_bool:
            return "T2_NEG_COMPARISON_POS"
        elif not t2 and not comparison_bool:
            return "ALL_NEG"
        else:
            return "UNCLASSIFIED"

        

@dataclass
class Patient:
    id: int
    gender: str
    age: int
    samples: dict = field(default_factory=dict) 
    episodes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.samples is None:
            self.samples = {}

    def add_sample(self, sample):
        self.samples[sample.id] = sample

    def display_episodes(self):
        print(f"|-- Patient {self.id}, age {self.age}, gender {self.gender}")
        for episode in self.episodes.values():
            episode.display()


    def display(self):
        print(f"Patient {self.id}, age {self.age}, gender {self.gender}")
        for episode in self.episodes.values():
            episode.display()

    def add_episode(self, episode: Episode):
        self.episodes[episode.id] = episode

    def remove_episode(self, episode: Episode):
        self.episodes.remove(episode)
        
from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Set
from datetime import datetime
from bactdict import CONTAMINANT_LIST, T2_NORMALIZED_NAMES, BC_NORMALIZED_NAMES, T2_LIST, SPECIMEN_CATEGORY_MAP

@dataclass
class Isolate:
    name: str
    is_contaminant: bool = field(init=False)  # This field will be set later, so init=False
    is_t2included: bool = field(init=False) 
    
    def __post_init__(self):
        self.is_contaminant = self.name in CONTAMINANT_LIST
        self.is_t2included = self.name in T2_LIST

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Isolate):
            return False
        return self.name == other.name  # Assuming name uniqueness for the purpose of this example
    
    def display(self):
        print(f"\t\t\t |-- Isolate: {self.name}, {self.relevance()}, {self.t2_included()}")
    

    def relevance(self):
        if self.is_contaminant:  
            return "Contaminant"
        else:
            return "Relevant"

    def t2_included(self):
        if self.is_t2included:
            return "In T2 panel"
        else: 
            return "Not in T2 panel"


        
    
                
@dataclass
class Sample:
    id: int
    clinic: str
    type: str
    material: str
    locale: str
    anamnesis: str
    sample_date: datetime 
    arrival_date: datetime 
    prel_report_date: datetime 
    final_report_date: datetime 
    isolates: Set = field(default_factory=set)

    def display(self):
        print(f"\t\t |-- Sample Id {self.id}, date: {self.sample_date}, clinic: {self.clinic}, time to prel report: {self.total_time}")
        for isolate in self.isolates:
            isolate.display()

        
    @property
    def icu(self):
        import re
        if re.search(r'\bIVA\b|\bMIVA\b', self.clinic):
            return True
        elif re.search(r'intensiv', self.clinic, re.IGNORECASE):
            return True
        else:
            return False

    @property
    def time_to_arrival(self):
        if self.isolates and self.arrival_date and self.sample_date:
            return self.arrival_date - self.sample_date
        return None

    @property
    def time_to_prel_report(self):
        if self.isolates and self.prel_report_date and self.arrival_date:
            return self.prel_report_date - self.arrival_date
        return None

    @property
    def total_time(self):
        if self.isolates and self.prel_report_date and self.sample_date:
            return self.prel_report_date - self.sample_date
        return None
        
            
        
        
            
@dataclass
class BC_Sample(Sample):

    prel_id_report_date: datetime = None
    
    def add_isolate(self, name):
            normalized_name = self.normalize_name(name)
            isolate = Isolate(normalized_name)
            self.isolates.add(isolate)
    def get_all_isolates(self):
        return self.isolates

    def get_t2panel_isolates(self):
        isolates = {isolate for isolate in self.isolates if isolate.is_t2included}
        return isolates

    def get_contaminant_isolates(self):
        contaminant_isolates = {isolate for isolate in self.isolates if isolate.is_contaminant}
        return contaminant_isolates
   
    def get_relevant_isolates(self):
        contaminant_isolates = self.get_contaminant_isolates()
        relevant_isolates = self.isolates - contaminant_isolates
        return relevant_isolates

    def normalize_name(self, name):
        return BC_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found

    @property
    def time_to_arrival(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.arrival_date and self.sample_date:
            return self.arrival_date - self.sample_date
        return None
    
    @property
    def time_to_prel_report(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_report_date and self.arrival_date:
            return self.prel_report_date - self.arrival_date
        return None

    @property
    def time_to_prel_id_report(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_id_report_date and self.arrival_date:
            return self.prel_id_report_date - self.arrival_date
        return None
    
    @property
    def total_time(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_report_date and self.sample_date:
            return self.prel_report_date - self.sample_date
        return None

    @property
    def total_time_to_id(self):
        if any(isolate.is_t2included for isolate in self.isolates) and \
           self.prel_id_report_date and self.sample_date:
            return self.prel_id_report_date - self.sample_date
        return None



@dataclass
class T2_Sample(Sample):

    invalid_T2: Set = field(default_factory=set)
    
    def add_isolate(self, analysis, positivity):
        
        if positivity == '**POSITIV**':  # If the bacteria is found to be positive
            normalized_name = self.normalize_name(analysis)
            new_isolate = Isolate(normalized_name)
            self.isolates.add(new_isolate)
    
    def display(self):
        super().display()
        if self.invalid_T2:
            print(f"\t\t\t |-- invalid T2 results = {self.invalid_T2}")
        else:
            print("\t\t\t |-- All results valid")
            
    def normalize_name(self, name):
        return T2_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found

    def add_invalid_T2(self, analysis):
        self.invalid_T2.add(analysis)
        
    @property
    def time_to_prel_report(self):
        return self.final_report_date - self.arrival_date if self.isolates else None

    @property
    def time_to_prel_id_report(self):
        return self.final_report_date - self.arrival_date if self.isolates else None
    

    

    @property
    def total_time(self):
        return self.final_report_date - self.sample_date if self.isolates else None


@dataclass
class Other_Sample(Sample):
    
    def add_isolate(self, name):
            normalized_name = self.normalize_name(name)
            isolate = Isolate(normalized_name)
            self.isolates.add(isolate)
    
    def normalize_name(self, name):
        return BC_NORMALIZED_NAMES.get(name, name)  # Default to the original name if no match is found
    
    @property
    def categorized_locale(self):
        return SPECIMEN_CATEGORY_MAP.get(self.material, self.material)

    def display(self):
        super().display()
        print(f"\t\t\t |-- Specimen locale: {self.categorized_locale}")

    
    @property
    def time_to_prel_report(self):
        return None
        
    @property
    def total_time(self):
        return None
    
    
    
@dataclass
class Episode:
    id: int
    t2_sample: T2_Sample
    bc_samples: list = field(default_factory=list)
    other_samples: list = field(default_factory=list)
    ##set of isolates found
    #T2
    t2_isolates: set = field(default_factory=set)
    #BC (all isolates)
    bc_isolates: set = field(default_factory=set)
    #BC (only in panel)
    bc_isolates_in_panel: set = field(default_factory=set)
    #Contaminants
    bc_contaminants: set = field(default_factory=set)
    #Isolates in other samples if in panel
    other_sample_isolates: set = field(default_factory=set)
    other_sample_isolates_in_panel: set = field(default_factory=set)

    def __post_init__(self):
        self.t2_isolates = self.t2_sample.isolates
        
    def add_bc_sample(self, bc_sample):
        self.bc_samples.append(bc_sample)
        relevant_isolates = {isolate for isolate in bc_sample.isolates if not isolate.is_contaminant}
        contaminants = {isolate for isolate in bc_sample.isolates if isolate.is_contaminant}
        in_panel = {isolate for isolate in bc_sample.isolates if isolate.is_t2included}
        self.bc_isolates.update(relevant_isolates)
        self.bc_contaminants.update(contaminants)  
        self.bc_isolates_in_panel.update(in_panel)

    def add_other_sample(self, other_sample):
        self.other_samples.append(other_sample)
        self.other_sample_isolates.update(other_sample.isolates)
        
        in_panel = {isolate for isolate in other_sample.isolates if isolate.is_t2included}
        self.other_sample_isolates_in_panel.update(in_panel)

    def display(self):
        t2_isolate_names = [isolate.name for isolate in self.t2_isolates]
        bc_isolate_names = [isolate.name for isolate in self.bc_isolates]
        other_isolate_names = [isolate.name for isolate in self.other_sample_isolates]
                
        print(f"\033[1m\t |-- Episode # {self.id}, T2 date: {self.t2_sample.sample_date}, classification: {self.classify(compare_with = 'BC_IN_PANEL')}\033[0m")

        if t2_isolate_names:
            print(f"Isolates found in T2 samples: {', '.join(t2_isolate_names)}")
        if bc_isolate_names:
            print(f"Isolates found in BC samples: {', '.join(bc_isolate_names)}")
        if other_isolate_names:
            print(f"Isolates found in other samples: {', '.join(other_isolate_names)}")
            
        
        self.t2_sample.display()
        for sample in self.bc_samples:
            sample.display()
        for sample in self.other_samples:
            sample.display()

    def classify(self, compare_with: str = "ALL_BC") -> str:
        valid_categories = ["ALL_BC", "BC_IN_PANEL", "OTHER_IN_PANEL"]
        
        if compare_with not in valid_categories:
            raise ValueError(f"Invalid comparison category. Choose from {valid_categories}")
        
        t2 = bool(self.t2_isolates)

        # Initialize bc or other depending on the comparison category
        if compare_with == "ALL_BC":
            comparison_set = self.bc_isolates
        elif compare_with == "BC_IN_PANEL":
            comparison_set = self.bc_isolates_in_panel
        elif compare_with == "OTHER_IN_PANEL":
            comparison_set = self.other_sample_isolates_in_panel
        else:
            return "INVALID_COMPARISON_CATEGORY"
        
        comparison_bool = bool(comparison_set)
    
        if t2 and comparison_bool:
            if self.t2_isolates == comparison_set:
                return "BOTH_POS_MATCH"
            elif self.t2_isolates.issubset(comparison_set):
                return "T2_SUBSET_OF_COMPARISON_SET"
            elif comparison_set.issubset(self.t2_isolates):
                return "COMPARISON_SUBSET_OF_T2"
            else:
                return "BOTH_POS_NO_MATCH"
        elif t2 and not comparison_bool:
            return "T2_POS_COMPARISON_NEG"
        elif not t2 and comparison_bool:
            return "T2_NEG_COMPARISON_POS"
        elif not t2 and not comparison_bool:
            return "ALL_NEG"
        else:
            return "UNCLASSIFIED"

        

@dataclass
class Patient:
    id: int
    gender: str
    age: int
    samples: dict = field(default_factory=dict) 
    episodes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.samples is None:
            self.samples = {}

    def add_sample(self, sample):
        self.samples[sample.id] = sample

    def display_episodes(self):
        print(f"|-- Patient {self.id}, age {self.age}, gender {self.gender}")
        for episode in self.episodes.values():
            episode.display()


    def display(self):
        print(f"Patient {self.id}, age {self.age}, gender {self.gender}")
        for episode in self.episodes.values():
            episode.display()

    def add_episode(self, episode: Episode):
        self.episodes[episode.id] = episode

    def remove_episode(self, episode: Episode):
        self.episodes.remove(episode)
        
