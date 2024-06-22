CONTAMINANT_LIST = {
    'Lactobacillus species', 'Cutibacterium (Propionibacterium) acnes', 
    'Staphylococcus epidermidis', 'Micrococcus species', 
    'Koagulas-negativ stafylokock (KNS)', 'Grampositiv stav', 
    'Bacillus cereus', 'Corynebacterium amycolatum', 'Corynebacterium species', 'Corynebacterium striatum', 'Aerococcus viridans'
}

T2_LIST = [
    'Enterococcus faecium',
    'Escherichia coli',
    'Klebsiella pneumoniae',
    'Acinetobacter baumannii',
    'Staphylococcus aureus',
    'Pseudomonas aeruginosa'
]



T2_NORMALIZED_NAMES = {
    'Enterococcus faecium-DNA, T2': 'Enterococcus faecium',
    'Escherichia coli-DNA, T2': 'Escherichia coli',
    'Klebsiella pneumoniae-DNA, T2': 'Klebsiella pneumoniae',
    'Acinetobacter baumannii-DNA, T2': 'Acinetobacter baumannii',
    'Staphylococcus aureus-DNA, T2': 'Staphylococcus aureus',
    'Pseudomonas aeruginosa-DNA, T2': 'Pseudomonas aeruginosa'
}

CANDIDA_LIST = {
    'Candida albicans': 'Candida albicans/glabrata',
    'Candida glabrata': 'Candida albicans/glabrata',
    'Candida krusei': 'Candida krusei/tropicalis',
    'Candida tropicalis': 'Candida krusei/tropicalis'
}


BC_NORMALIZED_NAMES = {
    'Escherichia coli ESBL': 'Escherichia coli',
    'Klebsiella pneumoniae ESBL': 'Klebsiella pneumoniae',
    'Klebsiella pneumoniae ESBL-CARBA': 'Klebsiella pneumoniae',
    'Klebsiella variicola' : 'Klebsiella pneumoniae',
    'Staphylococcus aureus MRSA': 'Staphylococcus aureus',
    'Acinetobacter species, non-baumannii gruppen': 'Acinetobacter non-baumannii',
    'Klebsiella pneumoniae-CARBA' : 'Klebsiella pneumoniae'
    'Acinetobacter baumannii-gruppen : Acinetobacter baumannii',
    'Klebsiella variicola ESBL-CARBA' : 'Klebsiella pneumoniae'
}

OTHER_SAMPLES = ['AC', 'AR', 'DA', 'DB', 'DC', 'DD', 'DQ']

SPECIMEN_CATEGORY_MAP = {
    'BAL': 'Lower respiratory tract',
    'Bronksekret': 'Lower respiratory tract',
    'Bronksekret, borste': 'Lower respiratory tract',
    'Sputum': 'Lower respiratory tract',
    'Trakealsekret': 'Lower respiratory tract',
    'Öronsekret': 'Ear/sinus',
    'Ascites': 'Deep wound/abcess/drainage',
    'Galla': 'Deep wound/abcess/drainage',
    'Dränagespets':'Deep wound/abcess/drainage',
    'Dränagevätska': 'Deep wound/abcess/drainage',
    'Urin': 'Urine',
    'Urin, KAD': 'Urine',
    'Urin, nefrostomi': 'Urine',
    'Vaginalsekret': 'Gynecological',
    'Cerebrospinalvätska': 'Cerebrospinal fluid',
    'Abscess': 'Deep wound/abcess/drainage',
    'Sårsekret': 'Superficial wound',
    'Benbit': 'Orthopedic',
    'Instickställe': 'Superficial wound',
    'Vävnad': 'Deep wound/abcess/drainage',
    'Konjunktivalsekret': 'Miscellaneous/Other',
    'Kornea': 'Miscellaneous/Other',
    'Ledvätska': 'Orthopedic',
    'Ortopediskt implantat': 'Orthopedic',
    'Kärlkateterspets': 'Vascular catheter',
    'Pleuravätska': 'Lower respiratory tract',
    'PD-vätska': 'Miscellaneous/Other',
    'Cervixsekret': 'Gynecological',
    'Bröstmjölk': 'Miscellaneous/Other',
    'Isolat': 'Miscellaneous/Other',
    'Sekret': 'Superficial wound',
    'Punktat': 'Deep wound/abcess/drainage',
    'Sinussekret': 'Ear/sinus',
    'Stamceller': 'Miscellaneous/Other',
    'Övrigt provmaterial': 'Miscellaneous/Other',
}
