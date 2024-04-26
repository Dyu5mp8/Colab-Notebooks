from models import Isolate, Sample, BC_Sample, T2_Sample, Episode, Patient
from collections import Counter
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import subprocess



class Plotter:
    
    def __init__(self, patients, samples, episodes = {}):
        self.patients = patients
        self.episodes = episodes 
        self.samples = samples


    def saveplot_tofile(fig, name, format, path='/Users/davidyu/Library/CloudStorage/GoogleDrive-vichy576@gmail.com/Min enhet/Forskning/Annas grejer/T2 candida/T2 bakt/Manuskript/Figures/'):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(format)
        if format == "svg":
        # Save the figure using the date and time in the filename
            fig.savefig(f"{path}{name}_{current_datetime}.svg")

        elif format == "pdf":
            fig.savefig(f"{path}{name}_{current_datetime}.pdf")
    
        # Open the directory in the file explorer
        if os.name == 'nt':  # For Windows
            os.startfile(path)
        elif os.name == 'posix':  # For macOS and Linux
            subprocess.run(['open', path])