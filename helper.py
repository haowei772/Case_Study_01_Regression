import pandas as pd
import numpy as np

def normalize_MachineHoursCurrentMeter2(df1):
    df = df1.copy()
    mh0 = df['MachineHoursCurrentMeter']==0
    df['MachineHoursCurrentMeter'][mh0] = np.nan
    m1 = df['MachineHoursCurrentMeter'].mean()
    df['MachineHoursCurrentMeter'] = df['MachineHoursCurrentMeter'].fillna(m1)
    df['MachineHoursCurrentMeter'] = np.log10(np.array(df['MachineHoursCurrentMeter']))
    return df
