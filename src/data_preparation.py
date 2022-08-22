# Purpose of script

import os
import numpy as np
import pandas as pd
from scipy import interpolate as interp

# INPUT DATA

# Move to raw data folder 
os.chdir('../data/input')

# Load the measured concentrations into a pandas dataframe
c_measured_raw = pd.read_csv('measured_concentrations_raw.csv')

# Define the stoichiometric matrix, X = B - A
A, B = [], []

with open('A.csv', 'r') as file:
    for line in file: 
        A.append([int(item) for item in line.strip().split(',')])

with open('B.csv', 'r') as file:
    for line in file: 
        B.append([int(item) for item in line.strip().split(',')])

A = np.asarray(A)
B = np.asarray(B)

X = B - A
# X = np.vstack((X, -X)) # stacks the stoic. coefficients of the reverse reactions (-X) onto that of the forward reactions (X)


# CLEANING and pH CONVERSION

# Set original length
n = len(c_measured_raw)

#  Turn all zeros (missing raw data) into NaN's, for IC in list of Interfering Compounds
columns = ['Fe'] # list of IC's
for ic in columns:
    for i in range(n):
        value = int(c_measured_raw.loc[i, [ic]].values)
        if value == 0:
            c_measured_raw.loc[i, [ic]] = np.nan

# Add empty column for to populate with [H+]
empty_column = np.zeros(n)
c_measured_raw['H+'] = empty_column

# Convert each pH to [H+]
for i in range(n):
    pH = c_measured_raw.loc[i, 'pH']
    c_measured_raw.loc[i, 'H+'] = 10**(-pH)

# Drop pH?

# Create an empty template like c_measured_raw - to be altered & populated
c_measured = pd.DataFrame(columns=c_measured_raw.columns)

# Loop through the Days and calculate delta(t) = Day(i+1) - Day(i).
# If N = Delta(t) > 1, insert N - 1 empty rows at index i.
for i in range(n):
    # Append raw entry to template dataframe
    entry = c_measured_raw.iloc[i,:]
    c_measured=c_measured.append(entry, ignore_index=True)
    if i < n - 1:
        # Calculate delta_t
        current_day = c_measured_raw.loc[i, ['Day']].values
        next_day = c_measured_raw.loc[i+1, ['Day']].values
        delta_t = next_day - current_day
    else: 
        break
    if delta_t > 1:
        # Insert n empty rows
        for _ in range(int(*delta_t)-1):
            c_measured=c_measured.append(pd.Series(), ignore_index=True)


# INTERPOLATION

# Define the independent variable - days
N = len(c_measured)
days = c_measured.loc[:, ['Day']].values
days = np.asarray(days[:, 0])

# Return interpolation on given column
def interp_column(dataframe, column_name, days):
    y = dataframe.loc[:, [column_name]].values
    y = np.asarray(y[:, 0])
    indices_to_delete = [] # ignore NaN's
    for i in range(len(y)):
        is_nan = np.isnan(y[i])
        if is_nan:
            indices_to_delete.append(i)
    # Delete NaN's
    y = np.delete(y, indices_to_delete) 
    days = np.delete(days, indices_to_delete)
    # Interpolate (and extrpolate)
    y_interp = interp.interp1d(days, y, fill_value='extrapolate')
    return y_interp

# Interpolate on all columns
column_names = c_measured.columns
for column_name in column_names: #.remove('Days'):
    y_interp = interp_column(c_measured, column_name, days)
    for i in range(N):
        value = c_measured.loc[i, [column_name]].values[0]
        if np.isnan(value):
            c_measured.loc[i, column_name] = y_interp(i)


# OUTPUT PROCESSED DATA

# Move to processed data folder
os.path.abspath('..')
os.makedirs('processed', exist_ok=True)

# Output processed data to .csv
c_measured.to_csv('processed/c_processed.csv')  


print(c_measured)