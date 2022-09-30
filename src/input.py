# PURPOSE OF THIS SCRIPT:
    # Input data (stoichiometric matrices & the measured concentrations)
    # Clean data (equalize time spacings, remove zeros, etc.)
    # Convert pH to [H+]
    # Normalize
    # Interpolate
    # Output prepared data (for validation)


# IMPORTS 

import os
import numpy as np
import pandas as pd
from scipy import interpolate as interp


# INPUT DATA

# Set base directory and move to 'raw' data folder
base_dir = os.getcwd()
os.chdir(os.path.join(base_dir, 'data', 'raw'))

# Load the measured concentrations into a pandas dataframe
c_measured = pd.read_csv('2_measured_states.csv')

# Define function to load .csv for the stoich. matrices and normalize each row
def load_csv(file_name):
    matrix = []
    with open(f'{file_name}.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                header = line.strip().split(',')
                continue
            row = [int(item) for item in line.strip().split(',')]
            row_max = max(row)
            row_normalized = np.array(row) / row_max
            matrix.append(row_normalized)
    matrix = np.asarray(matrix)
    return matrix, header

# Define the stoich. matrices A & B
A, A_header = load_csv('2_A')
B, B_header = load_csv('2_B')

# Define the stoich. matrix, X = B - A
X = B - A
X = np.vstack( (X, -X) ) # stacks the stoic. coefficients of the reverse reactions (-X) onto that of the forward reactions (X)
X = X.transpose() # such that # columns = # reactions & # rows = # species
A = np.vstack( (A, B) )

# Check species are consistent across A, B, and c_measured
c_header = c_measured.columns[1:]
N_c = len( A_header )
for i in range( N_c ):
    consistent = bool( c_header[i] == A_header[i] and c_header[i] == B_header[i] )
    if not consistent:
        print( f'Species mismatch on column {i}: ', c_header[i], A_header[i], B_header[i] )

# Store header indices for hidden states (prefixed 'H_')
hidden_states = []
for i, species in enumerate(A_header):
    prefix = species[0:2]
    # print(prefix)
    if prefix == 'H_':
        hidden_states.append(i)
# print('\n', hidden_states)

# DATA CLEANING and pH CONVERSION

# Number of rows in the dataframe of raw measurements
n = len(c_measured)

#  Turn all zeros (missing raw data) into NaN's, for IC in list of Interfering Compounds
columns = ['Fe'] # list of IC's
for ic in columns:
    if ic in c_measured.columns:
        for i in range(n):
            value = int(c_measured.loc[i, [ic]].values)
            if value == 0:
                c_measured.loc[i, [ic]] = np.nan

# Convert micro-grams into miligrams, for REE in list of Rare Earth Elements
columns = ['Sc_aqueous']
for ree in columns:
    for i in range(n):
        c_measured.loc[i, [ree]] *= 0.001 

# Add empty column to be populated with [H+]
# empty_column = np.zeros(n)
# c_measured['H+'] = empty_column

# Convert each pH to [H+], then drop pH column
# for i in range(n):
#     pH = c_measured.loc[i, 'pH']
#     c_measured.loc[i, 'H+'] = 10**(-pH)
# c_measured = c_measured.drop(['pH'], axis=1)

# Create an empty template like c_measured - to be altered & populated
c_prepared = pd.DataFrame(columns=c_measured.columns)

# Loop through the Days and calculate delta(t) = Day(i+1) - Day(i).
# If N = Delta(t) > 1, insert N - 1 empty rows at index i.
for i in range(n):
    # Append raw entry to template dataframe
    entry = c_measured.iloc[i,:]
    c_prepared = c_prepared.append(entry, ignore_index=True)
    if i < n - 1:
        # Calculate delta_t
        current_day = c_measured.loc[i, ['Day']].values
        next_day = c_measured.loc[i+1, ['Day']].values
        delta_t = next_day - current_day
    else: 
        break
    if delta_t > 1:
        # Insert n empty rows
        for _ in range(int(*delta_t)-1):
            c_prepared=c_prepared.append(pd.Series(), ignore_index=True)


# NORMALIZE 

# def min_max_scaling(series):
#     return (series - series.min()) / (series.max() - series.min())

# for column in c_prepared.columns:
#     c_prepared[column] = min_max_scaling(c_prepared[column])


# INTERPOLATION

# Define the independent variable, 'days'
N = len(c_prepared)
days = c_prepared.loc[:, ['Day']].values
days = np.asarray(days[:, 0])

# Return interpolation on given column, or False if not possible
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
    if len(y) > 0:
        y = interp.interp1d(days, y, fill_value='extrapolate')
        return y
    else:
        return False

# Interpolate on each column, if possible. Store list of NaN-columns.
column_names = c_prepared.columns
nan_columns = []
for column_name in column_names:
    y_interp = interp_column(c_prepared, column_name, days)
    if y_interp: # if interp. possible, i.e. not a column of NaN's
        for i in range(N):
            value = c_prepared.loc[i, [column_name]].values[0]
            if np.isnan(value):
                c_prepared.loc[i, column_name] = y_interp(i)
    else:
        nan_columns.append(column_name)

# Add Water, 1 mg/L? and 10 L => 10 mg


# OUTPUT PREPARED DATA

# Move to 'prepared' data folder (create it if non-existent)
os.chdir(os.path.join(base_dir, 'data'))
os.makedirs('prepared', exist_ok=True)
os.chdir('prepared')

# Output prepared data to .csv
c_prepared.to_csv('2_prepared_states.csv') 
A.tofile('A.csv',sep=',')
B.tofile('B.csv',sep=',')
X.tofile('X.csv',sep=',')

# Drop 'Day' column (not a state of the system)
days = c_prepared.loc[:, ['Day']].values
c_prepared_no_days = c_prepared.drop(['Day'], axis=1)

# Final input into objective function
states_measured = c_prepared_no_days.to_numpy()

# Go back to repo parent folder
os.chdir(base_dir)


