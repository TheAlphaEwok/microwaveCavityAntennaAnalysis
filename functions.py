import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.optimize import least_squares

def f0_Q_extraction_3dBmethod_csv(file_name: str) -> Tuple[float, float, float, int]:
    """
    Calculates the resonance frequency (f0), quality factor (Q), and maximum 
    magnitude of the S21 parameter (S21_max) using the 3dB method from a 
    **CSV (Comma-Separated Values) data file**.

    The input file is expected to have three columns:
    1. Frequency (Freq_Raw)
    2. Magnitude in dB (Mag_Raw_dB)
    3. Phase in degrees (Phase_Raw_deg)

    Args:
        file_name: The path/name of the CSV data file.

    Returns:
        A tuple containing:
        - f0: Resonance frequency.
        - Q: Quality factor.
        - S21_max_linear: Maximum S21 magnitude in linear scale.
        - i_max: Index of the maximum S21 magnitude (0-based).
    """
    
    # ************Read Input File (CSV)******** try:
        # Key Change: Use the 'delimiter' argument to specify a comma
    # The 'try' keyword must precede the block you are attempting to execute
    try:
        # Load the data, assuming it's a CSV with a comma delimiter
        Data_df = pd.read_csv(
            file_name, 
            skiprows=3, 
            header=None, 
            delimiter=',', 
            comment='#', # <--- ADD THIS
            decimal='.'  # Keep this for robustness against regional decimal formats
        )
    # The 'except' block handles errors from the 'try' block
    except Exception as e:
        print(f"Error loading CSV file {file_name}: {e}")
        return 0.0, 0.0, 0.0, 0
    

    
    # Execution continues here if loading succeeds (Data is defined)
    # ... rest of the function logic ...

    # Specify columns (same as before)
    Freq_Raw = Data_df.iloc[:, 0]
    Mag_Raw_dB = Data_df.iloc[:, 1]
    Phase_Raw_deg = Data_df.iloc[:, 2]

    # Convert magnitude from dB to linear scale (Mag = 10^(Mag_dB / 20))
    Mag_Raw_linear = 10**(Mag_Raw_dB / 20)
    
    # Convert phase from degrees to radians for trigonometric functions
    Phase_Raw_rad = Phase_Raw_deg * (math.pi / 180.0)

    MagS21_Circle = Mag_Raw_linear
    Freq = Freq_Raw

    # Unused but calculated for completeness
    ReS21_Circle = Mag_Raw_linear * np.cos(Phase_Raw_rad)
    ImS21_Circle = Mag_Raw_linear * np.sin(Phase_Raw_rad)

    # *** Find S21_max and f0 (Resonance Point) ***
    i_max = np.argmax(MagS21_Circle)
    S21_max_linear = MagS21_Circle[i_max]
    f0 = Freq[i_max]
    
    # 3dB threshold in linear scale
    threshold_3dB = S21_max_linear / math.sqrt(2)
    
    # *** Find 3dB Bandwidth (DeltaF) ***
    
    # 1. Find i_left3dB (the lower frequency boundary)
    i_left3dB = 0 
    for i in range(len(MagS21_Circle)):
        if MagS21_Circle[i] > threshold_3dB:
            i_left3dB = i
            break
    
    # 2. Find i_right3dB (the higher frequency boundary)
    i_right3dB = len(MagS21_Circle) - 1 
    for i in range(len(MagS21_Circle) - 1, -1, -1):
        if MagS21_Circle[i] > threshold_3dB:
            i_right3dB = i
            break

    # Calculate bandwidth DeltaF
    if i_right3dB > i_left3dB:
        DeltaF = Freq[i_right3dB] - Freq[i_left3dB] 
        Q = f0 / DeltaF
    else:
        print("Warning: 3dB points not found or data is too narrow/flat. Setting Q to 0.")
        DeltaF = 0
        Q = 0.0

    return f0, Q, S21_max_linear, i_max

def f0_Q_extraction_Lorentzian_fitting_method_errorbar(
        file_name, f03dB, Q3dB, S21_max, i_max, interval):
    """
    Lorentzian fitting with error bar estimation using pandas for CSV input.
    
    Returns:
        a: fit parameters
        error_bar: error bar for f0
        error_barQ: error bar for Q
    """
    # Load CSV with pandas (assumes columns: freq, S21, Phase21)
    df = pd.read_csv(file_name)
    df_grouped = df.groupby(df.iloc[:,0]).agg({
        df.columns[1]: 'mean',   # average S21
        df.columns[2]: 'mean'    # average Phase
    }).reset_index()

    # Extract arrays
    Freq_Raw = df_grouped.iloc[:,0].values
    Mag_Raw = 10**(df_grouped.iloc[:,1].values / 20)  # convert dB to linear
    Phase_Raw = df_grouped.iloc[:,2].values

    start_i = max(i_max - interval//2, 0)
    end_i = min(i_max + interval//2, len(Freq_Raw)-1)
    
    MagS21_Circle = Mag_Raw[start_i:end_i+1]*1e3
    ReS21_Circle = Mag_Raw[start_i:end_i+1]*np.cos(np.deg2rad(Phase_Raw[start_i:end_i+1]))
    ImS21_Circle = Mag_Raw[start_i:end_i+1]*np.sin(np.deg2rad(Phase_Raw[start_i:end_i+1]))
    Freq = Freq_Raw[start_i:end_i+1]/1e9
    
    # Lorentzian fit function
    def lorentz(a, F):
        return a[0] + a[1]*F + (a[2]+a[5]*F)/np.sqrt(1 + 4*((F-a[3])/a[4])**2)
    
    a0 = [0, 0, S21_max*1e3, f03dB/1e9, f03dB/(Q3dB*1e9), 0]
    
    def residuals(a, F, y):
        return lorentz(a, F) - y
    
    result = least_squares(residuals, a0, args=(Freq, MagS21_Circle))
    a = result.x
    resnorm = np.sum(result.fun**2)
    
    # Plot fit
    times = np.linspace(Freq[0], Freq[-1], 500)
    plt.plot(Freq, MagS21_Circle, 'k*')
    plt.plot(times, lorentz(a, times), 'b-', linewidth=2)
    plt.legend(['Data','Lorentzian fit'])
    plt.xlabel('Freq (GHz)')
    plt.ylabel('S21 (lin)*1e3')
    plt.title(file_name)
    plt.show()
    
    # -------- Error bar estimation for f0 --------
    prevf0 = a[3]
    stepf0 = a[3]*0.01
    nextf0 = prevf0 + stepf0
    vvariance = 1*resnorm
    n = 0
    n_max = 700
    
    while abs(vvariance) > resnorm*1e-4 and n < n_max:
        ddeviation = (lorentz([*a[:3], nextf0, *a[4:]], Freq) - MagS21_Circle)**2
        NEWvvariance = np.sum(ddeviation) - 1.03*resnorm
        if abs(vvariance) > abs(NEWvvariance) and vvariance*NEWvvariance > 0:
            nextf0 += stepf0
        elif abs(vvariance) < abs(NEWvvariance) and vvariance*NEWvvariance > 0:
            stepf0 = -stepf0
            nextf0 += stepf0
        else:
            stepf0 *= -0.5
            nextf0 += stepf0
        vvariance = NEWvvariance
        n += 1
    error_bar = abs(prevf0 - nextf0)
    
    # -------- Error bar estimation for Q --------
    prevDF = a[4]
    stepDF = a[4]*0.01
    nextDF = prevDF + stepDF
    vvariance = 1*resnorm
    n = 0
    
    while abs(vvariance) > resnorm*1e-6 and n < n_max:
        ddeviation = (lorentz([*a[:3], a[3], nextDF, a[5]], Freq) - MagS21_Circle)**2
        NEWvvariance = np.sum(ddeviation) - 1.03*resnorm
        if abs(vvariance) > abs(NEWvvariance) and vvariance*NEWvvariance > 0:
            nextDF += stepDF
        elif abs(vvariance) < abs(NEWvvariance) and vvariance*NEWvvariance > 0:
            stepDF = -stepDF
            nextDF += stepDF
        else:
            stepDF *= -0.5
            nextDF += stepDF
        vvariance = NEWvvariance
        n += 1
    
    error_barQ = abs(a[3]*(prevDF - nextDF) / (nextDF**2))
    
    return a, error_bar, error_barQ
