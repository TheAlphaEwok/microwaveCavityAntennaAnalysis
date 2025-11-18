import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.optimize import least_squares

def f0_Q_extraction_3dBmethod_csv(file_name: str) -> Tuple[float, float, float, int]:
    """
    Calculates the resonance frequency (f0), quality factor (Q), maximum 
    magnitude of the S21 parameter (S21_max), and finds the index of S21_max
    using the 3dB method from a CSV data file

    The input file is expected to have three columns:
    1. Frequency (Freq_Raw)
    2. Magnitude in dB (Mag_Raw_dB)
    3. Phase in degrees (Phase_Raw_deg)

    Args:
        file_name: The full path of the CSV data file

    Returns:
        A tuple containing:
        - f0: Resonance frequency.
        - Q: Quality factor.
        - S21_max_linear: Maximum S21 magnitude in linear scale.
        - i_max: Index of the maximum S21 magnitude.
    """
    
    # Load the data from the CSV
    try:
        Data_df = pd.read_csv(
            file_name, 
            skiprows=3, 
            header=None, 
            delimiter=',', 
            comment='#',
            decimal='.'
        )
    except Exception as e:
        print(f"Error loading CSV file {file_name}: {e}")
        return 0.0, 0.0, 0.0, 0

    # Specify columns from CSV
    Freq_Raw = Data_df.iloc[:, 0]
    Mag_Raw_dB = Data_df.iloc[:, 1]
    Phase_Raw_deg = Data_df.iloc[:, 2]

    # Convert magnitude from dB to linear scale (Mag = 10^(Mag_dB / 20))
    Mag_Raw_linear = 10**(Mag_Raw_dB / 20)
    
    # Convert phase from degrees to radians for trigonometric functions
    Phase_Raw_rad = Phase_Raw_deg * (math.pi / 180.0)

    MagS21_Circle = Mag_Raw_linear
    Freq = Freq_Raw

    # Finds S21 max and resonant frequency
    i_max = np.argmax(MagS21_Circle)
    S21_max_linear = MagS21_Circle[i_max]
    f0 = Freq[i_max]
    
    # 3dB threshold in linear scale
    threshold_3dB = S21_max_linear / math.sqrt(2)
    
    # Finds the 3dB bandwidth
    
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

    # Calculate the bandwidth DeltaF
    if i_right3dB > i_left3dB:
        DeltaF = Freq[i_right3dB] - Freq[i_left3dB] 
        Q = f0 / DeltaF
    else:
        print("Warning: 3dB points not found or data is too narrow/flat. Setting Q to 0.")
        DeltaF = 0
        Q = 0.0

    return f0, Q, S21_max_linear, i_max

def f0_Q_extraction_Lorentzian_fitting_method_errorbar_dB(
        file_name, f03dB, Q3dB, S21_max, i_max):
    """
    Finds a Lorentzian fitting function with error bar estimation based on 
    data from 3dB method
    
    Args:
        file_name: The full path of the CSV data file
        f03dB: The resonant frequency found from 3dB method
        Q3dB: The quality factor found from 3dB method
        S21_max: The maximum magnitude found from 3dB method
        i_max: The index of S21)max

    Returns:
        a: fit parameters
        error_bar: error bar for f0
        error_barQ: error bar for Q
    """
    
    # Calculate an interval for fitting functionn based on data set
    DeltaF = f03dB / Q3dB  # 3 dB bandwidth in same units as f0
    Freq = pd.read_csv(file_name, skiprows=3, header=None, delimiter=',', comment='#', decimal='.').iloc[:, 0].values
    freq_step = np.mean(np.diff(np.unique(Freq)))
    interval = max(int(10 * DeltaF / freq_step), 1)

    # Load and group data to find averages for better graph
    df = pd.read_csv(       
            file_name, 
            skiprows=3, 
            header=None, 
            delimiter=',', 
            comment='#',
            decimal='.'
            )
    df_grouped = df.groupby(df.iloc[:,0]).agg({
        df.columns[1]: 'mean',   # average S21 in dB
        df.columns[2]: 'mean'    # average Phase
    }).reset_index()
    i_max = np.argmax(df_grouped.iloc[:,1].values)

    # Extract arrays
    Freq_Raw = df_grouped.iloc[:,0].values
    Mag_Raw_dB = df_grouped.iloc[:,1].values
    Phase_Raw = df_grouped.iloc[:,2].values

    start_i = max(i_max - interval//2, 0)
    end_i = min(i_max + interval//2, len(Freq_Raw)-1)
    
    # Array for fitting
    MagS21_dB = Mag_Raw_dB[start_i:end_i+1]
    Freq = Freq_Raw[start_i:end_i+1]/1e9 # Freq in GHz
    
    # Calculation of Lorentzian fitting function
    def lorentz_dB(a, F):
        # a[0]: Baseline (dB)
        # a[1]: Coupling coefficient or linear background (for phase/cable effects)
        # a[2]: Peak height (dB)
        # a[3]: Resonance Frequency f0 (GHz)
        # a[4]: Half-width at 3dB (Delta F) (GHz)
        # a[5]: Asymmetry/Fano parameter (for non-symmetric features)
        
        # Baseline + peak on a log scale
        # The peak is the power magnitude: |1 - coupling_term|^2
        return a[0] + a[1]*F + 10 * np.log10(
            (10**(a[2]/10)) / (1 + 4 * ((F - a[3]) / a[4])**2)
        )

    # a0 = [baseline, linear_slope, peak_height_dB, f0, DeltaF, asymmetry]
    a0 = [-60, 0, S21_max, f03dB/1e9, f03dB/(Q3dB*1e9), 0]
    
    def residuals(a, F, y):
        return lorentz_dB(a, F) - y
    
    # Fit the data using least squares
    result = least_squares(residuals, a0, args=(Freq, MagS21_dB))
    a = result.x
    resnorm = np.sum(result.fun**2)
    
    # Plotting the data and fitting function
    times = np.linspace(Freq[0], Freq[-1], 500)
    plt.plot(Freq, MagS21_dB, 'b*')
    plt.plot(times, lorentz_dB(a, times), 'r-', linewidth=1)
    plt.legend(['Data','Lorentzian fit'])
    plt.xlabel('Freq (GHz)')
    plt.ylabel('S21 (dB)')
    plt.title(file_name + ' (dB Fit)')
    plt.show()
    
    # -------- Error bar estimation for f0 and Q --------
    # The calculations for f0 and Q error bars use the same loop logic by 
    # finding where the residual sum increases by 3%

    # -------- Error bar estimation for f0 --------
    prevf0 = a[3]
    stepf0 = a[3]*0.01
    nextf0 = prevf0 + stepf0
    vvariance = 1*resnorm
    n = 0
    n_max = 700
    
    while abs(vvariance) > resnorm*1e-4 and n < n_max:
        ddeviation = (lorentz_dB([*a[:3], nextf0, *a[4:]], Freq) - MagS21_dB)**2
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
        ddeviation = (lorentz_dB([*a[:3], a[3], nextDF, a[5]], Freq) - MagS21_dB)**2
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