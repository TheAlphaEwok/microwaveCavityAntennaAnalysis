import numpy as np
import math
from typing import Tuple
import pandas as pd
import os

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