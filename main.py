import numpy as np
import functions
import os
from typing import Tuple

def main():

    # Takes file input from anywhere on pc given a full file path using os library 
    file_name = input("Please enter the name or path of the CSV data file (e.g., mydata.csv): ")
    
    # Check if file/path exists
    if not os.path.exists(file_name):
        print(f"ERROR: File not found at path: {file_name}")
        return
    
    print(f"--- Calling function with file: {file_name} ---")

    # --- Call the function ---
    # Since we used 'import functions', we access the function using dot notation:
    try:
        f0_res, Q_factor, S21_max_lin, i_max_idx = functions.f0_Q_extraction_3dBmethod_csv(file_name)
    except AttributeError:
        # Fallback if the function name used is the original one
        f0_res, Q_factor, S21_max_lin, i_max_idx = functions.f0_Q_extraction_3dBmethod_csv(file_name)
    except Exception as e:
        print(f"An error occurred during function execution: {e}")
        return

    # --- Print Results ---
    print(f"✅ Analysis Complete:")
    print(f"  Resonance Frequency (f0): {f0_res}")
    print(f"  Quality Factor (Q): {Q_factor}")
    print(f"  Maximum S21 Magnitude: {S21_max_lin}")
    print(f"  Max Index: {i_max_idx}")

    a, error_bar, error_barQ = functions.f0_Q_extraction_Lorentzian_fitting_method_errorbar(
        file_name, f0_res, Q_factor, S21_max_lin, i_max_idx, 3000
    )

    # Print results
    print("Fit parameters (a):", a)
    print("Resonance frequency error bar:", error_bar)
    print("Q factor error bar:", error_barQ)

if __name__ == "__main__":
    main()