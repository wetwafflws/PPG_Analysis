# PPG Signal Analysis Tool

A Python application using CustomTkinter to visualize and analyze Photoplethysmogram (PPG) signals, focusing on infrared data. The tool provides preprocessing, Discrete Wavelet Transform (DWT) decomposition using a custom filter bank, Heart Rate Variability (HRV) analysis in both time and frequency domains, and respiratory/vasometric rate estimation.

## Features

* **Load CSV Data:** Load PPG data from a CSV file.
* **Resampling:** Resample the signal to a desired frequency and re-process.
* **Preprocessing:**
    * Displays the raw loaded signal.
    * Applies Z-score normalization and a 0.5-40 Hz bandpass filter.
    * Displays the preprocessed signal.
* **Filter Response:**
    * Calculates and plots the cascaded frequency response of the 8 DWT scales based on the current sampling frequency, using the custom filter bank defined in the script.
* **RR Time Domain Analysis:**
    * Detects systolic peaks on the preprocessed signal.
    * Plots the peak detection results, the RR interval tachogram, and the Poincaré plot.
    * Calculates and displays time-domain HRV features: Mean RR, Mean HR, SDNN, RMSSD, NN50, pNN50, HTI, TINN (approx), SD1, SD2.
* **RR Frequency Domain Analysis:**
    * Calculates the Power Spectral Density (PSD) of the RR intervals using Welch's method.
    * Plots the PSD with VLF, LF, and HF bands highlighted.
    * Plots an Autonomic Balance Diagram (LFnu vs. HFnu) divided into 9 sections.
    * Calculates and displays frequency-domain HRV features: VLF, LF, HF power, Total Power (LF+HF), LF/HF Ratio, LFnu, HFnu.
* **DWT Decompositions:**
    * Applies a custom 8-level DWT based on specific low-pass (`h`) and high-pass (`g`) filters.
    * Allows selection and display of any DWT detail coefficient (d1-d8) plotted against time.
    * Displays the theoretical frequency range for the selected scale.
* **DWT Rate Analysis:**
    * Allows selection of specific DWT scales for respiratory and vasometric analysis.
    * **Respiratory Rate:** Finds peaks in the selected DWT detail coefficient, calculates Breaths Per Minute (BPM), and plots the signal with detected peaks against time.
    * **Vasometric Rate:** Calculates the Fast Fourier Transform (FFT) of the selected DWT detail coefficient, identifies the peak frequency, and plots the magnitude spectrum.
* **Interactive Plots:** All plots include a toolbar for zooming, panning, and saving.
* **Scrollable UI:** The main interface is scrollable to accommodate various screen sizes and plot heights.

## Setup

### Prerequisites
- **Python 3.8 or higher**: Download and install from [python.org](https://www.python.org/downloads/). Ensure `python` and `pip` are available in your PATH.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/wetwafflws/PPG_Analysis.git
    cd PPG_Analysis 
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    * Activate on Windows: `.\venv\Scripts\activate`
    * Activate on macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This installs the required Python libraries:
    - **CustomTkinter**: For the modern GUI interface.
    - **NumPy**: For efficient numerical computations and array handling.
    - **Matplotlib**: For creating and displaying plots.
    - **SciPy**: For signal processing functions like filtering and FFT.

## Usage

1.  **Run the Script:**
    ```bash
    python main.py
    ```

2.  **Load Data:**
    * Click the "Load CSV File" button.
    * Select your PPG data file.
    * **CSV Format:** The script expects a CSV file with:
        * A **1-row header** (which is skipped).
        * Comma (`,`) delimiters.
        * At least 3 columns.
        * **Column 0:** Time data in **microseconds**.
        * **Column 2:** Infrared PPG signal data.
        * The original sampling frequency is assumed to be **100 Hz**.

3.  **Resample (Optional):**
    * Enter a new sampling frequency (in Hz) in the text box at the top.
    * Click "Resample & Re-process". The original 100 Hz data will be resampled, and all analyses will be redone based on the new rate.

4.  **Explore Tabs:**
    * **Preprocessing:** Shows the raw and filtered signals.
    * **Filter Response:** Click "Plot Filter Response" to see the DWT filter bank characteristics for the *current* sampling frequency.
    * **RR Time Domain:** Click "Run Time Domain Analysis" to see peak detection, tachogram, Poincaré plot, and calculated time-domain features.
    * **RR Freq. Domain:** Click "Run Freq. Domain Analysis" to see the PSD, Autonomic Balance Diagram, and calculated frequency-domain features.
    * **DWT Decompositions:** Use the dropdown to select and view different DWT detail coefficients (d1-d8).
    * **DWT Rate Analysis:** Select the desired DWT scales for respiratory and vasometric analysis, then click "Run DWT Analysis".

## Building the Executable

To create a standalone executable (.exe) for Windows distribution:

1. **Ensure Setup is Complete:** Follow the setup instructions above to create a virtual environment and install dependencies.

2. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

3. **Build the Executable:**
   ```bash
   pyinstaller --onefile --windowed main.py
   ```
   - `--onefile`: Creates a single executable file.
   - `--windowed`: Hides the console window for a cleaner GUI experience.

4. **Locate the Executable:** The resulting `main.exe` will be in the `dist/` folder.

5. **Run the Application:**
   - Place `main.exe` and any required CSV data files (e.g., `PPG_Data_1.csv`, `PPG_Data_2.csv`) in the same directory.
   - Double-click `main.exe` to launch the PPG analysis GUI.

**Notes:**
- The executable is self-contained and includes all dependencies (NumPy, SciPy, Matplotlib, CustomTkinter, etc.), so it runs on Windows machines without requiring Python installation.
- For distribution, ensure the CSV files are included if users need sample data.

## Dependencies
To install all dependencies run 
