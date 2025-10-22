import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import butter, filtfilt, find_peaks, resample, welch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import warnings
from customtkinter import filedialog

# Suppress warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class PPGVisualizerApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Infrared PPG Data Visualization (Custom DWT)")
        self.geometry("1000x800") # Initial size, might be expanded by scroll
        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("blue")

        # --- Make the main window scrollable ---
        self.main_scroll_frame = customtkinter.CTkScrollableFrame(self)
        self.main_scroll_frame.pack(fill="both", expand=True)
        # --- End Scrollable Frame ---

        # --- 1. Data Initialization ---
        self.original_fs = 100.0
        self.original_t = np.array([])
        self.original_raw_infrared = np.array([])

        self.fs = 0.0
        self.t = np.array([])
        self.raw_infrared = np.array([])
        self.filtered_infrared = np.array([])
        self.dwt_coeffs = []

        self.h_coeffs = np.array([0, 1/8, 3/8, 3/8])
        self.g_coeffs = np.array([0, 0, -2, 2])
        self.n_range = np.arange(-2, 2)

        self.canvas_list = []
        self.dwt_canvas_widget = None
        self.dwt_rate_canvas_widget = None
        self.filter_canvas_widget = None
        self.hrv_time_canvas_widget = None
        self.hrv_freq_canvas_widget = None

        # --- 2. GUI Setup (Place inside main_scroll_frame) ---

        # --- Top Controls Frame ---
        top_controls_frame = customtkinter.CTkFrame(self.main_scroll_frame) # <-- Place inside scroll frame
        top_controls_frame.pack(fill="x", pady=10, padx=20)

        self.load_button = customtkinter.CTkButton(top_controls_frame, text="Load CSV File", command=self.load_csv_data)
        self.load_button.pack(side="left", padx=10, pady=10)

        self.resample_label = customtkinter.CTkLabel(top_controls_frame, text="New Sample Rate (Hz):")
        self.resample_label.pack(side="left", padx=(20, 0))

        self.fs_entry = customtkinter.CTkEntry(top_controls_frame, width=60)
        self.fs_entry.pack(side="left", padx=10)

        self.resample_button = customtkinter.CTkButton(
            top_controls_frame,
            text="Resample & Re-process",
            command=self.resample_and_reprocess
        )
        self.resample_button.pack(side="left", padx=10, pady=10)

        # --- Tab View ---
        self.tab_view = customtkinter.CTkTabview(self.main_scroll_frame) # <-- Place inside scroll frame
        self.tab_view.pack(padx=20, pady=10, fill="both", expand=True)

        self.preprocess_tab = self.tab_view.add("Preprocessing")
        self.filter_resp_tab = self.tab_view.add("Filter Response")
        self.hrv_time_tab = self.tab_view.add("RR Time Domain")
        self.hrv_freq_tab = self.tab_view.add("RR Freq. Domain")
        self.dwt_tab = self.tab_view.add("DWT Decompositions")
        self.dwt_rate_tab = self.tab_view.add("DWT Rate Analysis")

        # --- Tab 1: Preprocessing ---
        self.preprocess_plot_frame = customtkinter.CTkFrame(self.preprocess_tab)
        self.preprocess_plot_frame.pack(fill="both", expand=True)

        # --- Tab 2: Filter Response ---
        filter_controls_frame = customtkinter.CTkFrame(self.filter_resp_tab)
        filter_controls_frame.pack(fill="x", pady=5)
        self.plot_resp_button = customtkinter.CTkButton(
            filter_controls_frame,
            text="Plot Filter Response",
            command=self.plot_filter_response
        )
        self.plot_resp_button.pack(padx=10, pady=10)
        self.filter_plot_frame = customtkinter.CTkFrame(self.filter_resp_tab)
        self.filter_plot_frame.pack(fill="both", expand=True)

        # --- Tab 3: RR Time Domain ---
        hrv_time_controls_frame = customtkinter.CTkFrame(self.hrv_time_tab)
        hrv_time_controls_frame.pack(fill="x", pady=5)
        self.run_hrv_time_button = customtkinter.CTkButton(
            hrv_time_controls_frame,
            text="Run Time Domain Analysis",
            command=self.run_hrv_time_analysis
        )
        self.run_hrv_time_button.pack(padx=10, pady=10)
        hrv_time_main_frame = customtkinter.CTkFrame(self.hrv_time_tab)
        hrv_time_main_frame.pack(fill="both", expand=True)
        hrv_time_main_frame.grid_columnconfigure(0, weight=3) # Plot
        hrv_time_main_frame.grid_columnconfigure(1, weight=1) # Results
        hrv_time_main_frame.grid_rowconfigure(0, weight=1)
        self.hrv_time_plot_frame = customtkinter.CTkFrame(hrv_time_main_frame)
        self.hrv_time_plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.hrv_time_results_frame = customtkinter.CTkScrollableFrame(hrv_time_main_frame, label_text="Time Domain Features")
        self.hrv_time_results_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # --- Tab 4: RR Freq. Domain ---
        hrv_freq_controls_frame = customtkinter.CTkFrame(self.hrv_freq_tab)
        hrv_freq_controls_frame.pack(fill="x", pady=5)
        self.run_hrv_freq_button = customtkinter.CTkButton(
            hrv_freq_controls_frame,
            text="Run Freq. Domain Analysis",
            command=self.run_hrv_freq_analysis
        )
        self.run_hrv_freq_button.pack(padx=10, pady=10)
        hrv_freq_main_frame = customtkinter.CTkFrame(self.hrv_freq_tab)
        hrv_freq_main_frame.pack(fill="both", expand=True)
        hrv_freq_main_frame.grid_columnconfigure(0, weight=3) # Plot
        hrv_freq_main_frame.grid_columnconfigure(1, weight=1) # Results
        hrv_freq_main_frame.grid_rowconfigure(0, weight=1)
        self.hrv_freq_plot_frame = customtkinter.CTkFrame(hrv_freq_main_frame)
        self.hrv_freq_plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.hrv_freq_results_frame = customtkinter.CTkScrollableFrame(hrv_freq_main_frame, label_text="Frequency Domain Features")
        self.hrv_freq_results_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # --- Tab 5: DWT Decompositions ---
        dwt_controls_frame = customtkinter.CTkFrame(self.dwt_tab)
        dwt_controls_frame.pack(fill="x", pady=5)
        dwt_label = customtkinter.CTkLabel(dwt_controls_frame, text="Select DWT Scale to Display:")
        dwt_label.pack(side="left", padx=10)
        scale_values = [f"Scale {i}" for i in range(1, 9)]
        self.dwt_option_menu = customtkinter.CTkOptionMenu(
            dwt_controls_frame,
            values=scale_values,
            command=self.update_dwt_plot
        )
        self.dwt_option_menu.pack(side="left", padx=10, pady=10)
        self.dwt_plot_frame = customtkinter.CTkFrame(self.dwt_tab)
        self.dwt_plot_frame.pack(fill="both", expand=True)

        # --- Tab 6: DWT Rate Analysis ---
        dwt_rate_controls_frame = customtkinter.CTkFrame(self.dwt_rate_tab)
        dwt_rate_controls_frame.pack(fill="x", pady=5)
        rr_label = customtkinter.CTkLabel(dwt_rate_controls_frame, text="Respiratory Scale:")
        rr_label.pack(side="left", padx=(10,0))
        self.rr_option_menu = customtkinter.CTkOptionMenu(dwt_rate_controls_frame, values=scale_values)
        self.rr_option_menu.pack(side="left", padx=10)
        vr_label = customtkinter.CTkLabel(dwt_rate_controls_frame, text="Vasometric Scale:")
        vr_label.pack(side="left", padx=(10,0))
        self.vr_option_menu = customtkinter.CTkOptionMenu(dwt_rate_controls_frame, values=scale_values)
        self.vr_option_menu.pack(side="left", padx=10)
        self.run_dwt_rate_button = customtkinter.CTkButton(
            dwt_rate_controls_frame,
            text="Run DWT Analysis",
            command=self.run_dwt_rate_analysis
        )
        self.run_dwt_rate_button.pack(side="left", padx=20, pady=10)
        self.dwt_rate_plot_frame = customtkinter.CTkFrame(self.dwt_rate_tab)
        self.dwt_rate_plot_frame.pack(fill="both", expand=True)

    # --- 3. Custom DWT Functions ---

    def lpf(self, x):
        return np.convolve(x, self.h_coeffs, 'same')

    def hpf(self, x):
        return np.convolve(x, self.g_coeffs, 'same')

    def downsample(self, x):
        return x[::2]

    def dwt(self, x, J):
        a = x
        d = []
        for j in range(J):
            d_j = self.downsample(self.hpf(a))
            a_j = self.downsample(self.lpf(a))
            d.append(d_j)
            a = a_j
        return a, d

    # --- 4. Main Application Logic ---

    def load_csv_data(self):
        """Loads and sets up the original signal."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return

        self.clear_all_plots_for_new_file()

        try:
            raw_data = np.loadtxt(file_path, delimiter=',', skiprows=1)

            self.original_t = raw_data[:, 0] / 1000000.0
            self.original_raw_infrared = raw_data[:, 2]
            self.original_fs = 100.0

            self.t = self.original_t
            self.raw_infrared = self.original_raw_infrared
            self.fs = self.original_fs

            self.fs_entry.delete(0, 'end')
            self.fs_entry.insert(0, str(self.fs))

            self.process_and_display_data()

        except Exception as e:
            err_text = f"Error loading file: {e}\n\n"
            err_text += "Ensure it's a CSV file with at least 3 columns,\n"
            err_text += "a 1-row header, and comma delimiters."
            err_label = customtkinter.CTkLabel(self.preprocess_plot_frame, text=err_text, text_color="red")
            err_label.pack(pady=20, padx=20)
            self.canvas_list.append(err_label)

    def resample_and_reprocess(self):
        """Resamples the *original* signal and re-runs the full pipeline."""
        if len(self.original_raw_infrared) == 0:
            return

        try:
            new_fs = float(self.fs_entry.get())
            if new_fs <= 0:
                raise ValueError("Sample rate must be positive.")
        except ValueError as e:
            self.show_error_in_hrv_time_tab(f"Invalid sample rate: {e}")
            return

        if new_fs == self.fs:
            return

        self.clear_existing_plots()

        try:
            original_num_samples = len(self.original_raw_infrared)
            new_num_samples = int(original_num_samples * new_fs / self.original_fs)

            if new_num_samples <= 0:
                raise ValueError("Resulting signal has 0 samples.")

            self.raw_infrared = resample(self.original_raw_infrared, new_num_samples)
            self.t = np.linspace(
                self.original_t[0],
                self.original_t[-1],
                new_num_samples
            )
            self.fs = new_fs

            self.process_and_display_data()

        except Exception as e:
            self.show_error_in_hrv_time_tab(f"Error during resampling: {e}")
            self.t = self.original_t
            self.raw_infrared = self.original_raw_infrared
            self.fs = self.original_fs
            self.fs_entry.delete(0, 'end')
            self.fs_entry.insert(0, str(self.fs))

    def process_and_display_data(self):
        """Runs the preprocessing and DWT steps."""
        self.filtered_infrared = self.preprocess_signal(self.raw_infrared, fs=self.fs)

        a, d_coeffs_list = self.dwt(self.filtered_infrared, J=8)
        self.dwt_coeffs = d_coeffs_list

        self.create_preprocessing_plots()

        default_dwt_scale = "Scale 1"
        self.dwt_option_menu.set(default_dwt_scale)
        self.update_dwt_plot(default_dwt_scale)

        self.rr_option_menu.set("Scale 5")
        self.vr_option_menu.set("Scale 8")

    # --- RR Interval Analysis (Time and Freq) ---

    def _get_rr_intervals(self):
        """Helper function to find peaks and return RR intervals."""
        if len(self.filtered_infrared) == 0:
            raise ValueError("No data loaded. Please load a CSV file.")

        min_dist_samples = int(self.fs * 0.4) # Min dist 0.4s (max 150 BPM)
        peak_indices, _ = find_peaks(self.filtered_infrared, height=0, distance=max(1, min_dist_samples))

        if len(peak_indices) < 2:
            raise ValueError("Not enough peaks found (< 2) to calculate RR intervals.")

        peak_times_sec = peak_indices / self.fs
        rr_intervals_sec = np.diff(peak_times_sec)
        rr_intervals_ms = rr_intervals_sec * 1000

        if len(rr_intervals_sec) < 2:
            raise ValueError("Not enough RR intervals (< 2) for analysis.")

        return peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms

    def run_hrv_time_analysis(self):
        """Finds peaks and calculates time-domain HRV features."""
        if self.hrv_time_canvas_widget:
            self.hrv_time_canvas_widget.destroy()
            self.hrv_time_canvas_widget = None
        for widget in self.hrv_time_results_frame.winfo_children():
            widget.destroy()

        try:
            peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms = self._get_rr_intervals()

            features = self.calculate_time_domain_features(rr_intervals_ms)

            self.create_hrv_time_plots(peak_indices, peak_times_sec, rr_intervals_ms, features)

        except Exception as e:
            self.show_error_in_hrv_time_tab(f"Error during Time analysis: {e}")

    def calculate_time_domain_features(self, rr_ms):
        """Calculates time domain features including HTI, TINN, SD1, SD2."""
        features = {}
        if len(rr_ms) < 2: return features

        diff_rr_ms = np.diff(rr_ms)

        # Basic stats
        features["Mean RR"] = np.mean(rr_ms)
        features["Mean HR"] = (60 * 1000) / features["Mean RR"] if features["Mean RR"] != 0 else 0
        features["SDNN"] = np.std(rr_ms)
        features["RMSSD"] = np.sqrt(np.mean(diff_rr_ms**2)) if len(diff_rr_ms) > 0 else 0
        features["NN50"] = np.sum(np.abs(diff_rr_ms) > 50) if len(diff_rr_ms) > 0 else 0
        features["pNN50"] = (features["NN50"] / len(diff_rr_ms)) * 100 if len(diff_rr_ms) > 0 else 0

        # Histogram features (HTI, TINN approximation)
        try:
            bin_width_ms = 8
            data_range = np.ptp(rr_ms) # Peak-to-peak (max-min)
            num_bins = max(10, int(np.ceil(data_range / bin_width_ms)))
            counts, bin_edges = np.histogram(rr_ms, bins=num_bins)
            bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

            if len(counts) > 0:
                peak_count = np.max(counts)
                features["HTI"] = len(rr_ms) / peak_count if peak_count > 0 else 0
                features["TINN (approx)"] = bin_edges[-1] - bin_edges[0]
            else:
                features["HTI"] = 0
                features["TINN (approx)"] = 0
        except Exception:
            features["HTI"] = np.nan
            features["TINN (approx)"] = np.nan


        # Poincaré features (SD1, SD2)
        if len(rr_ms) >= 2:
            rr_n = rr_ms[:-1]
            rr_n1 = rr_ms[1:]
            diff = rr_n - rr_n1
            summ = rr_n + rr_n1
            # Use ddof=0 for population standard deviation
            features["SD1"] = np.sqrt(np.std(diff, ddof=0)**2 * 0.5)
            features["SD2"] = np.sqrt(np.std(summ, ddof=0)**2 * 0.5)
        else:
            features["SD1"] = 0
            features["SD2"] = 0

        return features

    def run_hrv_freq_analysis(self):
        """Calculates and plots the PSD of the RR intervals."""
        if self.hrv_freq_canvas_widget:
            self.hrv_freq_canvas_widget.destroy()
            self.hrv_freq_canvas_widget = None
        for widget in self.hrv_freq_results_frame.winfo_children():
            widget.destroy()

        try:
            peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms = self._get_rr_intervals()

            rr_times_sec = peak_times_sec[1:] # Time of each RR interval

            fs_interp = 4.0
            t_interp = np.arange(rr_times_sec[0], rr_times_sec[-1], 1/fs_interp)

            if len(rr_times_sec) < 3:
                raise ValueError("Not enough RR intervals (< 3) for cubic interpolation.")

            interp_func = interp1d(rr_times_sec, rr_intervals_sec, kind='cubic', fill_value="extrapolate")
            rr_interp_sec = interp_func(t_interp)

            nperseg = min(256, len(rr_interp_sec))
            if nperseg == 0:
                 raise ValueError("Interpolated signal too short for Welch method.")
            fxx, pxx = welch(rr_interp_sec, fs=fs_interp, nperseg=nperseg)

            features = self.calculate_freq_domain_features(fxx, pxx)

            self.create_hrv_freq_plots(fxx, pxx, features)

        except Exception as e:
            self.show_error_in_hrv_freq_tab(f"Error during Freq. analysis: {e}")

    def calculate_freq_domain_features(self, fxx, pxx):
        """Calculates VLF, LF, HF power, LF/HF ratio, LFnu, HFnu."""
        features = {}
        vlf_band = (fxx >= 0.003) & (fxx < 0.04)
        lf_band = (fxx >= 0.04) & (fxx < 0.15)
        hf_band = (fxx >= 0.15) & (fxx < 0.4)

        features["VLF"] = trapezoid(pxx[vlf_band], fxx[vlf_band])
        features["LF"] = trapezoid(pxx[lf_band], fxx[lf_band])
        features["HF"] = trapezoid(pxx[hf_band], fxx[hf_band])

        lf_plus_hf = features["LF"] + features["HF"]
        features["Total Power (LF+HF)"] = lf_plus_hf
        features["LF/HF Ratio"] = features["LF"] / features["HF"] if features["HF"] > 0 else 0

        features["LFnu"] = (features["LF"] / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
        features["HFnu"] = (features["HF"] / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0

        return features

    # --- DWT-Based Rate Analysis ---

    def run_dwt_rate_analysis(self):
        """Runs peak finding and FFT based on user's DWT scale selection."""
        if not self.dwt_coeffs:
            return

        if self.dwt_rate_canvas_widget:
            self.dwt_rate_canvas_widget.destroy()
            self.dwt_rate_canvas_widget = None

        try:
            rr_scale_str = self.rr_option_menu.get()
            vr_scale_str = self.vr_option_menu.get()

            rr_level = int(rr_scale_str.split(' ')[1])
            vr_level = int(vr_scale_str.split(' ')[1])

            rr_scale_index = rr_level - 1
            vr_scale_index = vr_level - 1

            rr_signal_data = self.dwt_coeffs[rr_scale_index]
            rr_signal, rr_peaks, bpm = self.analyze_respiratory(rr_signal_data, rr_level)

            vr_signal_data = self.dwt_coeffs[vr_scale_index]
            vr_freqs, vr_mag, vr_peak_freq, vr_peak_mag = self.analyze_vasometric(vr_signal_data, vr_level)

            self.create_dwt_rate_plots(
                rr_signal, rr_peaks, bpm, rr_level,
                vr_freqs, vr_mag, vr_peak_freq, vr_peak_mag, vr_level
            )

        except Exception as e:
            self.show_error_in_dwt_rate_tab(f"Error during DWT analysis: {e}")

    # --- Signal Processing Helpers ---

    def preprocess_signal(self, data, fs):
        """Applies normalization and bandpass filtering."""
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std

        nyquist = fs / 2
        lowcut = 0.5
        highcut = min(40.0, nyquist - 1.0)

        if highcut <= lowcut:
             return normalized

        b, a = butter(5, [lowcut, highcut], btype='band', fs=fs)
        filtered = filtfilt(b, a, normalized)
        return filtered

    def analyze_respiratory(self, d_signal, level):
        """Finds respiratory peaks from the selected d-signal."""
        fs_level = self.fs / (2**level)

        min_distance_samples = fs_level * 0.5
        safe_distance = max(1, int(np.ceil(min_distance_samples)))

        peaks, _ = find_peaks(d_signal, height=np.mean(d_signal), distance=safe_distance)

        duration_s = len(d_signal) / fs_level
        num_peaks = len(peaks)
        bpm = (num_peaks / duration_s) * 60 if duration_s > 0 else 0

        return d_signal, peaks, bpm

    def analyze_vasometric(self, d_signal, level):
        """Calculates FFT for the selected d-signal."""
        n = len(d_signal)
        fs_level = self.fs / (2**level)

        if fs_level <= 0: return np.array([]), np.array([]), 0, 0

        fft_values = np.fft.fft(d_signal)
        fft_freq = np.fft.fftfreq(n, d=1/fs_level)
        fft_magnitude = np.abs(fft_values)

        positive_freqs = fft_freq[fft_freq > 0]
        positive_magnitude = fft_magnitude[fft_freq > 0]

        if len(positive_magnitude) > 0:
            peak_idx = np.argmax(positive_magnitude)
            peak_freq = positive_freqs[peak_idx]
            peak_mag = positive_magnitude[peak_idx]
        else:
            peak_freq, peak_mag = 0, 0

        return positive_freqs, positive_magnitude, peak_freq, peak_mag

    def get_freq_range(self, level):
        """Calculates the theoretical frequency band for a given DWT level."""
        f_high = self.fs / (2**level)
        f_low = self.fs / (2**(level + 1))
        return f_low, f_high

    # --- 5. GUI Plotting Functions ---

    def clear_existing_plots(self):
        """Destroys all existing plot canvases."""
        for widget in self.canvas_list:
            widget.destroy()
        self.canvas_list = []

        if self.dwt_canvas_widget: self.dwt_canvas_widget.destroy()
        if self.dwt_rate_canvas_widget: self.dwt_rate_canvas_widget.destroy()
        if self.filter_canvas_widget: self.filter_canvas_widget.destroy()
        if self.hrv_time_canvas_widget: self.hrv_time_canvas_widget.destroy()
        if self.hrv_freq_canvas_widget: self.hrv_freq_canvas_widget.destroy()

        self.dwt_canvas_widget = None
        self.dwt_rate_canvas_widget = None
        self.filter_canvas_widget = None
        self.hrv_time_canvas_widget = None
        self.hrv_freq_canvas_widget = None

        for frame in [self.hrv_time_results_frame, self.hrv_freq_results_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        for frame in [self.preprocess_plot_frame, self.filter_plot_frame,
                      self.hrv_time_plot_frame, self.hrv_freq_plot_frame,
                      self.dwt_plot_frame, self.dwt_rate_plot_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

    def clear_all_plots_for_new_file(self):
        """Destroys all existing plot canvases."""
        self.clear_existing_plots()

    # --- Error Display Helpers ---
    def show_error_in_dwt_rate_tab(self, err_text):
        err_label = customtkinter.CTkLabel(self.dwt_rate_plot_frame, text=err_text, text_color="red")
        err_label.pack(pady=20, padx=20)
        self.dwt_rate_canvas_widget = err_label

    def show_error_in_hrv_time_tab(self, err_text):
        err_label = customtkinter.CTkLabel(self.hrv_time_plot_frame, text=err_text, text_color="red")
        err_label.pack(pady=20, padx=20)
        self.hrv_time_canvas_widget = err_label

    def show_error_in_hrv_freq_tab(self, err_text):
        err_label = customtkinter.CTkLabel(self.hrv_freq_plot_frame, text=err_text, text_color="red")
        err_label.pack(pady=20, padx=20)
        self.hrv_freq_canvas_widget = err_label

    def show_error_in_filter_tab(self, err_text):
        err_label = customtkinter.CTkLabel(self.filter_plot_frame, text=err_text, text_color="red")
        err_label.pack(pady=20, padx=20)
        self.filter_canvas_widget = err_label

    def embed_plot(self, fig, tab_or_frame):
        """Embeds a Matplotlib figure and its toolbar into a customtkinter tab/frame."""
        plot_container = customtkinter.CTkFrame(tab_or_frame)
        plot_container.pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=plot_container)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_container)
        toolbar.update()
        toolbar.pack(side=customtkinter.BOTTOM, fill=customtkinter.X)
        canvas.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=True)
        plt.close(fig)
        return plot_container

    # --- Plot Creation Functions ---

    def update_dwt_plot(self, selected_scale_str):
        """Clears and redraws the DWT plot based on the option menu."""
        if not self.dwt_coeffs: return
        if self.dwt_canvas_widget: self.dwt_canvas_widget.destroy()

        try:
            scale_j = int(selected_scale_str.split(' ')[1])
            signal_data = self.dwt_coeffs[scale_j - 1]
            f_low, f_high = self.get_freq_range(scale_j)

            fs_level = self.fs / (2**scale_j)
            duration_s = len(signal_data) / fs_level
            t_level = np.linspace(0, duration_s, len(signal_data))

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#2B2B2B')

            ax.plot(t_level, signal_data, color='lime')
            title = f"DWT d{scale_j} Coefficient (Freq: {f_low:.3f} - {f_high:.3f} Hz)"
            ax.set_title(title, color='white')
            ax.set_xlabel("Time (s)", color='white')
            ax.set_ylabel("Amplitude", color='white')
            ax.tick_params(colors='white')
            ax.set_facecolor('#3C3C3C')
            fig.tight_layout()

            self.dwt_canvas_widget = self.embed_plot(fig, self.dwt_plot_frame)

        except Exception as e:
            err_label = customtkinter.CTkLabel(self.dwt_plot_frame, text=f"Error plotting scale: {e}", text_color="red")
            err_label.pack(pady=20, padx=20)
            self.dwt_canvas_widget = err_label

    def create_preprocessing_plots(self):
        """Plots raw and filtered signals."""
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        fig.patch.set_facecolor('#2B2B2B')

        ax1.plot(self.t, self.raw_infrared, color='cyan')
        ax1.set_title(f"Raw Infrared Signal (fs = {self.fs:.1f} Hz)", color='white')
        ax1.set_xlabel("Time (s)", color='white')
        ax1.set_ylabel("Amplitude", color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#3C3C3C')

        ax2.plot(self.t, self.filtered_infrared, color='magenta')
        ax2.set_title("Preprocessed (Filtered) Infrared Signal", color='white')
        ax2.set_xlabel("Time (s)", color='white')
        ax2.set_ylabel("Normalized Amplitude", color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#3C3C3C')

        fig.tight_layout()
        widget = self.embed_plot(fig, self.preprocess_plot_frame)
        self.canvas_list.append(widget)

    def create_hrv_time_plots(self, peak_indices, peak_times_sec, rr_intervals_ms, features):
        """Plots peak detection, tachogram, Poincaré, and displays features."""

        # 1. Create the plot (3 subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10)) # Adjusted figsize
        fig.patch.set_facecolor('#2B2B2B')

        # --- Plot 1: Peak Detection ---
        ax1.plot(self.t, self.filtered_infrared, color='magenta', label='Filtered Signal')
        ax1.plot(self.t[peak_indices], self.filtered_infrared[peak_indices], 'x', color='red', markersize=8, label='Detected Peaks')
        ax1.set_title(f"Peak Detection on Preprocessed Signal", color='white')
        ax1.set_ylabel("Amplitude", color='white')
        ax1.tick_params(colors='white', labelbottom=False) # Hide x-labels
        ax1.set_facecolor('#3C3C3C')
        ax1.legend()

        # --- Plot 2: RR Tachogram ---
        ax2.plot(peak_times_sec[1:], rr_intervals_ms, marker='o', linestyle='-', markersize=4, color='cyan')
        ax2.set_title(f"RR Interval Tachogram", color='white')
        ax2.set_ylabel("RR Interval (ms)", color='white')
        ax2.tick_params(colors='white', labelbottom=False) # Hide x-labels
        ax2.set_facecolor('#3C3C3C')
        ax2.sharex(ax1) # Link x-axis with ax1

        # --- Plot 3: Poincaré Plot ---
        rr_n = rr_intervals_ms[:-1]
        rr_n1 = rr_intervals_ms[1:]
        ax3.scatter(rr_n, rr_n1, color='lime', alpha=0.5, s=10) # Use scatter
        ax3.set_title("Poincaré Plot", color='white')
        ax3.set_xlabel("RRn (ms)", color='white')
        ax3.set_ylabel("RRn+1 (ms)", color='white')
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#3C3C3C')
        min_rr = np.min(rr_intervals_ms)
        max_rr = np.max(rr_intervals_ms)
        ax3.plot([min_rr, max_rr], [min_rr, max_rr], color='gray', linestyle='--')
        ax3.set_aspect('equal', adjustable='box')

        fig.tight_layout(h_pad=2.0) # Add vertical padding
        self.hrv_time_canvas_widget = self.embed_plot(fig, self.hrv_time_plot_frame)

        # 2. Create the results labels
        font_settings = ("Roboto", 14)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"Mean RR: {features.get('Mean RR', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"Mean HR: {features.get('Mean HR', 0):.2f} BPM", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"SDNN: {features.get('SDNN', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"RMSSD: {features.get('RMSSD', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"NN50: {features.get('NN50', 0)} counts", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"pNN50: {features.get('pNN50', 0):.2f} %", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"HTI: {features.get('HTI', 0):.2f}", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"TINN (approx): {features.get('TINN (approx)', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"SD1: {features.get('SD1', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_time_results_frame, text=f"SD2: {features.get('SD2', 0):.2f} ms", font=font_settings).pack(anchor="w", padx=10)

        customtkinter.CTkLabel(self.hrv_time_results_frame, text="\nNote: SDANN/SDNN index \nnot calculated (req. >5min).", justify="left").pack(anchor="w", padx=10, pady=10)


    def create_hrv_freq_plots(self, fxx, pxx, features):
        """Plots the RR PSD, Autonomic Balance, and displays frequency features."""

        # 1. Create the plot (2 subplots)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8)) # Adjusted figsize
        fig.patch.set_facecolor('#2B2B2B')

        # --- Plot 1: PSD ---
        ax1.plot(fxx, pxx, color='cyan')
        ax1.set_title(f"RR Interval Power Spectral Density (Welch's)", color='white')
        ax1.set_xlabel("Frequency (Hz)", color='white')
        ax1.set_ylabel("Power (s^2/Hz)", color='white') # Use ^2
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#3C3C3C')
        ax1.axvspan(0.003, 0.04, color='blue', alpha=0.3, label='VLF (0.003-0.04 Hz)')
        ax1.axvspan(0.04, 0.15, color='green', alpha=0.3, label='LF (0.04-0.15 Hz)')
        ax1.axvspan(0.15, 0.4, color='red', alpha=0.3, label='HF (0.15-0.4 Hz)')
        ax1.legend(fontsize='small')
        ax1.set_xlim(0, 0.5)

        # --- Plot 2: Autonomic Balance Diagram (3x3 grid) ---
        lfnu = features.get('LFnu', 0)
        hfnu = features.get('HFnu', 0)
        ax2.plot(lfnu, hfnu, marker='o', markersize=10, color='lime', linestyle='')
        ax2.set_title("Autonomic Balance Diagram", color='white')
        ax2.set_xlabel("Normalized LF Power (LFnu %)", color='white')
        ax2.set_ylabel("Normalized HF Power (HFnu %)", color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#3C3C3C')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle='--', alpha=0.6)
        # Add reference lines for 3x3 grid
        ax2.plot([0, 100], [0, 100], color='gray', linestyle='--') # x=y line
        ax2.axvline(33.3, color='gray', linestyle=':')
        ax2.axvline(66.6, color='gray', linestyle=':')
        ax2.axhline(33.3, color='gray', linestyle=':')
        ax2.axhline(66.6, color='gray', linestyle=':')
        # Add text labels (simplified for 3x3 grid)
        ax2.text(83, 17, 'Sympathetic', color='orange', ha='center', va='center', alpha=0.7)
        ax2.text(17, 83, 'Parasympathetic', color='cyan', ha='center', va='center', alpha=0.7)
        ax2.text(17, 17, 'Low Tone', color='gray', ha='center', va='center', alpha=0.7)
        ax2.text(50, 50, 'Balanced', color='white', ha='center', va='center', alpha=0.7)
        ax2.text(83, 83, 'High Tone', color='gray', ha='center', va='center', alpha=0.7)
        ax2.set_aspect('equal', adjustable='box')


        fig.tight_layout(h_pad=3.0) # Add vertical padding
        self.hrv_freq_canvas_widget = self.embed_plot(fig, self.hrv_freq_plot_frame)

        # 2. Create the results labels
        font_settings = ("Roboto", 14)
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"VLF Power: {features.get('VLF', 0):.4f} s^2", font=font_settings).pack(anchor="w", padx=10) # Use ^2
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"LF Power: {features.get('LF', 0):.4f} s^2", font=font_settings).pack(anchor="w", padx=10) # Use ^2
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"HF Power: {features.get('HF', 0):.4f} s^2", font=font_settings).pack(anchor="w", padx=10) # Use ^2
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"Total Power (LF+HF): {features.get('Total Power (LF+HF)', 0):.4f} s^2", font=font_settings).pack(anchor="w", padx=10) # Use ^2
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"LF/HF Ratio: {features.get('LF/HF Ratio', 0):.3f}", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"Normalized LF (LFnu): {features.get('LFnu', 0):.2f} %", font=font_settings).pack(anchor="w", padx=10)
        customtkinter.CTkLabel(self.hrv_freq_results_frame, text=f"Normalized HF (HFnu): {features.get('HFnu', 0):.2f} %", font=font_settings).pack(anchor="w", padx=10)


    def create_dwt_rate_plots(self, rr_signal, rr_peaks, bpm, rr_level,
                          vr_freqs, vr_mag, vr_peak_freq, vr_peak_mag, vr_level):
        """Plots the DWT-based RR peaks and VR FFT."""

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        fig.patch.set_facecolor('#2B2B2B')

        fs_level_rr = self.fs / (2**rr_level)
        duration_s_rr = len(rr_signal) / fs_level_rr
        t_level_rr = np.linspace(0, duration_s_rr, len(rr_signal))

        # Respiratory Peaks Plot
        f_low_rr, f_high_rr = self.get_freq_range(rr_level)
        bpm_title = f"Respiratory Rate: {bpm:.1f} BPM (from d{rr_level}: [{f_low_rr:.3f}-{f_high_rr:.3f} Hz])"

        ax1.plot(t_level_rr, rr_signal, color='lime', label=f'd{rr_level} Signal')
        ax1.plot(t_level_rr[rr_peaks], rr_signal[rr_peaks], 'x', color='red', label='Detected Peaks')
        ax1.set_title(bpm_title, color='white')
        ax1.set_xlabel("Time (s)", color='white')
        ax1.set_ylabel("Amplitude", color='white')
        ax1.legend()
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#3C3C3C')

        # Vasometric FFT Plot
        f_low_vr, f_high_vr = self.get_freq_range(vr_level)
        vaso_title = f"Vasometric Rate FFT (from d{vr_level}: [{f_low_vr:.3f}-{f_high_vr:.3f} Hz])"

        ax2.plot(vr_freqs, vr_mag, color='yellow', label=f'FFT of d{vr_level}')
        ax2.plot(vr_peak_freq, vr_peak_mag, 'x', color='red', label=f'Peak: {vr_peak_freq:.3f} Hz')
        ax2.set_title(vaso_title, color='white')
        ax2.set_xlabel("Frequency (Hz)", color='white')
        ax2.set_ylabel("Magnitude", color='white')
        ax2.set_xlim(0, 0.5)
        ax2.legend()
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#3C3C3C')

        fig.tight_layout()
        self.dwt_rate_canvas_widget = self.embed_plot(fig, self.dwt_rate_plot_frame)

    def plot_filter_response(self):
        """Calculates and plots the cascaded frequency response using the notebook's manual method."""
        if self.fs <= 0: return

        if self.filter_canvas_widget: self.filter_canvas_widget.destroy()

        try:
            fs_calc = self.fs
            h = self.h_coeffs
            g = self.g_coeffs

            # --- 1. Calculate Hw and Gw ---
            max_freq_index = 128 * int(np.round(fs_calc / 2))
            array_size = max(20000, max_freq_index + 1)
            Hw = np.zeros(array_size)
            Gw = np.zeros(array_size)

            for i_freq in range(0, array_size):
                reG, imG, reH, imH = 0.0, 0.0, 0.0, 0.0
                for k_idx, k in enumerate(self.n_range):
                    angle = k * 2 * np.pi * i_freq / fs_calc
                    reG += g[k_idx] * np.cos(angle)
                    imG -= g[k_idx] * np.sin(angle)
                    reH += h[k_idx] * np.cos(angle)
                    imH -= h[k_idx] * np.sin(angle)
                Hw[i_freq] = np.sqrt((reH**2) + (imH**2))
                Gw[i_freq] = np.sqrt((reG**2) + (imG**2))

            # --- 2. Calculate Q ---
            n_plot_points = int(np.round(fs_calc / 2)) + 1
            Q = np.zeros((8, n_plot_points))
            i_vals_plot = np.arange(0, n_plot_points)

            for i in i_vals_plot:
                # Need to check bounds because index can exceed array_size
                Q[0][i] = Gw[i] if i < array_size else 0
                Q[1][i] = Gw[2*i] * Hw[i] if 2*i < array_size and i < array_size else 0
                Q[2][i] = Gw[4*i] * Hw[2*i] * Hw[i] if 4*i < array_size and 2*i < array_size and i < array_size else 0
                Q[3][i] = Gw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
                Q[4][i] = Gw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
                Q[5][i] = Gw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
                Q[6][i] = Gw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 64*i < array_size and 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0
                Q[7][i] = Gw[128*i] * Hw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i] if 128*i < array_size and 64*i < array_size and 32*i < array_size and 16*i < array_size and 8*i < array_size and 4*i < array_size and 2*i < array_size and i < array_size else 0


            # --- 3. Plot Q ---
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#2B2B2B')
            for j in range(8):
                ax.plot(i_vals_plot, Q[j], label=f"Q{j+1}")
            ax.set_title(f'DWT Cascaded Filter Response (fs = {self.fs:.1f} Hz)', color='white')
            ax.set_xlabel('Frequency (Hz)', color='white')
            ax.set_ylabel('Magnitude', color='white')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(colors='white')
            ax.set_facecolor('#3C3C3C')
            fig.tight_layout()

            self.filter_canvas_widget = self.embed_plot(fig, self.filter_plot_frame)

        except Exception as e:
            self.show_error_in_filter_tab(f"Error plotting filter response: {e}")


if __name__ == "__main__":
    app = PPGVisualizerApp()
    app.mainloop()