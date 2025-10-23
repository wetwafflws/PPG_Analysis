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
        customtkinter.set_appearance_mode("Light")
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

    def dirac(self, x):
        return 1 if x == 0 else 0

    def dwt(self, x, J):
        length = len(x)
        d_coeffs_list = []
        for j in range(1, J+1):
            start_k = -(round(2**j) + round(2**(j-1)) - 2)
            end_k = -(1 - round(2**(j-1)))
            q_values = []
            for k in range(start_k, end_k + 1):
                if j == 1:
                    q_k = -2 * (self.dirac(k) - self.dirac(k+1))
                elif j == 2:
                    q_k = -1/4 * (self.dirac(k-1) + 3*self.dirac(k) + 2*self.dirac(k+1) - 2*self.dirac(k+2) - 3*self.dirac(k+3) - self.dirac(k+4))
                elif j == 3:
                    q_k = -1/32 * (self.dirac(k-3) + 3*self.dirac(k-2) + 6*self.dirac(k-1) + 10*self.dirac(k) + 11*self.dirac(k+1) + 9*self.dirac(k+2) + 4*self.dirac(k+3) - 4*self.dirac(k+4) - 9*self.dirac(k+5) - 11*self.dirac(k+6) - 10*self.dirac(k+7) - 6*self.dirac(k+8) - 3*self.dirac(k+9) - self.dirac(k+10))
                elif j == 4:
                    q_k = -1/256 * (self.dirac(k-7) + 3*self.dirac(k-6) + 6*self.dirac(k-5) + 10*self.dirac(k-4) + 15*self.dirac(k-3) + 21*self.dirac(k-2) + 28*self.dirac(k-1) + 36*self.dirac(k) + 41*self.dirac(k+1) + 43*self.dirac(k+2) + 42*self.dirac(k+3) + 38*self.dirac(k+4) + 31*self.dirac(k+5) + 21*self.dirac(k+6) + 8*self.dirac(k+7) - 8*self.dirac(k+8) - 21*self.dirac(k+9) - 31*self.dirac(k+10) - 38*self.dirac(k+11) - 42*self.dirac(k+12) - 43*self.dirac(k+13) - 41*self.dirac(k+14) - 36*self.dirac(k+15) - 28*self.dirac(k+16) - 21*self.dirac(k+17) - 15*self.dirac(k+18) - 10*self.dirac(k+19) - 6*self.dirac(k+20) - 3*self.dirac(k+21) - self.dirac(k+22))
                elif j == 5:
                    q_k = -1/2048 * (self.dirac(k-15) + 3*self.dirac(k-14) + 6*self.dirac(k-13) + 10*self.dirac(k-12) + 15*self.dirac(k-11) + 21*self.dirac(k-10) + 28*self.dirac(k-9) + 36*self.dirac(k-8) + 45*self.dirac(k-7) + 55*self.dirac(k-6) + 66*self.dirac(k-5) + 78*self.dirac(k-4) + 91*self.dirac(k-3) + 105*self.dirac(k-2) + 120*self.dirac(k-1) + 136*self.dirac(k) + 149*self.dirac(k+1) + 159*self.dirac(k+2) + 166*self.dirac(k+3) + 170*self.dirac(k+4) + 171*self.dirac(k+5) + 169*self.dirac(k+6) + 164*self.dirac(k+7) + 156*self.dirac(k+8) + 145*self.dirac(k+9) + 131*self.dirac(k+10) + 114*self.dirac(k+11) + 94*self.dirac(k+12) + 71*self.dirac(k+13) + 45*self.dirac(k+14) + 16*self.dirac(k+15) - 16*self.dirac(k+16) - 45*self.dirac(k+17) - 71*self.dirac(k+18) - 94*self.dirac(k+19) - 114*self.dirac(k+20) - 131*self.dirac(k+21) - 145*self.dirac(k+22) - 156*self.dirac(k+23) - 164*self.dirac(k+24) - 169*self.dirac(k+25) - 171*self.dirac(k+26) - 170*self.dirac(k+27) - 166*self.dirac(k+28) - 159*self.dirac(k+29) - 149*self.dirac(k+30) - 136*self.dirac(k+31) - 120*self.dirac(k+32) - 105*self.dirac(k+33) - 91*self.dirac(k+34) - 78*self.dirac(k+35) - 66*self.dirac(k+36) - 55*self.dirac(k+37) - 45*self.dirac(k+38) - 36*self.dirac(k+39) - 28*self.dirac(k+40) - 21*self.dirac(k+41) - 15*self.dirac(k+42) - 10*self.dirac(k+43) - 6*self.dirac(k+44) - 3*self.dirac(k+45) - self.dirac(k+46))
                elif j == 6:
                    coeffs = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 557, 583, 606, 626, 643, 657, 668, 676, 681, 683, 682, 678, 671, 661, 648, 632, 613, 591, 566, 538, 507, 473, 436, 396, 353, 307, 258, 206, 151, 93, 32, -32, -93, -151, -206, -258, -307, -353, -396, -436, -473, -507, -538, -566, -591, -613, -632, -648, -661, -671, -678, -682, -683, -681, -676, -668, -657, -643, -626, -606, -583, -557, -528, -496, -465, -435, -406, -378, -351, -325, -300, -276, -253, -231, -210, -190, -171, -153, -136, -120, -105, -91, -78, -66, -55, -45, -36, -28, -21, -15, -10, -6, -3, -1]
                    q_k = 0
                    for idx, coef in enumerate(coeffs):
                        q_k += coef * self.dirac(k - 31 + idx)
                    q_k *= -1 / 16384
                elif j == 7:
                    coeffs = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2141, 2199, 2254, 2306, 2355, 2401, 2444, 2484, 2521, 2555, 2586, 2614, 2639, 2661, 2680, 2696, 2709, 2719, 2726, 2730, 2731, 2729, 2724, 2716, 2705, 2691, 2674, 2654, 2631, 2605, 2576, 2544, 2509, 2471, 2430, 2386, 2339, 2289, 2236, 2180, 2121, 2059, 1994, 1926, 1855, 1781, 1704, 1624, 1541, 1455, 1366, 1274, 1179, 1081, 980, 876, 769, 659, 546, 430, 311, 189, 64, -64, -189, -311, -430, -546, -659, -769, -876, -980, -1081, -1179, -1274, -1366, -1455, -1541, -1624, -1704, -1781, -1855, -1926, -1994, -2059, -2121, -2180, -2236, -2289, -2339, -2386, -2430, -2471, -2509, -2544, -2576, -2605, -2631, -2654, -2674, -2691, -2705, -2716, -2724, -2729, -2731, -2730, -2726, -2719, -2709, -2696, -2680, -2661, -2639, -2614, -2586, -2555, -2521, -2484, -2444, -2401, -2355, -2306, -2254, -2199, -2141, -2080, -2016, -1953, -1891, -1830, -1770, -1711, -1653, -1596, -1540, -1485, -1431, -1378, -1326, -1275, -1225, -1176, -1128, -1081, -1035, -990, -946, -903, -861, -820, -780, -741, -703, -666, -630, -595, -561, -528, -496, -465, -435, -406, -378, -351, -325, -300, -276, -253, -231, -210, -190, -171, -153, -136, -120, -105, -91, -78, -66, -55, -45, -36, -28, -21, -15, -10, -6, -3, -1]
                    q_k = 0
                    for idx, coef in enumerate(coeffs):
                        q_k += coef * self.dirac(k - 63 + idx)
                    q_k *= -1 / 131072
                elif j == 8:
                    coeffs = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128, 8256, 8381, 8503, 8622, 8738, 8851, 8961, 9068, 9172, 9273, 9371, 9466, 9558, 9647, 9733, 9816, 9896, 9973, 10047, 10118, 10186, 10251, 10313, 10372, 10428, 10481, 10531, 10578, 10622, 10663, 10701, 10736, 10768, 10797, 10823, 10846, 10866, 10883, 10897, 10908, 10916, 10921, 10923, 10922, 10918, 10911, 10901, 10888, 10872, 10853, 10831, 10806, 10778, 10747, 10713, 10676, 10636, 10593, 10547, 10498, 10446, 10391, 10333, 10272, 10208, 10141, 10071, 9998, 9922, 9843, 9761, 9676, 9588, 9497, 9403, 9306, 9206, 9103, 8997, 8888, 8776, 8661, 8543, 8422, 8298, 8171, 8041, 7908, 7772, 7633, 7491, 7346, 7198, 7047, 6893, 6736, 6576, 6413, 6247, 6078, 5906, 5731, 5553, 5372, 5188, 5001, 4811, 4618, 4422, 4223, 4021, 3816, 3608, 3397, 3183, 2966, 2746, 2523, 2297, 2068, 1836, 1601, 1363, 1122, 878, 631, 381, 128, -128, -381, -631, -878, -1122, -1363, -1601, -1836, -2068, -2297, -2523, -2746, -2966, -3183, -3397, -3608, -3816, -4021, -4223, -4422, -4618, -4811, -5001, -5188, -5372, -5553, -5731, -5906, -6078, -6247, -6413, -6576, -6736, -6893, -7047, -7198, -7346, -7491, -7633, -7772, -7908, -8041, -8171, -8298, -8422, -8543, -8661, -8776, -8888, -8997, -9103, -9206, -9306, -9403, -9497, -9588, -9676, -9761, -9843, -9922, -9998, -10071, -10141, -10208, -10272, -10333, -10391, -10446, -10498, -10547, -10593, -10636, -10676, -10713, -10747, -10778, -10806, -10831, -10853, -10872, -10888, -10901, -10911, -10918, -10922, -10923, -10921, -10916, -10908, -10897, -10883, -10866, -10846, -10823, -10797, -10768, -10736, -10701, -10663, -10622, -10578, -10531, -10481, -10428, -10372, -10313, -10251, -10186, -10118, -10047, -9973, -9896, -9816, -9733, -9647, -9558, -9466, -9371, -9172, -9068, -8961, -8851, -8738, -8622, -8503, -8381, -8256, -8128, -8001, -7875, -7750, -7626, -7503, -7381, -7260, -7140, -7021, -6903, -6786, -6670, -6555, -6441, -6328, -6216, -6105, -5995, -5886, -5778, -5671, -5565, -5460, -5356, -5253, -5151, -5050, -4950, -4851, -4753, -4656, -4560, -4465, -4371, -4278, -4186, -4095, -4005, -3916, -3828, -3741, -3655, -3570, -3486, -3403, -3321, -3240, -3160, -3081, -3003, -2926, -2850, -2775, -2701, -2628, -2556, -2485, -2415, -2346, -2278, -2211, -2145, -2080, -2016, -1953, -1891, -1830, -1770, -1711, -1653, -1596, -1540, -1485, -1431, -1378, -1326, -1275, -1225, -1176, -1128, -1081, -1035, -990, -946, -903, -861, -820, -780, -741, -703, -666, -630, -595, -561, -528, -496, -465, -435, -406, -378, -351, -325, -300, -276, -253, -231, -210, -190, -171, -153, -136, -120, -105, -91, -78, -66, -55, -45, -36, -28, -21, -15, -10, -6, -3, -1]
                    q_k = 0
                    for idx, coef in enumerate(coeffs):
                        q_k += coef * self.dirac(k - 127 + idx)
                    q_k *= -1 / 1048576
                q_values.append(q_k)
            d_j = np.convolve(q_values, x, mode='same')
            d_coeffs_list.append(d_j)
        return None, d_coeffs_list

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
            self.filtered_infrared = self.preprocess_signal(self.raw_infrared, fs=self.fs)

            self.fs_entry.delete(0, 'end')
            self.fs_entry.insert(0, str(self.fs))

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

        _, d_coeffs_list = self.dwt(self.filtered_infrared, J=8)
        self.dwt_coeffs = d_coeffs_list
        self.create_preprocessing_plots()

        default_dwt_scale = "Scale 1"
        self.dwt_option_menu.set(default_dwt_scale)
        self.update_dwt_plot(default_dwt_scale)

        self.rr_option_menu.set("Scale 7")
        self.vr_option_menu.set("Scale 8")

    # --- RR Interval Analysis (Time and Freq) ---

    def _get_rr_intervals(self):
        """Helper function to find peaks and return RR intervals."""
        if len(self.filtered_infrared) == 0:
            raise ValueError("No data loaded. Please load a CSV file.")

        # Apply 4th order Butterworth bandpass 0.1-40 Hz for HRV
        nyquist = self.fs / 2
        lowcut = 0.1
        highcut = min(40.0, nyquist - 1.0)
        b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        hrv_signal = filtfilt(b, a, self.filtered_infrared)

        min_dist_samples = int(self.fs * 0.4) # Min dist 0.4s (max 150 BPM)
        peak_indices, _ = find_peaks(hrv_signal, height=0, distance=max(1, min_dist_samples))

        if len(peak_indices) < 2:
            raise ValueError("Not enough peaks found (< 2) to calculate RR intervals.")

        peak_times_sec = peak_indices / self.fs
        rr_intervals_sec = np.diff(peak_times_sec)
        rr_intervals_ms = rr_intervals_sec * 1000

        if len(rr_intervals_sec) < 2:
            raise ValueError("Not enough RR intervals (< 2) for analysis.")

        return peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms, hrv_signal

    def run_hrv_time_analysis(self):
        """Finds peaks and calculates time-domain HRV features."""
        if self.hrv_time_canvas_widget:
            self.hrv_time_canvas_widget.destroy()
            self.hrv_time_canvas_widget = None
        for widget in self.hrv_time_results_frame.winfo_children():
            widget.destroy()

        try:
            peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms, hrv_signal = self._get_rr_intervals()

            features = self.calculate_time_domain_features(rr_intervals_ms)

            self.create_hrv_time_plots(peak_indices, peak_times_sec, rr_intervals_ms, features, hrv_signal)

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
            peak_indices, peak_times_sec, rr_intervals_sec, rr_intervals_ms, _ = self._get_rr_intervals()

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
        """Applies normalization."""
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std
        return normalized

    def analyze_respiratory(self, d_signal, level):
        """Finds respiratory peaks from the selected d-signal."""
        fs_level = self.fs

        min_distance_samples = fs_level * 0.5
        safe_distance = max(1, int(np.ceil(min_distance_samples)))

        # Apply moving average filter to reduce spurious peaks
        window_size = int(fs_level * 0.5)  # 0.5 second window
        if window_size > 1:
            smoothed_signal = np.convolve(d_signal, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_signal = d_signal

        peaks, _ = find_peaks(smoothed_signal, prominence=np.mean(np.abs(smoothed_signal)) * 0.2, distance=safe_distance)

        duration_s = len(d_signal) / fs_level
        num_peaks = len(peaks)
        bpm = (num_peaks / duration_s) * 60 if duration_s > 0 else 0

        return d_signal, peaks, bpm

    def analyze_vasometric(self, d_signal, level):
        """Calculates FFT for the selected d-signal."""
        n = len(d_signal)
        fs_level = self.fs

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

            fs_level = self.fs
            duration_s = len(signal_data) / fs_level
            t_level = np.linspace(0, duration_s, len(signal_data))

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')

            ax.plot(t_level, signal_data, color='green')
            title = f"DWT d{scale_j} Coefficient (Freq: {f_low:.3f} - {f_high:.3f} Hz)"
            ax.set_title(title, color='black')
            ax.set_xlabel("Time (s)", color='black')
            ax.set_ylabel("Amplitude", color='black')
            ax.tick_params(colors='black')
            ax.set_facecolor('white')
            fig.tight_layout()

            self.dwt_canvas_widget = self.embed_plot(fig, self.dwt_plot_frame)

        except Exception as e:
            err_label = customtkinter.CTkLabel(self.dwt_plot_frame, text=f"Error plotting scale: {e}", text_color="red")
            err_label.pack(pady=20, padx=20)
            self.dwt_canvas_widget = err_label

    def create_preprocessing_plots(self):
        """Plots raw and filtered signals."""
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        fig.patch.set_facecolor('white')

        ax1.plot(self.t, self.raw_infrared, color='red')
        ax1.set_title(f"Raw Infrared Signal (fs = {self.fs:.1f} Hz)", color='black')
        ax1.set_xlabel("Time (s)", color='black')
        ax1.set_ylabel("Amplitude", color='black')
        ax1.tick_params(colors='black')
        ax1.set_facecolor('white')

        ax2.plot(self.t, self.filtered_infrared, color='blue')
        ax2.set_title("Preprocessed (Filtered) Infrared Signal", color='black')
        ax2.set_xlabel("Time (s)", color='black')
        ax2.set_ylabel("Normalized Amplitude", color='black')
        ax2.tick_params(colors='black')
        ax2.set_facecolor('white')

        fig.tight_layout()
        widget = self.embed_plot(fig, self.preprocess_plot_frame)
        self.canvas_list.append(widget)

    def create_hrv_time_plots(self, peak_indices, peak_times_sec, rr_intervals_ms, features, hrv_signal):
        """Plots peak detection, tachogram, Poincaré, and displays features."""

        # 1. Create the plot (4 subplots)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10, 12)) # Adjusted figsize
        fig.patch.set_facecolor('white')

        # --- Plot 1: Peak Detection ---
        ax1.plot(self.t, hrv_signal, color='red', label='Bandpassed Signal')
        ax1.plot(self.t[peak_indices], hrv_signal[peak_indices], 'x', color='blue', markersize=8, label='Detected Peaks')
        ax1.set_title(f"Peak Detection on Bandpassed Signal", color='black')
        ax1.set_ylabel("Amplitude", color='black')
        ax1.tick_params(colors='black', labelbottom=False) # Hide x-labels
        ax1.set_facecolor('white')
        ax1.legend()

        # --- Plot 2: RR Tachogram ---
        ax2.plot(peak_times_sec[1:], rr_intervals_ms, marker='o', linestyle='-', markersize=4, color='blue')
        ax2.set_title(f"RR Interval Tachogram", color='black')
        ax2.set_ylabel("RR Interval (ms)", color='black')
        ax2.tick_params(colors='black', labelbottom=False) # Hide x-labels
        ax2.set_facecolor('white')
        ax2.sharex(ax1) # Link x-axis with ax1

        # --- Plot 3: Poincaré Plot ---
        rr_n = rr_intervals_ms[:-1]
        rr_n1 = rr_intervals_ms[1:]
        ax3.scatter(rr_n, rr_n1, color='lime', alpha=0.5, s=10) # Use scatter
        ax3.set_title("Poincaré Plot", color='black')
        ax3.set_xlabel("RRn (ms)", color='black')
        ax3.set_ylabel("RRn+1 (ms)", color='black')
        ax3.tick_params(colors='black')
        ax3.set_facecolor('white')
        min_rr = np.min(rr_intervals_ms)
        max_rr = np.max(rr_intervals_ms)
        ax3.plot([min_rr, max_rr], [min_rr, max_rr], color='gray', linestyle='--')
        ax3.set_aspect('equal', adjustable='box')

        # Add semi-transparent ellipse for SD1 and SD2
        sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
        sd2 = np.sqrt(2 * np.var(rr_intervals_ms) - sd1**2)
        from matplotlib.patches import Ellipse
        center_x = np.mean(rr_n)
        center_y = np.mean(rr_n1)
        ellipse = Ellipse((center_x, center_y), width=2*sd2, height=2*sd1, angle=45, alpha=0.3, color='red')
        ax3.add_patch(ellipse)

        # --- Plot 4: RR Interval Histogram ---
        mean_rr = np.mean(rr_intervals_ms)
        diff = rr_intervals_ms - mean_rr
        numerator = np.mean(diff**3)
        denominator = (np.sqrt(np.mean(diff**2)))**3
        skewness = numerator / denominator if denominator != 0 else 0
        ax4.hist(rr_intervals_ms, bins=20, alpha=0.7, color='blue')
        ax4.set_title(f"RR Interval Histogram (Skewness: {skewness:.3f})", color='black')
        ax4.set_xlabel("RR Interval (ms)", color='black')
        ax4.set_ylabel("Frequency", color='black')
        ax4.tick_params(colors='black')
        ax4.set_facecolor('white')

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
        fig.patch.set_facecolor('white')

        # --- Plot 1: PSD ---
        ax1.plot(fxx, pxx, color='red')
        ax1.set_title(f"RR Interval Power Spectral Density (Welch's)", color='black')
        ax1.set_xlabel("Frequency (Hz)", color='black')
        ax1.set_ylabel("Power (s^2/Hz)", color='black') # Use ^2
        ax1.tick_params(colors='black')
        ax1.set_facecolor('white')
        ax1.axvspan(0.003, 0.04, color='blue', alpha=0.3, label='VLF (0.003-0.04 Hz)')
        ax1.axvspan(0.04, 0.15, color='green', alpha=0.3, label='LF (0.04-0.15 Hz)')
        ax1.axvspan(0.15, 0.4, color='red', alpha=0.3, label='HF (0.15-0.4 Hz)')
        ax1.legend(fontsize='small')
        ax1.set_xlim(0, 0.5)

        # --- Plot 2: Autonomic Balance Diagram (3x3 grid) ---
        lfnu = features.get('LFnu', 0)
        hfnu = features.get('HFnu', 0)
        ax2.plot(lfnu, hfnu, marker='o', markersize=10, color='lime', linestyle='')
        ax2.set_title("Autonomic Balance Diagram", color='black')
        ax2.set_xlabel("Normalized LF Power (LFnu %)", color='black')
        ax2.set_ylabel("Normalized HF Power (HFnu %)", color='black')
        ax2.tick_params(colors='black')
        ax2.set_facecolor('white')
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
        ax2.text(17, 83, 'Parasympathetic', color='red', ha='center', va='center', alpha=0.7)
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
        fig.patch.set_facecolor('white')

        fs_level_rr = self.fs
        duration_s_rr = len(rr_signal) / fs_level_rr
        t_level_rr = np.linspace(0, duration_s_rr, len(rr_signal))

        # Respiratory Peaks Plot
        f_low_rr, f_high_rr = self.get_freq_range(rr_level)
        bpm_title = f"Respiratory Rate: {bpm:.1f} BPM (from d{rr_level}: [{f_low_rr:.3f}-{f_high_rr:.3f} Hz])"

        ax1.plot(t_level_rr, rr_signal, color='lime', label=f'd{rr_level} Signal')
        ax1.plot(t_level_rr[rr_peaks], rr_signal[rr_peaks], 'x', color='red', label='Detected Peaks')
        ax1.set_title(bpm_title, color='black')
        ax1.set_xlabel("Time (s)", color='black')
        ax1.set_ylabel("Amplitude", color='black')
        ax1.legend()
        ax1.tick_params(colors='black')
        ax1.set_facecolor('white')

        # Vasometric FFT Plot
        f_low_vr, f_high_vr = self.get_freq_range(vr_level)
        vaso_title = f"Vasometric Rate FFT (from d{vr_level}: [{f_low_vr:.3f}-{f_high_vr:.3f} Hz])"

        ax2.plot(vr_freqs, vr_mag, color='blue', label=f'FFT of d{vr_level}')
        ax2.plot(vr_peak_freq, vr_peak_mag, 'x', color='red', label=f'Peak: {vr_peak_freq:.3f} Hz')
        ax2.set_title(vaso_title, color='black')
        ax2.set_xlabel("Frequency (Hz)", color='black')
        ax2.set_ylabel("Magnitude", color='black')
        ax2.set_xlim(0, 0.5)
        ax2.legend()
        ax2.tick_params(colors='black')
        ax2.set_facecolor('white')

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
            fig.patch.set_facecolor('white')
            for j in range(8):
                ax.plot(i_vals_plot, Q[j], label=f"Q{j+1}")
            ax.set_title(f'DWT Cascaded Filter Response (fs = {self.fs:.1f} Hz)', color='black')
            ax.set_xlabel('Frequency (Hz)', color='black')
            ax.set_ylabel('Magnitude', color='black')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(colors='black')
            ax.set_facecolor('white')
            fig.tight_layout()

            self.filter_canvas_widget = self.embed_plot(fig, self.filter_plot_frame)

        except Exception as e:
            self.show_error_in_filter_tab(f"Error plotting filter response: {e}")


if __name__ == "__main__":
    app = PPGVisualizerApp()
    app.mainloop()