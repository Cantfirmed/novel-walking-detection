"""
walking_detection.py

A clean, readable implementation of the walking detection and step counting
algorithm from the paper:
"A Novel Walking Detection and Step Counting Algorithm Using Unconstrained Smartphones"
(Sensors 2018, 18, 297) by Xiaomin Kang, Baoqi Huang, and Guodong Qi.

The algorithm consists of:
1. Resampling 3D gyroscope data to 20 Hz.
2. Sliding time window of N = 64 samples (3.2 seconds) with a 1.2s step.
3. Axis Selection: Choosing the most sensitive axis based on L1 norm (Eq. 1).
4. Spectrum Analysis: 64-point FFT (Eq. 2-3).
5. Walking Detection: Dual threshold condition on walking vs. low frequency bands (Eq. 4-5).
6. Frequency Estimation: 4th-order polynomial curve fitting (Eq. 7).
7. Smoothing & Step Counting: Exponential moving average and accumulation (Eq. 6, 8).
"""

import numpy as np
import pandas as pd


def resample_gyro_data(df: pd.DataFrame, target_fs: float = 20.0):
    """
    Resample the non-uniformly sampled sensor time series to a uniform sampling
    rate (default: 20 Hz as specified in Section 4.1 of the paper).
    """
    t_start = df['time'].iloc[0]
    t_end = df['time'].iloc[-1]
    dt = 1.0 / target_fs
    
    t_uniform = np.arange(t_start, t_end, dt)
    wx = np.interp(t_uniform, df['time'], df['wx'])
    wy = np.interp(t_uniform, df['time'], df['wy'])
    wz = np.interp(t_uniform, df['time'], df['wz'])
    
    return t_uniform, wx, wy, wz


def select_sensitive_axis(win_x: np.ndarray, win_y: np.ndarray, win_z: np.ndarray) -> np.ndarray:
    """
    Section 4.2, Equation (1):
    Select the axis whose data has the maximum sum of absolute angular velocities:
        max sum(|w_a(i)|) for a in {x, y, z}
    """
    sum_x = np.sum(np.abs(win_x))
    sum_y = np.sum(np.abs(win_y))
    sum_z = np.sum(np.abs(win_z))
    
    if sum_x >= sum_y and sum_x >= sum_z:
        return win_x
    elif sum_y >= sum_x and sum_y >= sum_z:
        return win_y
    else:
        return win_z


def detect_walking_and_count_steps(
    csv_file_path: str,
    fs: float = 20.0,
    window_size: int = 64,
    step_duration: float = 1.2,
    alpha: float = 0.8,
    min_amplitude_threshold: float = 10.0,
    stride_to_step_factor: float = 2.0
):
    """
    Executes the paper's complete pipeline on a CSV recording:
    
    Parameters:
    - csv_file_path: path to CSV with columns ['time', 'wx', 'wy', 'wz']
    - fs: target sampling frequency in Hz (default 20 Hz, from paper)
    - window_size: number of points per window (default 64, from paper)
    - step_duration: sliding step in seconds (default 1.2s, from paper)
    - alpha: smoothing weight for frequency moving average (default 0.8, Eq. 8)
    - min_amplitude_threshold: threshold for average amplitude (default 10.0, Eq. 5)
    - stride_to_step_factor: 2.0 if walking frequency represents gait cycles (strides),
      since 1 stride = 2 steps (left + right). Set to 1.0 if counting gait cycles.
    
    Returns:
    - results: dictionary containing summary metrics and per-window detections
    """
    df = pd.read_csv(csv_file_path)
    t_uniform, wx, wy, wz = resample_gyro_data(df, target_fs=fs)
    
    step_samples = int(round(step_duration * fs))  # 1.2s * 20 Hz = 24 samples
    
    # Frequency bin calculation (Section 4.3, Eq. 3):
    # f_k = k * fs / N = k * (20 / 64) = k * 0.3125 Hz
    #
    # Walking frequency band: [0.6 Hz, 2.0 Hz]
    # k = 2 (0.625 Hz), 3 (0.9375 Hz), 4 (1.25 Hz), 5 (1.5625 Hz), 6 (1.875 Hz) -> 5 points
    walking_band_indices = [2, 3, 4, 5, 6]
    walking_band_freqs = np.array([k * fs / window_size for k in walking_band_indices])
    
    # Low frequency band: [0 Hz, 0.6 Hz) -> k = 0 (0.0 Hz), 1 (0.3125 Hz)
    low_band_indices = [0, 1]
    
    # Dense grid for finding the polynomial maximum in [0.6, 2.0] Hz
    f_dense = np.linspace(0.6, 2.0, 200)
    
    total_steps = 0.0
    walking_duration = 0.0
    f_smooth = None
    window_records = []
    
    # Slide the window across the recording
    num_windows = (len(t_uniform) - window_size) // step_samples + 1
    
    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_size
        win_time_center = t_uniform[start_idx + window_size // 2]
        
        # 1. Select the most sensitive axis (Eq. 1)
        win_axis = select_sensitive_axis(
            wx[start_idx:end_idx],
            wy[start_idx:end_idx],
            wz[start_idx:end_idx]
        )
        
        # 2. Compute 64-point discrete Fourier transform (Eq. 2)
        X = np.fft.fft(win_axis, n=window_size)
        amplitudes = np.abs(X)
        
        # Average amplitude in typical walking band (0.6 - 2.0 Hz)
        w_c = np.mean(amplitudes[walking_band_indices])
        
        # Average amplitude in low-frequency band (0 - 0.6 Hz)
        w_0 = np.mean(amplitudes[low_band_indices])
        
        # 3. Walking Detection (Section 4.3, Eq. 4 & Eq. 5)
        # Condition 1: w_c > w_0 (cyclic walking energy dominates low-frequency drift)
        # Condition 2: w_c > 10.0 (filters out idle motions like stationary typing)
        is_walking = (w_c > w_0) and (w_c > min_amplitude_threshold)
        
        estimated_freq = None
        if is_walking:
            walking_duration += step_duration
            
            # 4. Polynomial curve fitting (Section 5, Eq. 7):
            # A(f) = a*f^4 + b*f^3 + c*f^2 + d*f + e
            amp_points = amplitudes[walking_band_indices]
            poly_coeffs = np.polyfit(walking_band_freqs, amp_points, deg=4)
            poly_curve = np.polyval(poly_coeffs, f_dense)
            
            # Find the frequency that maximizes amplitude A in [0.6, 2.0] Hz
            f_hat = f_dense[np.argmax(poly_curve)]
            
            # 5. Weighted moving average smoothing (Section 5, Eq. 8):
            # f_bar_i = alpha * f_bar_{i-1} + (1 - alpha) * f_hat_i
            if f_smooth is None:
                f_smooth = f_hat
            else:
                f_smooth = alpha * f_smooth + (1.0 - alpha) * f_hat
                
            estimated_freq = f_smooth
            
            # 6. Step count accumulation (Section 5, Eq. 6):
            # c = t * fw * stride_to_step_factor
            steps_in_window = step_duration * f_smooth * stride_to_step_factor
            total_steps += steps_in_window
            
        window_records.append({
            'window_idx': i,
            'time_s': round(win_time_center, 2),
            'w_c': round(float(w_c), 2),
            'w_0': round(float(w_0), 2),
            'is_walking': bool(is_walking),
            'est_freq_hz': round(float(estimated_freq), 3) if estimated_freq is not None else None,
            'cumulative_steps': round(float(total_steps), 1)
        })
        
    return {
        'csv_file': csv_file_path,
        'total_windows': num_windows,
        'walking_windows': sum(1 for w in window_records if w['is_walking']),
        'walking_duration_s': round(walking_duration, 1),
        'total_steps': int(round(total_steps)),
        'final_cadence_spm': round((total_steps / walking_duration * 60), 1) if walking_duration > 0 else 0,
        'windows': window_records
    }


if __name__ == '__main__':
    # Test across all three dataset files
    for filepath in ['data1.csv', 'data2.csv', 'data3.csv']:
        res = detect_walking_and_count_steps(filepath)
        print(f"=== {res['csv_file']} ===")
        print(f"Total Windows:     {res['total_windows']} (1.2s each)")
        print(f"Walking Windows:   {res['walking_windows']}")
        print(f"Walking Duration:  {res['walking_duration_s']} s")
        print(f"Estimated Steps:   {res['total_steps']}")
        print(f"Walking Cadence:   {res['final_cadence_spm']} steps/min\n")
