"""
activity_tracking.py

Direct Python implementation of the lecture slides pseudocode
(Activity Tracking with IMUs, slides 59-60).

Original MATLAB Pseudocode:
---------------------------
X = csvread('data.csv');            % time gyro_x gyro_y gyro_z
fs = 100;                           % sampling frequency
N = 320;                            % window size fft (320 / 100 = 3.2s)
ts = 1.25;                          % duration of the sliding window
res = fs / N;                       % resolution of the fft (0.3125 Hz)
ls = ts * fs;                       % length of the sliding window (125 samples)

while (i+N < length(X))
    [val, idx] = max(mean(abs(X(i:i+N-1, 2:4))))   % determine most sensitive axis
    S = 2*abs(fft(X(i:i+N-1, idx+1)));             % obtain spectrum
    S = S(1:length(S)/2);                          % discard mirrored part
    w0 = mean(S(1:2));                             % average amplitudes below walking
    wc = mean(S(3:7));                             % average amplitudes walking frequencies
    coefficients = polyfit([1, 2, 3, 4, 5], S(3:7)', 4); % fit polynomial
    f = @(x) -polyval(coefficients, x);            % "minus" to find maximum
    maximum = fminbnd(f, 1, 5);                    % find maximum
    
    if ((wc > w0) && (wc > 10))                    % walking condition
        fw = res * (maximum + 1);                  % calculate frequency from maximum
        c = ts * fw;                               % steps in current window
        stepCount = stepCount + c;                 % update step count
    endif
    i = i + ls;                                    % advance sliding window
endwhile
"""

import numpy as np
import pandas as pd


def track_steps_lecture(csv_filename: str):
    # Load dataset: [time, wx, wy, wz]
    df = pd.read_csv(csv_filename)
    X = df.values
    
    fs = 100.0                       # sampling frequency in Hz
    N = 320                          # window size (320 samples / 100 Hz = 3.2 seconds)
    ts = 1.25                        # step advance in seconds
    res = fs / N                     # frequency resolution = 0.3125 Hz
    ls = int(round(ts * fs))         # 125 samples per slide
    
    i = 0
    step_count = 0.0
    total_windows = 0
    walking_windows = 0
    
    # Dense grid to evaluate polynomial maximum in [1, 5] (matches fminbnd)
    grid_x = np.linspace(1.0, 5.0, 400)
    
    while i + N <= len(X):
        total_windows += 1
        
        # 1. Determine most sensitive axis: max mean(|X|) across columns wx, wy, wz (cols 1:3)
        window_gyro = X[i : i + N, 1:4]
        axis_means = np.mean(np.abs(window_gyro), axis=0)
        idx = np.argmax(axis_means)
        
        # 2. Obtain spectrum via FFT and scale by 2 (single-sided amplitude)
        signal = window_gyro[:, idx]
        fft_vals = np.fft.fft(signal)
        S = 2.0 * np.abs(fft_vals)
        S = S[: len(S) // 2]  # Discard mirrored half
        
        # 3. Average amplitudes:
        # MATLAB S(1:2) -> Python S[0:2] (0.0 Hz, 0.3125 Hz)
        w0 = np.mean(S[0:2])
        # MATLAB S(3:7) -> Python S[2:7] (0.625, 0.9375, 1.25, 1.5625, 1.875 Hz)
        wc = np.mean(S[2:7])
        
        # 4. Fit 4th-degree polynomial to points 1..5
        x_pts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pts = S[2:7]
        coefficients = np.polyfit(x_pts, y_pts, deg=4)
        
        # Find maximum in [1, 5] (matches fminbnd)
        curve = np.polyval(coefficients, grid_x)
        maximum = grid_x[np.argmax(curve)]
        
        # 5. Walking condition check
        if (wc > w0) and (wc > 10.0):
            walking_windows += 1
            fw = res * (maximum + 1.0)
            c = ts * fw
            step_count += c
            
        i += ls
        
    return {
        "file": csv_filename,
        "total_windows": total_windows,
        "walking_windows": walking_windows,
        "step_count": round(step_count, 2),
        "steps_if_stride_x2": round(step_count * 2.0, 1)
    }


if __name__ == "__main__":
    for filename in ["data1.csv", "data2.csv", "data3.csv"]:
        res = track_steps_lecture(filename)
        print(f"=== {res['file']} ===")
        print(f"Total Windows:   {res['total_windows']}")
        print(f"Walking Windows: {res['walking_windows']}")
        print(f"Pseudocode c:    {res['step_count']} (strides/cycles)")
        print(f"Individual Steps (2x): {res['steps_if_stride_x2']} footsteps\n")
