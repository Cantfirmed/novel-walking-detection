# 🏃 Activity Tracking with IMUs — Walking Detection & Step Counting

This project implements the unconstrained smartphone walking detection and step counting algorithm based on **Kang et al. (Sensors 2018)** and course lecture specifications (*Activity Tracking with IMUs*).

---

## 1. How It Was Done Before the Paper

Traditionally, pedometers and smartphone step counters relied on accelerometers operating in the time domain. These algorithms searched for local impact peaks, zero-crossings, or fixed acceleration thresholds caused by foot strikes. However, this approach performed poorly on unconstrained smartphones carried arbitrarily in pockets, bags, or hands because tilting the phone shifted the gravity vector across axes. Furthermore, everyday non-walking motions like typing, vehicle vibrations, leg shaking, or taking the phone out of a pocket produced false peaks, leading to inaccurate step counts.

---

## 2. What the Paper Introduces

The paper introduces a frequency-domain approach using a 3D gyroscope instead of an accelerometer to capture the clean, pendulum-like rotational motion of human limb swings. By selecting the most sensitive axis based on the largest sum of absolute angular velocities, the algorithm works regardless of how the phone is oriented. A fast Fourier transform (FFT) analyzes spectral energy in a 3.2-second window, classifying an activity as walking only when the walking band (0.6–2.0 Hz) dominates low-frequency drift and exceeds a noise threshold. Rather than detecting fragile individual peaks, steps are counted indirectly by multiplying the continuous walking duration by the fitted cadence frequency.

---

## 3. Algorithm Code Samples & Explanations

The algorithm processes 100 Hz tri-axial gyroscope data (`wx, wy, wz`) using a sliding window of $N = 320$ samples (3.2 seconds) advancing by $t_s = 1.25$ seconds ($l_s = 125$ samples).

### Step 1: Select Most Sensitive Axis
Finds the axis ($X, Y,$ or $Z$) with the highest mean absolute rotational rate, removing the need to calibrate phone orientation.

```python
# window_gyro shape: (320, 3) representing wx, wy, wz
axis_means = np.mean(np.abs(window_gyro), axis=0)
best_axis = np.argmax(axis_means)
signal = window_gyro[:, best_axis]
```

### Step 2: FFT Spectrum & Walking Condition
Computes the single-sided amplitude spectrum ($S = 2 \cdot |X|$) and compares average energy in the walking band ($w_c$: bins 3–7, $\approx 0.6\text{--}1.9\text{ Hz}$) against low-frequency drift ($w_0$: bins 1–2, $< 0.6\text{ Hz}$).

```python
# 320-point FFT with single-sided scaling (2x)
S = 2.0 * np.abs(np.fft.fft(signal))[:160]

w0 = np.mean(S[0:2])  # Bins 1-2: Drift & baseline (0.0 to 0.31 Hz)
wc = np.mean(S[2:7])  # Bins 3-7: Walking cadence band (0.62 to 1.88 Hz)

# Walking identified if walking harmonics dominate drift and noise
is_walking = (wc > w0) and (wc > 10.0)
```

### Step 3: Polynomial Fitting & Step Accumulation
Fits a 4th-order polynomial through the 5 walking bins to pinpoint the continuous peak cadence without bin-quantization error, then accumulates steps as duration $\times$ frequency.

```python
if is_walking:
    # Fit 4th-degree polynomial across walking points 1 to 5
    coeffs = np.polyfit([1, 2, 3, 4, 5], S[2:7], deg=4)
    
    # Find continuous maximum within [1, 5]
    grid_x = np.linspace(1.0, 5.0, 400)
    maximum = grid_x[np.argmax(np.polyval(coeffs, grid_x))]
    
    # Convert peak position to Hz (res = fs / N = 100 / 320 = 0.3125 Hz)
    fw = 0.3125 * (maximum + 1.0)
    
    # Accumulate steps: Duration * Cadence
    step_count += ts * fw
```

> [!NOTE]
> **Steps vs. Strides:**
> In biomechanics, a **step** is the movement from one foot to the opposite foot (left $\to$ right), while a **stride** is a full gait cycle by the same foot (left $\to$ right $\to$ left). When a phone is in a pocket, the gyroscope captures the pendulum swing of that specific leg, so $f_w$ measures **stride frequency** ($\approx 1\,\text{Hz}$). The lecture formula $c = t_s \cdot f_w$ outputs the number of **strides**; to get individual footsteps (as shown by Fitbit or Apple Health), multiply by 2 ($1\,\text{stride} = 2\,\text{steps}$).


---

## 🚀 Running the Project

* **Python Implementation:** Run `python3 activity_tracking.py` to evaluate `data1.csv`, `data2.csv`, and `data3.csv`.
* **Live Interactive Web App / PWA:** Open `index.html` in your browser or visit [https://cantfirmed.github.io/novel-walking-detection/](https://cantfirmed.github.io/novel-walking-detection/) to test live with your phone in your pocket.
