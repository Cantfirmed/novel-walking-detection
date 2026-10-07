# 🏃 Novel Walking Detection & Step Counting (PWA)

Interactive web application and progressive web app (PWA) implementing the research paper:
> **A Novel Walking Detection and Step Counting Algorithm Using Unconstrained Smartphones**  
> *Xiaomin Kang, Baoqi Huang, and Guodong Qi (Sensors 2018, 18, 297)*  
> [DOI: 10.3390/s18010297](https://doi.org/10.3390/s18010297)

---

## 🌟 Key Features

1. **📱 Live Pocket Recorder:**
   - Captures real-time tri-axial gyroscope angular velocities ($\omega_x, \omega_y, \omega_z$ in rad/s) via the browser's `DeviceMotionEvent` API.
   - **Screen Wake Lock API:** Keeps the phone awake while recording.
   - **Black Screen Pocket Lock:** Turns the screen into a pitch-black OLED touch shield with hold-to-unlock, preventing accidental pocket touches.
   - Instant step count, cadence, and duration calculation when you stop walking.
   - **Export to CSV:** Download your recorded walk as a clean `.csv` file.

2. **📂 Benchmark Datasets & Custom CSV Upload:**
   - Built-in benchmark datasets: `data1.csv` (continuous walking), `data2.csv` (intermittent walking with a 10s pause), and `data3.csv` (variable intensity).
   - Drag-and-drop or upload custom CSV recordings (from apps like Phyphox or Sensor Logger).
   - Live hyperparameter adjustment (minimum amplitude threshold $\tau$, drift rejection, exponential smoothing $\alpha$, stride multipliers).
   - Detailed window-by-window inspector displaying 64-point FFT spectra, polynomial fits, and decision criteria.

3. **📖 Guide & Mathematical Reference:**
   - Complete definitions of sensor axes, coordinate frames, and paper equations (Eq. 1 through 8).

---

## 🚀 Live Demo / GitHub Pages

Deployable on GitHub Pages by selecting the `main` branch root folder `/` in **Repository Settings &rarr; Pages**.

Once deployed, visit your GitHub Pages URL on your mobile phone, add it to your home screen as a PWA, tap **Start Pocket Walk**, slip it into your pocket, and start walking!
