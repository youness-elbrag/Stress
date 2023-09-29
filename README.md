## Part 1: Detecting Stress Levels from ECG-derived HRV Features

***This section of the project aims to develop predictive models capable of detecting stress levels using ECG-derived HRV features. By leveraging Deep learning***

## Introduction 

here we dive into Stage of Building the models which it will suitable based on Sequential data of HRV .

**Obejective**: Detec Stress or No-Stress

here the steps we wil go to handle the problem , 

- [x] Task 1: Processing The data  **The features engineering already done to select most importatnt of ECG**
    - Data Resource :
        - [x] WESAD : check the data Description [here](Data/WESAD/Data.txt)
        - [x] SWELL : check the data Description [here](Data/SWELL/Data.txt)
        
- [x] Task 2: Building The Models DL || ML 
- [x] Task 3: Quantative Results 


### 1. Per-Processing The data WESAD
 
we will go for Features Extraction the folder [ProcessUtils]() include all most important Functionalties to extracts features from a dataset, primarily for heart rate variability (HRV) analysis. following:

1. features Extraction contain two different Plans:

    ***Note*** : ***The PPG/BVP is the input signal to the algorithms that compute inter-beat-interval (IBI) timings and heart rate (HR)***

    - [x] Task-HRV Sequential Features: based on compute HRV metrcies based RR-interval using Peak-Detection on WESAD Column following **BVP is the Blood Volume Puls**

    [Resource-BVP](https://support.empatica.com/hc/en-us/articles/360029719792-E4-data-BVP-expected-signal)

    ```python
    python processingBinary.py --save_path FeatureTestTraining/binary --main_path ../Data/WESADs 
    ```

    - [x] Task Labeled Spectugarm RR-interval : Convert Signles RR-interval following it own **Label index** 
     whike sliding window: 0.25 sec (60*0.25 = 15) period  

    [Resource-RR-interval](https://www.intechopen.com/chapters/66329)

    ```python
    python rr_interval_spectral.py --save_path SpectralImages/binary/ --main_path "../Data/WESADs
    ```

### 2. HRV  ffeature Engineering WESDA from RR interval data based on PPG/BVP

Calculate various statistical and frequency domain features from a segment of PPG (Photoplethysmogram) data.

The PPG/BVP is the input signal to the algorithms that compute inter-beat-interval (IBI) timings and heart rate (HR). This function processes PPG data to extract essential features related to heart rate variability (HRV) and power spectral density (PSD). It computes a range of metrics, including heart rate statistics, RR interval characteristics, and frequency domain parameters.

- `ppg_seg` (list or array): A segment of PPG signal data.
- `window_length` (int): Length of the window in seconds for feature calculation.
- `label` (int, optional): An optional label associated with the features.

| Feature          | Description                                           |
|------------------|-------------------------------------------------------|
| `HR_mean`        | Mean heart rate.                                      |
| `HR_std`         | Standard deviation of heart rate.                    |
| `SD_mean`        | Mean RR interval.                                    |
| `SD_std`         | Standard deviation of RR interval.                  |
| `pNN50`          | Ratio of RR intervals greater than 50ms.            |
| `TINN`           | Triangular Interpolation of RR intervals.           |
| `RMSSD`          | Root Mean Square of Successive RR Interval Differences. |
| `LF`             | Low-frequency power in the frequency spectrum.      |
| `HF`             | High-frequency power in the frequency spectrum.     |
| `ULF`            | Ultra-low-frequency power in the frequency spectrum.|
| `VLF`            | Very low-frequency power in the frequency spectrum. |
| `LFHF`           | LF/HF ratio in the frequency spectrum.              |
| `Total_power`    | Total power in the frequency spectrum.               |
| `label` (optional)| An optional label associated with the features.     |

* Get RR-interval Spectrogram Stress || Non-Stress 

a PPG signal segment, detects RR intervals, and then calls the save_spectrogram function to compute Spectral Field saving spectrogram images in Struct Folders **Stress || Non-Stress**.














