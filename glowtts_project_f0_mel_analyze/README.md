# GlowTTS Project: F0 & Mel Spectrogram Analysis and Visualization

## 📋 Overview

This task involves conducting Fundamental Frequency (F0) analysis and Mel spectrogram visualization within a Text-to-Speech (TTS) system, based on the LJSpeech dataset. The primary objectives include understanding prosodic features in speech synthesis and validating relevant acoustic theories through practical data analysis.

## 👤 Responsible Person

- **Name**: Zhu Yixun
- **Last Updated**: October 19th

## 🎯 Core Tasks

### 1. Theoretical Foundation Research

- Study Chapter 9 "Prosodic Synthesis" in *Text-to-Speech Synthesis.pdf*
- Understand the concepts of Fundamental Frequency (F0) and Pitch (Section 9.1.1)
- Analyze intonation patterns and their variations (Sections 9.1.2 & 9.2)
- Master concepts including Pitch Accent, Core Accent, Continuation Rise, and Boundary Tone

### 2. Data Visualization Implementation

- Use matplotlib to plot F0 contours from the LJSpeech dataset
- Generate Mel spectrogram visualizations
- Conduct comparative analysis of audio samples with different pitches and speech rates
- Provide F0 contour comparisons corresponding to high and low pitch audio
- Display differences in Mel spectrograms for different speech rates

## 🛠️ Technical Implementation

### Code Structure

```
scripts/
├── find_samples.py          # Sample feature analysis and candidate selection
├── f0_analysis.py           # Main analysis script: F0 extraction & visualization
```



### Main Functional Modules

#### 1. Sample Discovery (`find_samples.py`)

```
def analyze_audio_characteristics(audio_path)
def find_sample_candidates()
```

- **Function**: Automatically analyzes audio features in the LJSpeech dataset
- **Output**: Recommends samples with significant feature differences (high/low pitch, fast/slow speech rate)

#### 2. F0 Analysis & Visualization (`f0_analysis.py`)

```
def extract_f0(audio_path)
def plot_f0_and_spectrogram(audio_path, output_path, title_suffix)
```

- **Function**: Extracts fundamental frequency (F0) and generates comprehensive visualizations
- **Output**: PNG files containing waveform, F0 contour, and Mel spectrogram

### Output Content

Each analyzed sample generates a comprehensive visualization containing three subplots:

1. **Original Waveform Plot**
   - Displays the amplitude variation of the audio over time
   - Used to observe the overall energy distribution of the speech
2. **F0 Fundamental Frequency Contour**
   - Shows the trajectory of the fundamental frequency over time
   - Red curve represents the extracted F0 values
   - Used to analyze pitch variations and intonation patterns
3. **Mel Spectrogram**
   - Displays the spectral energy distribution of the audio on the Mel frequency scale
   - Color intensity represents energy strength (dB)
   - Used to observe formants and speech rate characteristics

## 📊 Analysis Results

### Comparative Analysis Content

- **High Pitch vs. Low Pitch**: Differences in F0 contour position, frequency range comparison
- **Fast Speech vs. Slow Speech**: Mel spectrogram timeline density, waveform compactness

### For **High Pitch vs. Low Pitch** samples:

- **F0 Contour**: The F0 contour of high-pitch samples is generally positioned higher
- **Numerical Range**: In the generated examples, high pitch concentrates around 100-350Hz, peaking around 600Hz; low pitch concentrates around 75-250Hz, peaking around 350Hz.

### For **Fast Speech vs. Slow Speech** samples:

- **Mel Spectrogram**: The timeline of fast speech is more compact, while slow speech is more expanded
- **Waveform**: The waveform of fast speech is more dense

### Generated Files

```
results/plots/
├── high_pitch_analysis.png
├── low_pitch_analysis.png
├── fast_speech_analysis.png
└── slow_speech_analysis.png
```

## 🚀 Usage Instructions

1. **Environment Setup**:

   ```
   pip install librosa matplotlib numpy scipy pandas
   ```

2. **Data Preparation**:

   - Download the LJSpeech dataset to `data/LJSpeech-1.1/`

3. **Run Analysis**:

   ```
   cd scripts
   python find_samples.py    # Find characteristic samples
   python f0_analysis.py     # Generate visualization charts
   ```