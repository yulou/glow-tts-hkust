import librosa
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_f0(audio_path, sr=22050):
    """Extract fundamental frequency (F0) from audio"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        # Extract F0 using pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=2048,
            hop_length=256,
            fill_na=np.nan
        )
        times = librosa.times_like(f0, hop_length=256, sr=sr)
        return times, f0
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None, None

def plot_discontinuous_f0(times, f0, color, label):
    """Plot discontinuous F0 trajectory with downward extension lines at both ends"""
    # Find NaN value boundaries to separate continuous segments
    nan_positions = np.isnan(f0)
    
    # Find segment starts and ends
    segment_starts = []
    segment_ends = []
    
    if len(f0) > 0:
        if not nan_positions[0]:  # If first point is valid
            segment_starts.append(0)
        
        for i in range(1, len(f0)):
            if nan_positions[i-1] and not nan_positions[i]:  # NaN -> valid
                segment_starts.append(i)
            elif not nan_positions[i-1] and nan_positions[i]:  # valid -> NaN
                segment_ends.append(i-1)
        
        if not nan_positions[-1]:  # If last point is valid
            segment_ends.append(len(f0)-1)
    
    # Plot each continuous segment
    for start, end in zip(segment_starts, segment_ends):
        if end - start >= 1:  # Plot only if at least 2 points
            segment_times = times[start:end+1]
            segment_f0 = f0[start:end+1]
            
            # Plot main F0 trajectory
            plt.plot(segment_times, segment_f0, color=color, linewidth=1.5)
            
            # Add downward extension lines at both ends
            # Calculate extension length (0.3% of time axis)
            time_range = times[-1] - times[0]
            extension_length = time_range * 0.003
            
            # Start extension line - extend to bottom left
            if start > 0:  # If not the first frame
                start_time_before = max(0, segment_times[0] - extension_length)
            else:
                start_time_before = segment_times[0] - extension_length
                
            plt.plot([start_time_before, segment_times[0]], 
                    [0, segment_f0[0]], 
                    color=color, linewidth=1.5, alpha=0.7)
            
            # End extension line - extend to bottom right
            if end < len(times) - 1:  # If not the last frame
                end_time_after = min(times[-1], segment_times[-1] + extension_length)
            else:
                end_time_after = segment_times[-1] + extension_length
                
            plt.plot([segment_times[-1], end_time_after], 
                    [segment_f0[-1], 0], 
                    color=color, linewidth=1.5, alpha=0.7)

def plot_pitch_tracks_comparison(audio_files, speaker_names):
    """
    Plot pitch tracks comparison for multiple speakers
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Color settings
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Extract and plot F0 curves for each speaker
    for i, (audio_file, speaker_name) in enumerate(zip(audio_files, speaker_names)):
        print(f"Processing {speaker_name} audio...")
        
        times, f0 = extract_f0(audio_file)
        
        if f0 is not None and times is not None:
            # Plot discontinuous F0 trajectory
            plot_discontinuous_f0(times, f0, colors[i], speaker_name)
            
            # Add invisible continuous line for legend
            plt.plot([], [], color=colors[i], label=speaker_name, linewidth=1.5)
    
    # Set figure properties
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Frequency [Hz]', fontsize=12)
    plt.title('Pitch tracks of speech samples from different speaker identities', 
              fontsize=13, pad=20)
    plt.legend(fontsize=10, frameon=True, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.ylim(0, 350)
    plt.xlim(0, None)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('pitch_tracks_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Audio file list
    audio_files = [
        "./audio_data/arctic_a0407_slt.wav",  # Female 1
        "./audio_data/arctic_a0407_clb.wav",  # Female 2  
        "./audio_data/arctic_a0407_bdl.wav",   # Male 1
        "./audio_data/arctic_a0407_rms.wav"    # Male 2
    ]
    
    speaker_names = ["Female 1", "Female 2", "Male 1", "Male 2"]
    
    # Generate pitch tracks comparison plot
    plot_pitch_tracks_comparison(audio_files, speaker_names)
    print("Pitch tracks comparison plot generated: pitch_tracks_comparison.png")