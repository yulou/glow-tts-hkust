import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_f0(audio_path, sr=22050):
    """提取音频的基频F0"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        # 使用pyin算法提取F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),  # 约65Hz
            fmax=librosa.note_to_hz('C7'),  # 约2093Hz
            sr=sr,
            frame_length=2048,
            hop_length=256
        )
        return f0, voiced_flag, y, sr
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {e}")
        return None, None, None, None

def plot_f0_and_spectrogram(audio_path, output_path, title_suffix=""):
    """绘制F0曲线和梅尔谱图"""
    # 提取F0和音频数据
    f0, voiced_flag, y, sr = extract_f0(audio_path)
    
    if f0 is None:
        return False
    
    # 生成梅尔谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=256, win_length=1024
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 原始波形
    times_audio = np.arange(len(y)) / sr
    ax1.plot(times_audio, y, alpha=0.7, color='blue', linewidth=0.5)
    ax1.set_title(f'Original Audio Waveform {title_suffix}')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, times_audio[-1])
    ax1.grid(True, alpha=0.3)
    
    # 2. F0曲线
    times_f0 = librosa.times_like(f0, hop_length=256, sr=sr)
    ax2.plot(times_f0, f0, label='F0 contour', color='red', linewidth=2)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Fundamental Frequency (F0) {title_suffix}')
    ax2.legend()
    ax2.set_xlim(0, times_f0[-1])
    ax2.grid(True, alpha=0.3)
    
    # 3. 梅尔谱图
    img = librosa.display.specshow(
        mel_db, 
        sr=sr, 
        hop_length=256,
        x_axis='time', 
        y_axis='mel', 
        ax=ax3,
        fmin=0,
        fmax=8000
    )
    ax3.set_title(f'Mel Spectrogram {title_suffix}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mel Frequency')
    
    # 添加颜色条
    plt.colorbar(img, ax=ax3, format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")
    return True

def analyze_recommended_samples():
    """分析推荐样本"""
    # 基于LJSpeech数据集的典型样本（运行find_samples.py后更新这些路径）
    samples = {
        'high_pitch': '../data/LJSpeech-1.1/wavs/LJ001-0060.wav',
        'low_pitch': '../data/LJSpeech-1.1/wavs/LJ001-0097.wav',
        'fast_speech': '../data/LJSpeech-1.1/wavs/LJ001-0014.wav',
        'slow_speech': '../data/LJSpeech-1.1/wavs/LJ001-0008.wav'
    }
    
    # 确保输出目录存在
    os.makedirs('../results/plots', exist_ok=True)
    
    for category, filepath in samples.items():
        if os.path.exists(filepath):
            output_path = f'../results/plots/{category}_analysis.png'
            success = plot_f0_and_spectrogram(filepath, output_path, f'({category})')
            if success:
                print(f"成功分析: {category}")
        else:
            print(f"文件不存在: {filepath}")

def create_comparison_plot():
    """创建对比图"""
    # 这里可以添加对比不同样本的代码
    pass

if __name__ == "__main__":
    print("开始分析LJSpeech样本...")
    analyze_recommended_samples()
    print("分析完成！")