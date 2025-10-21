import pandas as pd
import librosa
import numpy as np
import os

def analyze_audio_characteristics(audio_path):
    """分析音频特征以便分类"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # 提取F0特征
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256
        )
        f0_clean = f0[voiced_flag] if voiced_flag.any() else np.array([])
        
        if len(f0_clean) > 0:
            avg_f0 = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
        else:
            avg_f0 = 0
            f0_std = 0
        
        # 计算语速（基于音频持续时间）
        speech_rate = duration  # 可以用文本长度进一步优化
        
        return {
            'file_path': audio_path,
            'duration': duration,
            'avg_f0': avg_f0,
            'f0_std': f0_std,
            'speech_rate': speech_rate
        }
    except Exception as e:
        print(f"分析 {audio_path} 时出错: {e}")
        return None

def find_sample_candidates():
    """寻找适合的样本"""
    metadata_path = "../data/LJSpeech-1.1/metadata.csv"
    wavs_dir = "../data/LJSpeech-1.1/wavs/"
    
    # 读取元数据
    metadata = pd.read_csv(metadata_path, sep='|', header=None, 
                          names=['file_name', 'transcription', 'normalized_transcription'])
    
    results = []
    
    # 分析前100个样本（避免处理全部数据）
    for i, row in metadata.head(100).iterrows():
        audio_path = os.path.join(wavs_dir, f"{row['file_name']}.wav")
        if os.path.exists(audio_path):
            analysis = analyze_audio_characteristics(audio_path)
            if analysis:
                results.append(analysis)
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(results)
    
    # 按特征排序寻找样本
    high_pitch = df.nlargest(5, 'avg_f0')
    low_pitch = df.nsmallest(5, 'avg_f0')
    long_duration = df.nlargest(5, 'duration')  # 假设为慢速
    short_duration = df.nsmallest(5, 'duration')  # 假设为快速
    
    print("=== 高音调候选样本 ===")
    print(high_pitch[['file_path', 'avg_f0', 'duration']])
    
    print("\n=== 低音调候选样本 ===")
    print(low_pitch[['file_path', 'avg_f0', 'duration']])
    
    print("\n=== 长持续时间（慢速）候选样本 ===")
    print(long_duration[['file_path', 'duration', 'avg_f0']])
    
    print("\n=== 短持续时间（快速）候选样本 ===")
    print(short_duration[['file_path', 'duration', 'avg_f0']])
    
    return {
        'high_pitch': high_pitch['file_path'].iloc[0],
        'low_pitch': low_pitch['file_path'].iloc[0],
        'slow_speech': long_duration['file_path'].iloc[0],
        'fast_speech': short_duration['file_path'].iloc[0]
    }

if __name__ == "__main__":
    samples = find_sample_candidates()
    print("\n=== 推荐样本 ===")
    for category, path in samples.items():
        print(f"{category}: {path}")