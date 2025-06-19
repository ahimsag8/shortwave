import numpy as np
import soundfile as sf
import gradio as gr
from scipy.signal import correlate, spectrogram
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.font_manager as fm

# 한글 폰트 설정
def setup_korean_font():
    # 시스템에서 사용 가능한 한글 폰트 찾기
    korean_fonts = []
    
    # Windows에서 일반적으로 사용되는 한글 폰트들
    windows_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Dotum', 'Batang']
    # macOS에서 일반적으로 사용되는 한글 폰트들
    mac_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic']
    # Linux에서 일반적으로 사용되는 한글 폰트들
    linux_fonts = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    
    all_fonts = windows_fonts + mac_fonts + linux_fonts
    
    for font in all_fonts:
        try:
            fm.findfont(font)
            korean_fonts.append(font)
        except:
            continue
    
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts + ['sans-serif']
    else:
        # 폴백 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
setup_korean_font()

def load_audio(file):
    audio, sr = sf.read(file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def find_best_match(long_audio, short_audio):
    correlation = correlate(long_audio, short_audio, mode='valid')
    best_offset = np.argmax(correlation)
    return best_offset

def calculate_snr(clean, test):
    noise = test - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    return snr, noise

def plot_signals(clean, matched, noise, sr):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(clean)) / sr
    axs[0].plot(t, clean)
    axs[0].set_title("1번: 테스트 오디오 (Clean Signal)")
    axs[1].plot(t, matched)
    axs[1].set_title("2번: 매칭된 구간 (Matched Segment)")
    axs[2].plot(t, noise)
    axs[2].set_title("잡음 (Matched - Clean)")

    axs[2].set_xlabel("Time [s]")
    fig.tight_layout()

    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory
    return img

# def process_audio(file1, file2):
#     short, sr1 = load_audio(file1)
#     long, sr2 = load_audio(file2)

#     if sr1 != sr2:
#         return "❌ 샘플레이트가 일치하지 않습니다.", None

#     offset = find_best_match(long, short)
#     matched = long[offset:offset + len(short)]

#     snr, noise = calculate_snr(short, matched)
#     fig_buf = plot_signals(short, matched, noise, sr1)

#     result_text = f"🎯 매칭 위치: {offset / sr1:.2f}초\n📊 SNR: {snr:.2f} dB"
#     return result_text, fig_buf

def process_audio(file1, file2):
    short, sr1 = load_audio(file1)
    long, sr2 = load_audio(file2)

    if sr1 != sr2:
        return "❌ 샘플레이트가 일치하지 않습니다.", None, None

    offset = find_best_match(long, short)
    matched = long[offset:offset + len(short)]

    snr, noise = calculate_snr(short, matched)
    fig_buf = plot_signals(short, matched, noise, sr1)

    result_text = f"🎯 매칭 위치: {offset / sr1:.2f}초\n📊 SNR: {snr:.2f} dB"
    return result_text, fig_buf, (sr1, matched)


demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type="filepath", label="1번: 테스트 오디오 (짧은 신호)"),
        gr.Audio(type="filepath", label="2번: 녹음 오디오 (긴 파일)")
    ],
    outputs=[
        gr.Text(label="결과 (SNR 및 위치)"),
        gr.Image(label="신호 비교 시각화"),
        gr.Audio(label="🔊 매칭된 오디오 구간 (듣기)")
    ],
    title="🔊 오디오 정량 비교 (SNR 평가)",
    description="테스트 오디오를 긴 오디오에서 찾아 SNR 계산 및 시각화합니다. WAV 또는 FLAC 권장."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
