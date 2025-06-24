import numpy as np
import soundfile as sf
import gradio as gr
from scipy.signal import correlate#, spectrogram
import matplotlib.pyplot as plt
import io
from PIL import Image

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

# 참고: plt는 한글 사용 시 폰트 문제 발생
def plot_signals(clean, matched, noise, sr):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(clean)) / sr
    axs[0].plot(t, clean)
    axs[0].set_title('Test Audio (Received Signal)')
    axs[1].plot(t, matched)
    axs[1].set_title('Matched Segment from the Original Audio')
    axs[2].plot(t, noise)
    axs[2].set_title('Noise (Test - Matched)')

    axs[2].set_xlabel("Time [s]")
    fig.tight_layout()

    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory
    return img

def process_audio(file1, file2):
    short, sr1 = load_audio(file1)
    long, sr2 = load_audio(file2)

    if sr1 != sr2:
        return "❌ 샘플레이트가 일치하지 않습니다.", None, None

    offset = find_best_match(long, short)
    matched = long[offset:offset + len(short)]

    snr, noise = calculate_snr(short, matched)
    fig_buf = plot_signals(short, matched, noise, sr1)

    result_text = f"🎯 매칭 위치: {offset / sr1:.2f}초\n\n## 📊 **SNR: {snr:.2f} dB**"
    gr.Info(f'신호대잡음비(SNR)는 {snr:.2f} dB 입니다.', duration=10)
    return result_text, fig_buf, (sr1, matched)

# Blocks를 사용해서 이미지와 Interface를 결합
with gr.Blocks() as demo:
    gr.Image("KBSMRI_crop.jpg", show_label=False, show_download_button=False, container=False)
    gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath", label="1번: 테스트 오디오 (수신 신호, 짧음)"),
            gr.Audio(type="filepath", label="2번: 원본 오디오 (송신 신호, 긺)")
        ],
        outputs=[
            gr.Text(label="결과 (SNR 및 위치)"),
            gr.Image(label="신호 비교 시각화"),
            gr.Audio(label="🔊 매칭된 원 오디오 구간 (듣기)")
        ],
        description=
        '''국제방송 수신 품질 평가를 위한 테스트 페이지입니다.<br>
        짧은 테스트(수신) 신호를 긴 원본(송신) 신호에서 찾아 매칭하고 SNR을 계산합니다.<br>
        오디오 신호는 WAV 또는 FLAC 권장하며 샘플링주파수는 동일해야 합니다.<br><br>
        담당자:<br>
        곽천섭 수석 hosu10@gmail.com<br>
        오주현 팀장 jhoh@kbs.co.kr'''
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7275, share=False)