import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip

# Global variable to hold the previous frame's magnitudes if you want
# simple smoothing. Set to None to start (i.e., no previous data yet).
# NOTE: When moving to codegen make_frame functions, all global states must be handled in its
# respective python code file.
prev_mags = None

def compute_fft(audio_segment, fft_size=1024):
    """
    Compute the FFT magnitude of a mono audio segment.
    Returns only the first half of the spectrum (real signals are symmetric).
    """
    # Optional: apply a window, e.g. Hanning, to reduce spectral leakage
    windowed = audio_segment * np.hanning(len(audio_segment))
    spectrum = np.fft.fft(windowed)
    mags = np.abs(spectrum[:len(spectrum)//2])  # keep only first half
    return mags

def smooth_magnitudes(current, previous, alpha=0.8):
    """
    Simple exponential smoothing to make amplitude changes less jumpy.
    alpha close to 1 means slower changes (more smoothing).
    alpha close to 0 means faster changes (less smoothing).
    """
    if previous is None:
        return current
    return alpha * previous + (1 - alpha) * current

def create_bar_graph_frame(magnitudes, width=800, height=400, bar_color=(0, 255, 0)):
    """
    Builds an RGB image (height x width x 3) where each column is a bar
    representing one frequency bin's magnitude.

    :param magnitudes: 1D array of FFT magnitudes
    :param width: width of the output image in pixels
    :param height: height of the output image in pixels
    :param bar_color: (R, G, B) color of the bars
    :return: A NumPy array (height, width, 3) with the bar visualization
    """

    # Number of frequency bins
    num_bins = len(magnitudes)

    # Create a black image [height, width, 3]
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # If we have more pixels than bins, compute how many pixels each bin should occupy
    # or you can scale if you want spacing. For simplicity: each bin is 1 pixel wide
    # and we skip if we exceed the width.
    # If you'd like each bin to be multiple pixels wide, adjust bin_width accordingly.
    bin_width = max(1, width // num_bins)

    # Determine the maximum magnitude so we can scale bar heights
    max_mag = np.max(magnitudes) if np.max(magnitudes) > 0 else 1e-6

    # Draw each bar
    for i, mag in enumerate(magnitudes):
        # Map the magnitude to a vertical bar size
        bar_height = int((mag / max_mag) * (height - 1))

        # Horizontal position for this bin:
        x_start = i * bin_width
        x_end = x_start + bin_width

        # Fill from the bottom of the image up to 'bar_height'
        # The bottom row is y=height-1, so we invert y when filling
        y_start = height - bar_height

        # Clip to frame width just in case
        if x_end > width:
            x_end = width

        # Color the region
        frame[y_start:height, x_start:x_end, 0] = bar_color[0]  # R
        frame[y_start:height, x_start:x_end, 1] = bar_color[1]  # G
        frame[y_start:height, x_start:x_end, 2] = bar_color[2]  # B

    return frame

def make_frame_fft(t, audio_clip, fft_size=2048, sample_rate=44100):
    global prev_mags

    # 1) Determine how many samples = fft_size. We'll center the segment on 't' if we want,
    #    but here we’ll just clip from t to t + (fft_size / sample_rate).
    segment_duration = fft_size / sample_rate
    start_time = t
    end_time = start_time + segment_duration
    if end_time > audio_clip.duration:
        end_time = audio_clip.duration

    # 2) Extract the raw audio samples as a NumPy array
    #    to_soundarray() will return shape (N, 2) if stereo, or (N,) if mono
    audio_segment = audio_clip.subclipped(start_time, end_time).to_soundarray(fps=sample_rate)

    # If stereo, convert to mono
    if audio_segment.ndim == 2 and audio_segment.shape[1] == 2:
        audio_segment = audio_segment.mean(axis=1)

    # If near end of the audio, we might get fewer samples than fft_size, so pad with zeros
    if len(audio_segment) < fft_size:
        padded = np.zeros(fft_size, dtype=audio_segment.dtype)
        padded[:len(audio_segment)] = audio_segment
        audio_segment = padded

    # 3) Compute FFT magnitudes
    mags = compute_fft(audio_segment, fft_size=fft_size)

    # 4) (Optional) Smooth with previous frame's FFT
    smoothed = smooth_magnitudes(mags, prev_mags, alpha=0.8)
    prev_mags = smoothed

    # 5) Create a clean bar-graph image directly from the magnitudes
    frame = create_bar_graph_frame(smoothed, width=800, height=400, bar_color=(0, 255, 0))

    # Return the (H, W, 3) NumPy array for MoviePy
    return frame

def generate_visualizer(input_audio, output_video, fps):
    audio_clip = AudioFileClip(input_audio)

    # A callback function that will be called for each frame when processing the VideoClip
    # This will eventually be generated code based on the user prompt.
    def video_frame(timestamp):
        return make_frame_fft(timestamp, audio_clip)
    
    video = VideoClip(video_frame, duration=audio_clip.duration)
    video.audio = CompositeAudioClip([audio_clip])
    video.write_videofile(
        output_video,
        fps=fps,
        codec="libx264",
        audio_codec="aac"
    )