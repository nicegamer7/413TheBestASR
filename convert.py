import librosa
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
import torchaudio

cv_path = './Project/cv-corpus-19.0-2024-09-13/en'
df = pd.read_csv('./413TheBestASR/final_data.csv', header=0)

def plot_spectrogram(specgram):
    _, ax = plt.subplots(1, 1)
    ax.imshow(librosa.power_to_db(specgram), origin='lower', aspect='auto', interpolation='nearest')

for i, row in df.iterrows():
    clip_file = row['path']
    clip_path = f'{cv_path}/clips/{clip_file}'
    waveform, sample_rate = torchaudio.load(clip_path)

    # Pad the audio to ensure all clips are the same length. The current
    # max length is 15 seconds (see the link below). Some clips are longer than
    # the maximum. I took a look at two of these, it looks like the actual speech
    # finishes before the 15 seconds mark. The `pad` function truncates these.
    #  
    # https://github.com/common-voice/common-voice/blob/ecbaa5a2c0ab192bb3be1d8a810b24a9d726b2cf/web/src/components/pages/contribution/speak/speak.tsx#L48)
    # length_seconds = 15
    # waveform = nn.functional.pad(waveform, (0, (sample_rate * length_seconds) - waveform.shape[1]))

    # See https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=2048, hop_length=512)
    # mel_specgram is a tuple containing (channel, n_mels, time)
    mel_specgram = transform(waveform)
    plot_spectrogram(mel_specgram[0])

    # Save only the spectrogram
    plt.axis('off')
    plt.savefig(f'./Project/Spectrograms_unpadded/{clip_file[:-4]}', bbox_inches='tight', pad_inches=0)
    plt.close()