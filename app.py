# import parser
import requests
import time
# imports ML
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import librosa
from scipy.io.wavfile import write

import soundfile as sf

#logging
def log(text):
    print(text)






# audio functions
def emphasize(signal, pre_emphasis=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal


def to_frames(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames



#parsing data
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.111 Safari/537.36'}
TEST_RADIO_URL = 'http://maslovka-home.ru:8000/thanosshow'
SAVE_LOCATION = 'data/real_time_chunk.mp3'
RECORD_LOCATION = 'data/real_time_record.mp3'


# model preparition
CHECKPOINTS_PATH = 'checkpoints/'
log('Model is ready')

model = tf.keras.models.load_model(CHECKPOINTS_PATH+'my_model_1.h5')

if __name__ == '__main__':
    start_flag = False
    while not start_flag:
        try:
            audio = requests.get(TEST_RADIO_URL, stream=True)
            audio.raise_for_status()
            start_flag = True
        except:
            time.sleep(10)

    audio = requests.get(TEST_RADIO_URL, stream=True)
    audio.raise_for_status()

    # save file
    # save_audio_file = open(RECORD_LOCATION, 'wb')

    # real time vars
    signal = np.array([])
    time_line = 0
    next_predict_ind = 0
    delited_time = 0
    preds = []
    found_flag = False
    found_ind = 0

    for chunk in audio.iter_content(chunk_size=8192):
        if chunk:  # filter out keep-alive new chunks
            audio_file = open(SAVE_LOCATION, 'wb+')
            # save_audio_file.write(chunk)

            audio_file.write(chunk)
            audio_file.close()

            try:
                my_signal,sample_rate = librosa.load(SAVE_LOCATION)
            except Exception as e:
                print(e)
                continue
            time_line += my_signal.shape[0]/sample_rate

            signal = np.concatenate([signal, my_signal])
            print(time_line)
            # prediction
            while next_predict_ind+2<time_line:
                sample = signal[int((next_predict_ind)*sample_rate):int((next_predict_ind+2)*sample_rate)]
                frames = to_frames(sample, sample_rate)
                pred = model.predict(np.array([frames]))
                pred = pred.argmax()
                preds.append(pred)
                if pred == 1:
                    found_flag = True
                    found_ind = next_predict_ind
                next_predict_ind+=1

            if found_flag and (found_ind+5<time_line):
                found_signal = signal[int((found_ind-1) * sample_rate):int((found_ind + 5) * sample_rate)]
                sf.write('data/found/' + str(found_ind) + '_sec.wav', found_signal, sample_rate, 'PCM_24')
                found_flag = False
