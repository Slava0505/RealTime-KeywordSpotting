from bs4 import BeautifulSoup
import requests
import io
from pydub import AudioSegment
# Запись в буфер

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.111 Safari/537.36'}
TEST_RADIO_URL = 'http://maslovka-home.ru:8000/soundcheck'
SAVE_LOCATION = 'data/examlpe.mp3'

# import ffmpeg

if __name__ == '__main__':
    # audio = requests.get(TEST_RADIO_URL, stream=True)
    # audio.raise_for_status()

    audio_file = io.BytesIO(open(SAVE_LOCATION,"rb").read())

    # with open(SAVE_LOCATION, 'wb') as audio_file:
    # for chunk in audio.iter_content(chunk_size=8192):
    #     if chunk:  # filter out keep-alive new chunks
    #         audio_file.write(chunk)
    recording = AudioSegment.from_file(audio_file, format="mp3")
