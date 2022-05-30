import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

from AudioProcessor.DeepSpeechTranscriber import Transcriber
from AudioProcessor.PyAudioFeatureExtractor import FeatureExtractor
from AudioLoader import al

# Do not normalize the sound data
sample_rate, audio = al.load_audio("../TestSound/test_sound.wav", False);
fe = FeatureExtractor(audio, sample_rate)
features = fe.extract_features()
print("Num frames: ", len(features), "Num features per frame: ", len(features[0]))

# Normalize the sound data
sample_rate, audio = al.load_audio("../TestSound/test_sound.wav", True)
tr = Transcriber("../DeepSpeech_2/models/librispeech_pretrained_v2.pth", audio)
decoded_output, decoded_offsets = tr.transcribe()
print(decoded_output, decoded_offsets)
