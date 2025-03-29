import numpy as np
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.signal import welch, resample

__all__ = ['AudioFeatureExtractor']


class AudioFeatureExtractor:

    def __init__(self, audio_params: Dict):

        self.sr = audio_params['sample_rate']
        self.hop_length = audio_params['hop_length']
        self.n_fft = audio_params['n_fft']
        self.n_mels = audio_params['n_mels']
        self.window_length = audio_params['window_length']
        self.duration = audio_params['duration']


    def create_spectrogram(self, y: np.ndarray, image_file: str) -> None:

        ms = lb.feature.melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length, 
                                     n_fft=self.n_fft, n_mels=self.n_mels)
        spectrogram = np.abs(ms)
        power_to_db = lb.power_to_db(spectrogram, ref=np.max)
        power_to_db = power_to_db[:, 0: self.window_length]

        fig = plt.figure(figsize=(1.28, 0.48), dpi=100)
        librosa.display.specshow(power_to_db, sr=self.sr, hop_length=self.hop_length)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)


    def create_stft(self, y: np.ndarray, image_file: str) -> None:

        audio_stft = lb.core.stft(y=y, hop_length=self.hop_length, n_fft=self.n_fft)
        spectrogram = np.abs(audio_stft)
        log_spectro = lb.amplitude_to_db(spectrogram)
        log_spectro = log_spectro[:, 0:self.window_length]

        fig = plt.figure(figsize=(1.28, 10.25), dpi=100)
        librosa.display.specshow(log_spectro, sr=self.sr, hop_length=self.hop_length)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)


    def create_chromogram(self, y: np.ndarray, image_file: str) -> None:

        chromo_signal = librosa.feature.chroma_stft(y=y, sr=self.sr, 
                                                  hop_length=self.hop_length, n_fft=self.n_fft)
        power_to_db = librosa.power_to_db(chromo_signal, ref=np.max)
        power_to_db = power_to_db[:, 0: self.window_length]

        fig = plt.figure(figsize=(1.28, 0.12), dpi=100)
        librosa.display.specshow(power_to_db, sr=self.sr, hop_length=self.hop_length)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)


    def create_mfcc(self, y: np.ndarray, image_file: str) -> None:

        chromo_signal = lb.feature.chroma_stft(y=y, sr=self.sr, 
                                             hop_length=self.hop_length, n_fft=self.n_fft)
        power_to_db = lb.power_to_db(chromo_signal, ref=np.max)
        mfcc = lb.feature.mfcc(S=power_to_db)
        mfcc = mfcc[:, 0: self.window_length]

        fig = plt.figure(figsize=(1.28, 0.12), dpi=100)
        lb.display.specshow(mfcc, sr=self.sr, hop_length=self.hop_length)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)


    def create_mfcc_delta(self, y: np.ndarray, image_file: str) -> None:

        chromo_signal = lb.feature.chroma_stft(y=y, sr=self.sr, 
                                             hop_length=self.hop_length, n_fft=self.n_fft)
        power_to_db = lb.power_to_db(chromo_signal, ref=np.max)
        mfcc = lb.feature.mfcc(S=power_to_db)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta = mfcc_delta[:, 0: self.window_length]

        fig = plt.figure(figsize=(1.28, 0.12), dpi=100)
        lb.display.specshow(mfcc_delta, sr=self.sr, hop_length=self.hop_length)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)

    def zero_crossing_rate(self, y: np.ndarray, image_file: str) -> None:
        frame_length = len(y) // self.window_length
        zcr = lb.feature.zero_crossing_rate(y=y, frame_length=frame_length, 
                                          hop_length=frame_length)
        # Ensure we get exactly window_length frames
        zcr = lb.util.fix_length(zcr, size=self.window_length, axis=1)
        
        #Save as image
        # fig = plt.figure(figsize=(1.28, 0.12), dpi=100)
        # lb.display.specshow(zcr, sr=self.sr, hop_length=self.hop_length)
        # plt.axis('off')
        # plt.tight_layout(pad=0)
        # plt.savefig(image_file, pad_inches=0)
        # plt.clf()
        # plt.close(fig)

        # Save numerical features in npz format
        npz_file = image_file.replace('.jpg', '.npz')
        np.savez_compressed(npz_file, zcr)

        return zcr

    def spectral_centroid(self, y: np.ndarray, image_file: str) -> None:
        # Calculate frame length to match window_length
        frame_length = len(y) // self.window_length
        
        # Calculate spectral centroid
        centroid = lb.feature.spectral_centroid(
            y=y, 
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=frame_length
        )
        
        # Ensure we get exactly window_length frames
        centroid = lb.util.fix_length(centroid, size=self.window_length, axis=1)
        
        # Scale the centroid values to a reasonable range while preserving relationships
        centroid = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + 1e-6)
        
        # Save numerical features in npz format
        npz_file = image_file.replace('.jpg', '.npz')
        np.savez_compressed(npz_file, centroid)

        return centroid

    def rms_energy(self, y: np.ndarray, image_file: str) -> None:

        frame_length = len(y) // self.window_length
        rms = lb.feature.rms(y=y, frame_length=frame_length, 
                           hop_length=frame_length)
        # Ensure we get exactly window_length frames
        rms = lb.util.fix_length(rms, size=self.window_length, axis=1)
        
        # Convert to dB scale
        rms_db = lb.amplitude_to_db(rms, ref=np.max)
        
        # Save as image
        # fig = plt.figure(figsize=(1.28, 0.12), dpi=100)
        # lb.display.specshow(rms_db, sr=self.sr, hop_length=self.hop_length)
        # plt.axis('off')
        # plt.tight_layout(pad=0)
        # plt.savefig(image_file, pad_inches=0)
        # plt.clf()
        # plt.close(fig)

        # Save numerical features in npz format
        npz_file = image_file.replace('.jpg', '.npz')
        np.savez_compressed(npz_file, rms_db)

        return rms_db
    

    def lpc_features(self, y: np.ndarray, image_file: str) -> None:
        # Calculate frame length to match window_length
        frame_length = len(y) // self.window_length
        
        # Initialize array to store LPC coefficients
        lpc_features = np.zeros((16, self.window_length))
        
        # Process audio in frames
        for i in range(self.window_length):
            start = i * frame_length
            end = start + frame_length
            if end <= len(y):
                frame = y[start:end]
                # Calculate LPC coefficients (order=8)
                lpc_coeffs = lb.lpc(frame, order=16)
                # Store coefficients (excluding first coefficient)
                lpc_features[:, i] = lpc_coeffs[1:]
        
        # Normalize features
        lpc_features = lb.util.normalize(lpc_features)
        
        # Save numerical features in npz format
        npz_file = image_file.replace('.jpg', '.npz')
        np.savez_compressed(npz_file, lpc_features)
        
        return lpc_features
    
    def power_spectral_density(self, y: np.ndarray, image_file: str) -> None:
        # Calculate frame length to match desired 128 time points 
        frame_length = len(y) // 128

        # Initialize array to store PSD values
        psd_features = np.zeros((256, 128))
        
        # Process audio in frames
        for i in range(128):
            start = i * frame_length
            end = start + frame_length
            if end <= len(y):
                frame = y[start:end]
                frequencies, psd = welch(frame, 
                                      fs=self.sr,
                                      nperseg=min(256, len(frame)),
                                      noverlap=min(128, len(frame)//2),
                                      nfft=512)
                psd_features[:, i] = psd[:256]

        # Avoid log of zero
        epsilon = 1e-10
        psd_features = np.maximum(psd_features, epsilon)
        
        # Convert to dB scale with fixed reference
        psd_features = lb.power_to_db(psd_features, ref=1.0)
        
        # Normalize to [0,1] range
        psd_features = (psd_features - np.min(psd_features)) / (np.max(psd_features) - np.min(psd_features) + epsilon)

        # Save as image
        fig = plt.figure(figsize=(1.28, 2.56), dpi=100)
        lb.display.specshow(psd_features, sr=self.sr, hop_length=self.hop_length)
        
        # Scale values to 0-255 range for better image representation
        # psd_features = (255 * psd_features).astype(np.uint8)
        
        # Create figure with exact pixel dimensions
        # Using 128x256 to match your data dimensions
        # fig = plt.figure(figsize=(1.28, 2.56), dpi=100)  # This will give 128x256 pixels
        # plt.imshow(psd_features, cmap='gray', aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(image_file, pad_inches=0)
        plt.clf()
        plt.close(fig)
        
        return psd_features

    def cnn_features(self, y: np.ndarray, image_file: str) -> None:
        # Implement a CNN model to extract features from the audio file
        return None


    def process_audio_files(self, input_path: str | Path, output_path: str | Path, 
                          files: List[str], dir_dict: Dict[str, str]) -> List[str]:
        
        input_path = Path(input_path)
        # output_path = Path(output_path)
                    
        defect_files = []

        for i, file in tqdm(enumerate(files[:]), total=len(files)):
            try:
                input_file = input_path / file
                input_file = Path(os.path.abspath(input_file))

                    
                y, _ = lb.load(str(input_file), sr=self.sr)  # librosa needs string path
                
                if len(y) < self.duration * self.sr - 1 or len(y) > self.duration * self.sr * 10:
                    del y
                    gc.collect()
                    defect_files.append(file)
                    continue
                
                feature_types = {}
                for key in dir_dict.keys():
                    os.makedirs(os.path.abspath(os.path.join(output_path, dir_dict[key])), exist_ok=True)
                    feature_types[key] = (self.map_dir_to_features(key), dir_dict[key])

                for key, (func, prefix) in feature_types.items():
                    # output_file = output_path / dir_dict[key] / f"{prefix}{file.replace('.wav', '.jpg')}"
                    output_file = os.path.abspath(os.path.join(output_path, dir_dict[key], f"{prefix}{file.replace('.wav', '.jpg')}"))
                    func(y, output_file)

                # print(output_file)
                # break
                
                if i % 100 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                
            
        return defect_files
    

    def map_dir_to_features(self, key):
        dir_to_func = { "mel": self.create_spectrogram,
                        "stf": self.create_stft,
                        "chr": self.create_chromogram,
                        "mfc": self.create_mfcc,
                        "dmf": self.create_mfcc_delta,
                        "zcr" : self.zero_crossing_rate,
                        "rms" : self.rms_energy,
                        "lpc" : self.lpc_features,
                        "sce" : self.spectral_centroid,
                        "psd" : self.power_spectral_density
                        }
        
        return dir_to_func[key]