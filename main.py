from pathlib import Path
import pandas as pd
import librosa as lb
import numpy as np

from src.utils import config
import src.preprocessing as preproc
import src.training as train
import src.models as models
import src.feature_selection as fs
import src.feature_selection.genetic as genetic
from src.preprocessing.dataset import construct_regression_df

# import src.preprocessing.feature_extraction as feature_extraction
from src.preprocessing import AudioFeatureExtractor

import os

def convert_to_WAV():

    preproc.audio_conversion.unarchive(config['data']['files_to_unarchive'], config['data']['pcm_dir'])
    preproc.audio_conversion.pcm_to_wav_batch(config['data']['pcm_dir'], config['data']['wav_dir'], 
                                              filter_hives=True, included_hives=config['data']['hives_to_include'])
    return None


def main():

    df = pd.read_csv(Path(config['data']['df_path']))

    df_model = construct_regression_df(df)

    df_model.to_csv(config['data']['df_model_path'])
    
    wav_dir = Path(config['data']['wav_dir'])
    output_dir = Path(config['data']['data_folder'])

    # print(os.path.abspath(wav_dir))
    
    extractor = AudioFeatureExtractor(audio_params = config['audio'])
    defect_files = extractor.process_audio_files(
        input_path = wav_dir,
        output_path=output_dir,
        files=df_model['audio'].values[:],
        dir_dict=config['data']['dir_dict']
    )

    # if defect_files:
    #     print(f"Found {len(defect_files)} defective files")
        
    return

if __name__ ==  "__main__":
    main()