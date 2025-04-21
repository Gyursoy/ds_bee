from pathlib import Path
import pandas as pd
import librosa as lb
from torchvision import transforms
import numpy as np
from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn as nn

from src.utils import config
import src.preprocessing as preproc
import src.training as train
import src.models as models
import src.feature_selection as fs
import src.feature_selection.genetic as genetic
from src.preprocessing.dataset import construct_regression_df
import src

# import src.preprocessing.feature_extraction as feature_extraction
from src.preprocessing import AudioFeatureExtractor

import os

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])

# Unarchive and convert PCM files to WAV
def convert_to_WAV():

    preproc.audio_conversion.unarchive(config['data']['files_to_unarchive'], config['data']['pcm_dir'])
    preproc.audio_conversion.pcm_to_wav_batch(config['data']['pcm_dir'], config['data']['wav_dir'], 
                                              filter_hives=True, included_hives=config['data']['hives_to_include'])
    return None

# Extract features from WAV files
def extract_features():
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

    if defect_files:
        print(f"Found {len(defect_files)} defective files")

# Check if dataset working properly
def check_dataset():

    return None

# Run experiments
def run_genetic_algorithm():
    df_model = pd.read_csv(config['data']['df_model_path'])
    dfs = preproc.dataset.get_dataframes(df_model)

    results = genetic.genetic_algorithm(dfs, config)
    res_path = config['data']['data_folder'] + "\\ga_results"
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    results.to_csv(res_path + "\\results.csv")

    return None

def test_new_features():

    test_file = config['data']['wav_dir']+'\\'+'__200616-054646-22.wav'
    output_path = config['data']['data_folder']+'\\'+'test'+'\\'+'test.jpg'

    extractor = AudioFeatureExtractor(audio_params = config['audio'])
    y, _ = lb.load(path=test_file, sr=config['audio']['sample_rate'])

    feature = extractor.power_spectral_density(y, config['data']['data_folder']+'\\'+'test'+'\\'+'test.jpg')

    print(feature.shape)

    return None

def tmp():

    extractor = AudioFeatureExtractor(audio_params = config['audio'])
    y, _ = lb.load(path='Data\WAV_data\__200716-201359-22.wav', sr=config['audio']['sample_rate'])
    psd_feat = extractor.power_spectral_density(y, config['data']['data_folder']+'\\'+'test'+'\\'+'test.jpg')

    

    print('Data returned by feature extractor:')
    print(psd_feat.shape)
    print(psd_feat)
    

    # tmp_feat = read_image('Data\psd_data\psd_data__200716-201359-22.jpg')

    # print('\n\n Data fetched by loading image:')
    # print(tmp_feat.shape)
    # print(tmp_feat)

    # TRANSFORM = transforms.Compose([
    #                 transforms.Grayscale(),
    #                 transforms.ToPILImage(),
    #                 transforms.RandomVerticalFlip(p=1),
    #                 transforms.ToTensor(),
    #             ])

    # df = pd.read_csv(config['data']['df_model_path'])

    # dataset = preproc.AudioDataset(transform=TRANSFORM)
    # print('\n\n Data fetched from AudioDataset')
    # print(dataset[0][0]['psd'].shape)
    # print(dataset[0][0]['psd'])

def test_train():
    df = pd.read_csv(config['data']['df_model_path'])
    dataframes = preproc.dataset.get_dataframes(df)

    individual = genetic.generate_individual()

    # model = models.BaseCNNModel(config['model'])
    # model = models.BaseCNNModel().to(config['device'])
    model = models.TCNModel(num_channels=[32, 64, 128]).to(config['device'])

    datasets = {TRAIN : preproc.AudioDataset(dataframes[TRAIN], transform=TRANSFORM, selected_features=individual),
                VAL : preproc.AudioDataset(dataframes[VAL], transform=TRANSFORM, selected_features=individual),
                TEST : preproc.AudioDataset(dataframes[TEST], transform=TRANSFORM, selected_features=individual)}

    dataloaders = {TRAIN: DataLoader(datasets[TRAIN], config['model']['train_batch_size'], shuffle=True),
                  VAL: DataLoader(datasets[VAL], config['model']['val_batch_size'], shuffle=False),
                  TEST: DataLoader(datasets[TEST], config['model']['test_batch_size'], shuffle=False)
                  }

    criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    optimizer = AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=config['model']['weight_decay'])

    scheduler = train.get_scheduler(
    optimizer,
    scheduler_type='one_cycle',
    max_lr=0.001,  # adjust based on your needs
    steps_per_epoch=len(dataloaders[TRAIN])
    )

    train.train_model(model, criterion, optimizer, dataloaders, scheduler=scheduler)

def tmp2():
    df = pd.read_csv(config['data']['df_model_path'])

    dataset = preproc.AudioDataset(df, transform=TRANSFORM)

    print(dataset[1][0]['stf'].shape)
    for key, value in dataset[1][0].items():
        print(key, value.shape)
    # print(dataset[1][0]['psd'])

def main():

    # print(torch.version.cuda)

    # convert_to_WAV()

    # extract_features()

    # test_new_features()
    # tmp()

    run_genetic_algorithm()

    # test_train()
    
    # tmp2()


if __name__ ==  "__main__":
    main()