import os
import glob
import wave
import sys
import zipfile

def unarchive(files_to_unarchive, target_dir = 'Data/PCM_data'):
    for file in files_to_unarchive:
        try:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"An error occured while unarchiving {file}: {e}")


def pcm_to_wav(file_path: str, source_dir: str, target_dir: str, nchannels: int = 1, sampwidth=2, framerate: int = 8000, 
                nframes: int = 1, comptype: str = 'NONE', compname: str = 'NONE'):
    try:
        with open(source_dir + '/' + file_path, 'rb') as pcmfile:
            pcmdata = pcmfile.read()

        with wave.open(target_dir + '/' + file_path.split('\\')[-1][:-4]+'.wav', 'wb') as wavfile:
            wavfile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
            wavfile.writeframes(pcmdata)

    except Exception as ex:
        print(f"Error converting {file_path}: {ex}")

def pcm_to_wav_batch(source_dir, target_dir, filter_hives=False, included_hives=None):
    par_dir = os.path.abspath(os.getcwd())

    if not os.path.isdir(par_dir+source_dir):
        print(f'Missing source directory: {par_dir+source_dir}')
        sys.exit(1)

    if not os.path.isdir(par_dir+target_dir):
        os.mkdir(par_dir+target_dir)
        print(f'Missing target directory. Creating: {target_dir}')

    idx = 0
    audiofiles = glob.glob("*/*/*/*.pcm", root_dir=par_dir+source_dir)

    if filter_hives:
        filtered_audiofiles = []
        for file in audiofiles:
            if int(file.split('-')[-1][:2]) in included_hives:
                filtered_audiofiles.append(file)
    else:
        filtered_audiofiles = audiofiles

    for file in filtered_audiofiles:
        # print(par_dir+target_dir)
        # print(file.split('\\')[-1][:-4])
        pcm_to_wav(file_path=file, source_dir=par_dir+source_dir, target_dir=par_dir+target_dir)
        # pcm_to_wav(file_path=file, source_dir=source_dir, target_dir=target_dir)
        idx += 1

    print(f"Total files converted: {len(filtered_audiofiles)}")
