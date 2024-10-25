import os
import argparse
import json
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
import soundfile as sf
from rich.progress import track
from engine import Engine
from sophon import sail

device = torch.device('cpu')

def inference(kwargs):
    model = Engine(kwargs.bmodel_path, device_id=kwargs.dev_id, graph_id=0, mode=sail.IOMode.SYSIO)
    max_length = model.input_shapes[0][2]
    max_wav_length = (max_length - 1) * kwargs.hop_size + (kwargs.hop_size - 1)
    if not kwargs.center:
        max_wav_length = (max_length - 1) * kwargs.hop_size + (kwargs.hop_size - 1) + kwargs.n_fft

    test_indexes = os.listdir(kwargs.input_noisy_wavs_dir)

    os.makedirs(kwargs.output_dir, exist_ok=True)

    for index in test_indexes:
        print(index)
        noisy_wav, _ = librosa.load(os.path.join(kwargs.input_noisy_wavs_dir, index), sr=kwargs.sampling_rate)
        noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
        noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
        noisy_wav_list = [noisy_wav[:, start_indx:start_indx+max_wav_length] for start_indx in range(0, noisy_wav.shape[1], max_wav_length)]
        audio_g_list = []
        for noisy_wav in noisy_wav_list:
            try:
                noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, kwargs.n_fft, kwargs.hop_size, kwargs.win_size, kwargs.compress_factor, kwargs.center)
            except Exception as e:
                print(e)
                continue
            real_len = noisy_amp.shape[-1]
            if noisy_amp.shape[-1] < 640:
                noisy_amp = torch.cat((noisy_amp, torch.zeros((noisy_amp.shape[0], noisy_amp.shape[1], 640-noisy_amp.shape[-1]), dtype=noisy_amp.dtype)), 2)
                noisy_pha = torch.cat((noisy_pha, torch.zeros((noisy_pha.shape[0], noisy_pha.shape[1], 640-noisy_pha.shape[-1]), dtype=noisy_pha.dtype)), 2)
            amp_g, pha_g, com_g = model([noisy_amp.numpy(), noisy_pha.numpy()])
            amp_g = amp_g[:, :, :real_len]
            pha_g = pha_g[:, :, :real_len]
            audio_g = mag_pha_istft(torch.from_numpy(amp_g), torch.from_numpy(pha_g), kwargs.n_fft, kwargs.hop_size, kwargs.win_size, kwargs.compress_factor)
            audio_g = audio_g / norm_factor
            audio_g_list.append(audio_g)

        output_file = os.path.join(kwargs.output_dir, index)

        audio_g = torch.cat(audio_g_list, 1)
        sf.write(output_file, audio_g.squeeze().cpu().numpy(), kwargs.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config_mpsenet.json')
    a = parser.parse_args()

    with open(a.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    kwargs = AttrDict(json_config)

    torch.manual_seed(kwargs.seed)
    inference(kwargs)


if __name__ == '__main__':
    main()
