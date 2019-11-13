import matplotlib.pylab as plt
import numpy as np
import torch
import argparse
import os
from hparams import create_hparams
from audio.stft import TacotronSTFT
from audio.audio_processing import griffin_lim
from train import load_model
from text import phone_to_sequence
from scipy.io.wavfile import write
from utils import to_gpu

def plot_data(data, filepath, figsize=(32, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        if data[i].ndim == 2:
            axes[i].imshow(data[i], aspect='auto', origin='bottom',
                           interpolation='none')
        elif data[i].ndim == 1:
            axes[i].scatter(range(len(data[i])), data[i],
                            color='red', marker='.', s=1, label='gate_predicted')
    plt.savefig(filepath)
    plt.close()

def synthesize(args, hparams):
    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
        
    checkpoint_path = args.checkpoint_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    
    with open(args.test_file, "r") as f:
        for line in f.readlines():
            line = line.strip()

            filepath, text, speaker = line.split("|")  
            filename = os.path.splitext(os.path.basename(filepath))[0]
            print("Synthesizing " + filename)

            sequence = np.array(phone_to_sequence(text))[None, :]
            sequence = to_gpu(torch.from_numpy(sequence)).long()
            speaker_id = np.array([int(speaker)])
            speaker_id = to_gpu(torch.from_numpy(speaker_id)).long()
            
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence, speaker_id)
            plot_data((mel_outputs.float().data.cpu().numpy()[0],
                       mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T,
                       torch.sigmoid(gate_outputs).float().data.cpu().numpy()[0]),
                       os.path.join(args.output_directory, filename+".png"))
            
            mel_spec = mel_outputs_postnet.float().data.cpu()
            denormed_mel_spec = stft.spectral_de_normalize(mel_spec)
            magnitudes = torch.matmul(torch.pinverse(stft.mel_basis), denormed_mel_spec)
            signal = griffin_lim(magnitudes, stft_fn=stft.stft_fn).numpy()[0]
            write(os.path.join(args.output_directory, filename+".wav"), hparams.sampling_rate, signal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default="syndir",
                        required=True, help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=True, help='checkpoint path')
    parser.add_argument('-t', '--test_file', type=str, default=None,
                        required=True, help='test file path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
   
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
   
    synthesize(args, hparams)
