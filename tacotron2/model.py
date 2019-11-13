from math import sqrt
import torch
from torch import nn
from utils import to_gpu, get_mask_from_lengths
from .encoder import Encoder
from .decoder import Decoder
from .postnet import Postnet

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.text_embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.text_embedding.weight.data.uniform_(-val, val)

        self.speaker_embedding = nn.Embedding(
            hparams.n_speakers, hparams.speakers_embedding_dim)
        self.speaker_embedding.weight.data.uniform_(-1e-4, 1e-4)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, self.n_frames_per_step)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, speaker_ids = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.text_embedding(text_inputs).transpose(1, 2)
        embedded_speakers = self.speaker_embedding(speaker_ids) 

        encoder_outputs = self.encoder(embedded_inputs, text_lengths, embedded_speakers)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, embedded_speakers, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, speaker_ids):
        embedded_inputs = self.text_embedding(inputs).transpose(1, 2)
        embedded_speakers = self.speaker_embedding(speaker_ids) 

        encoder_outputs = self.encoder.inference(embedded_inputs, embedded_speakers)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, embedded_speakers)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
