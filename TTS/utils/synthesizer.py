import time
from typing import List

import numpy as np
import pysbd
import torch

from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.speakers import SpeakerManager

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.tts.utils.synthesis import synthesis, trim_silence
from TTS.tts.utils.text import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input, setup_generator


class Synthesizer(object):
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        tts_speakers_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.speaker_manager = None
        self.num_speakers = 0
        self.tts_speakers = {}
        self.speaker_embedding_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_speakers(self, speaker_file: str) -> None:
        """Load the SpeakerManager to organize multi-speaker TTS. It loads the speakers meta-data and the speaker
        encoder if it is defined.

        Args:
            speaker_file (str): path to the speakers meta-data file.
        """
        print("Loading speakers ...")
        self.speaker_manager = SpeakerManager(encoder_model_path=self.encoder_checkpoint, encoder_config_path=self.encoder_config)
        self.speaker_manager.load_x_vectors_file(self.tts_config.get("external_speaker_embedding_file", speaker_file))
        self.num_speakers = self.speaker_manager.num_speakers
        self.speaker_embedding_dim = self.speaker_manager.x_vector_dim

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement

        global symbols, phonemes

        self.tts_config = load_config(tts_config_path)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(verbose=False, **self.tts_config.audio)

        if "characters" in self.tts_config.keys():
            symbols, phonemes = make_symbols(**self.tts_config.characters)

        if self.use_phonemes:
            self.input_size = len(phonemes)
        else:
            self.input_size = len(symbols)

        if self.tts_config.use_speaker_embedding is True:
            self._load_speakers(self.tts_speakers_file)

        self.tts_model = setup_model(
            self.input_size,
            num_speakers=self.num_speakers,
            c=self.tts_config,
            speaker_embedding_dim=self.speaker_embedding_dim,
        )
        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.

        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config["audio"])
        self.vocoder_model = setup_generator(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def _split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        self.ap.save_wav(wav, path, self.output_sample_rate)

    def tts(self, text: str, speaker_idx: str = "", speaker_wav=None, style_wav=None) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_idx (str, optional): spekaer id for multi-speaker models. Defaults to "".
            speaker_wav ():
            style_wav ([type], optional): style waveform for GST. Defaults to None.

        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []
        sens = self._split_into_sentences(text)
        print(" > Text splitted to sentences.")
        print(sens)

        # get the speaker embedding from the saved x_vectors.
        if speaker_idx and isinstance(speaker_idx, str):
            speaker_embedding = self.speaker_manager.get_x_vectors_by_speaker(speaker_idx)[0]
        else:
            speaker_embedding = None

        # compute a new x_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = self.speaker_manager.compute_x_vector_from_clip(speaker_wav)

        use_gl = self.vocoder_model is None

        for sen in sens:
            # synthesize voice
            waveform, _, _, mel_postnet_spec, _, _ = synthesis(
                model=self.tts_model,
                text=sen,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                ap=self.ap,
                speaker_id=None,
                style_wav=style_wav,
                truncated=False,
                enable_eos_bos_chars=self.tts_config.enable_eos_bos_chars,
                use_griffin_lim=use_gl,
                speaker_embedding=speaker_embedding,
            )
            if not use_gl:
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda and not use_gl:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            waveform = trim_silence(waveform, self.ap)

            wavs += list(waveform)
            wavs += [0] * 10000

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs
