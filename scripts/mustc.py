# coding=utf-8

import csv
import os
import yaml
from itertools import groupby
from pathlib import Path

import torchaudio

import datasets


_VERSION = "2.0.0"

_CITATION = """
@article{CATTONI2021101155,
    title = {MuST-C: A multilingual corpus for end-to-end speech translation},
    author = {Roldano Cattoni and Mattia Antonino {Di Gangi} and Luisa Bentivogli and Matteo Negri and Marco Turchi},
    journal = {Computer Speech & Language},
    volume = {66},
    pages = {101155},
    year = {2021},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2020.101155},
    url = {https://www.sciencedirect.com/science/article/pii/S0885230820300887},
}
"""

_DESCRIPTION = """
MuST-C is a multilingual speech translation corpus whose size and quality facilitates
the training of end-to-end systems for speech translation from English into several languages.
For each target language, MuST-C comprises several hundred hours of audio recordings
from English [TED Talks](https://www.ted.com/talks), which are automatically aligned
at the sentence level with their manual transcriptions and translations.
"""

_HOMEPAGE = "https://ict.fbk.eu/must-c/"

_LANGUAGES = ["de", "ja", "zh"]

_SAMPLE_RATE = 16_000


class MUSTC(datasets.GeneratorBasedBuilder):
    """MUSTC Dataset."""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"en-{lang}", version=datasets.Version(_VERSION)) for lang in _LANGUAGES
    ]

    @property
    def manual_download_instructions(self):
        return f"""Please download the MUST-C v3 from https://ict.fbk.eu/must-c/
        and unpack it with `tar xvzf MUSTC_v3.0_{self.config.name}.tar.gz`.
        Make sure to pass the path to the directory in which you unpacked the downloaded
        file as `data_dir`: `datasets.load_dataset('mustc', data_dir="path/to/dir")`
        """

    # MUSTC_ROOT  # <- point here in --data_dir in arg
    # └── en-de
    #     └── data
    #         ├── dev
    #         │   ├── txt
    #         │   │   ├── dev.de
    #         │   │   ├── dev.en
    #         │   │   └── dev.yaml
    #         │   └── wav
    #         │       ├── ted_767.wav
    #         │       ├── [...]
    #         │       └── ted_837.wav
    #         ├── train
    #         │   ├── txt/
    #         │   └── wav/
    #         ├── tst-COMMON
    #         │   ├── txt/
    #         │   └── wav/
    #         └── tst-HE
    #             ├── txt/
    #             └── wav/

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                client_id=datasets.Value("string"),
                file=datasets.Value("string"),
                audio=datasets.Audio(sampling_rate=_SAMPLE_RATE),
                sentence=datasets.Value("string"),
                translation=datasets.Value("string"),
                id=datasets.Value("string"),
            ),
            supervised_keys=("file", "translation"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        source_lang, target_lang = self.config.name.split("-")
        assert source_lang == "en"
        assert target_lang in _LANGUAGES

        data_root = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        root_path = Path(data_root) / self.config.name

        if not os.path.exists(root_path):
            raise FileNotFoundError(
                "Dataset not found. Manual download required. "
                f"{self.manual_download_instructions}"
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"root_path": root_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"root_path": root_path, "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split("tst.COMMON"),
                gen_kwargs={"root_path": root_path, "split": "tst-COMMON"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split("tst.HE"),
                gen_kwargs={"root_path": root_path, "split": "tst-HE"},
            ),
        ]

    def _generate_examples(self, root_path, split):
        source_lang, target_lang = self.config.name.split("-")

        # Load audio segments
        txt_root = Path(root_path) / "data" / split / "txt"
        with (txt_root / f"{split}.yaml").open("r") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)

        # Load source and target utterances
        with open(txt_root / f"{split}.{source_lang}", "r") as s_f:
            with open(txt_root / f"{split}.{target_lang}", "r") as t_f:
                s_lines = s_f.readlines()
                t_lines = t_f.readlines()
                assert len(s_lines) == len(t_lines) == len(segments)
                for i, (src, trg) in enumerate(zip(s_lines, t_lines)):
                    segments[i][source_lang] = src.rstrip()
                    segments[i][target_lang] = trg.rstrip()

        # Load waveforms
        _id = 0
        wav_root = Path(root_path) / "data" / split / "wav"
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * int(_SAMPLE_RATE))
                duration = int(float(segment["duration"]) * int(_SAMPLE_RATE))
                waveform, sr = torchaudio.load(wav_path,
                                               frame_offset=offset,
                                               num_frames=duration)
                assert duration == waveform.size(1), (duration, waveform.size(1))
                assert sr == int(_SAMPLE_RATE), (sr, int(_SAMPLE_RATE))

                yield _id, {
                    "file": wav_path.as_posix(),
                    "audio": {
                        "array": waveform.squeeze().numpy(),
                        "path": wav_path.as_posix(),
                        "sampling_rate": sr,
                    },
                    "sentence": segment[source_lang],
                    "translation": segment[target_lang],
                    "client_id": segment["speaker_id"],
                    "id": f"{wav_path.stem}_{i}",
                }
                _id += 1
