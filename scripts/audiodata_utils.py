#!/usr/bin/env python3
# coding: utf-8

# Adapted from
# https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/data_utils.py

import csv
import io
import itertools
import re
import string
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import sentencepiece as sp
from tqdm import tqdm

from joeynmt.helpers_for_audio import _is_npy_data

# default definitions
SPECIAL_SYMBOLS = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "sep_token": None,  # "<sep>",
    "unk_id": 0,
    "pad_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "sep_id": None,  # 4,
    "lang_tags": [],  # ["<de>", "<en>"],
}
N_WORKERS = 16  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 5000  # joint vocab
LOWERCASE = False
CHARACTER_COVERAGE = 1.0


def get_zip_manifest(zip_path: Path, npy_root: Optional[Path] = None):
    manifest = {}
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    # retrieve offsets
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        with zip_path.open("rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and _is_npy_data(data), (utt_id, len(data))
        manifest[utt_id] = f"{zip_path.name}:{offset}:{file_size}"
        # sanity check
        if npy_root is not None:
            byte_data = np.load(io.BytesIO(data))
            npy_data = np.load((npy_root / f"{utt_id}.npy").as_posix())
            assert np.allclose(byte_data, npy_data)
    return manifest


def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            try:
                f.write(path, arcname=path.name)
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(f"{path}") from e


def save_tsv(df: pd.DataFrame, path: Path, header: bool = True) -> None:
    df.to_csv(
        path.as_posix(),
        sep="\t",
        header=header,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE
    )


def load_tsv(path: Path):
    return pd.read_csv(
        path.as_posix(),
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False
    )


def build_sp_model(
    input_path: Path,
    model_path_prefix: Path,
    cfg: SimpleNamespace,
    **kwargs,
) -> None:
    """
    Build sentencepiece model
    """
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={model_path_prefix.as_posix()}",
        f"--model_type={kwargs.get('model_type', SP_MODEL_TYPE)}",
        f"--vocab_size={kwargs.get('vocab_size', VOCAB_SIZE)}",
        f"--character_coverage={kwargs.get('character_coverage', CHARACTER_COVERAGE)}",
        f"--num_threads={kwargs.get('num_workers', N_WORKERS)}",
        f"--unk_piece={cfg.unk_token}",
        f"--bos_piece={cfg.bos_token}",
        f"--eos_piece={cfg.eos_token}",
        f"--pad_piece={cfg.pad_token}",
        f"--unk_id={cfg.unk_id}",
        f"--bos_id={cfg.bos_id}",
        f"--eos_id={cfg.eos_id}",
        f"--pad_id={cfg.pad_id}",
        "--vocabulary_output_piece_score=false",
    ]
    if cfg.sep_token:
        arguments.append(f"--control_symbols={cfg.sep_token}")

    user_defined_symbols = cfg.lang_tags + kwargs.get("user_defined_symbols", [])
    if user_defined_symbols:
        arguments.append(f"--user_defined_symbols={','.join(user_defined_symbols)}")

    accept_language = kwargs.get("langs", [])
    if accept_language:
        arguments.append(f"--accept_language={','.join(accept_language)}")

    sp.SentencePieceTrainer.Train(" ".join(arguments))


class Normalizer:
    MAPPING = {
        'en': {'%': 'percent', '&': 'and', '=': 'equal to', '@': 'at'},
        'de': {'€': 'Euro'}, 'ja': {}
    }
    ESCAPE = {
        'en': [
            ('(noise)', '<noise>'),
            ('[unclear]', '<unclear>'),
            ('(applause)', '<applause>'),
            ('(laughter)', '<laughter>'),
            ('(laughing)', '<laughter>'),
            ('(laughs)', '<laughter>'),
        ],
        'de': [
            ('(Geräusch)', '<Geräusch>'),
            ('[unklar]', '<unklar>'),
            ('(Lachen)', '<Lachen>'),
            ('(Lacht)', '<Lachen>'),
            ('(lacht)', '<Lachen>'),
            ('(Gelächter)', '<Lachen>'),
            ('(Gelaechter)', '<Lachen>'),
            ('(Applaus)', '<Applaus>'),
            ('(Applause)', '<Applaus>'),
            ('(Beifall)', '<Applaus>'),
        ],
        'ja': [
            ('（ため息）', '<ため息>'),
            ('(笑)', '<笑>'),
            ('（笑）', '<笑>'),
            ('（笑い）', '<笑>'),
            ('（笑い声）', '<笑>'),
            ('（歌う）', '<音楽>'),
            ('（音楽）', '<音楽>'),
            ('（ヒーローの音楽）', '<音楽>'),
            ('（大音量の音楽）', '<音楽>'),
            ('(ビデオ)', '<音楽>'),
            ('（ビデオ）', '<音楽>'),
            ('（映像と音楽）', '<音楽>'),
            ('(映像)', '<音楽>'),
            ('（映像）', '<音楽>'),
            ('(拍手)', '<拍手>'),
            ('（拍手）', '<拍手>'),
            ('（録音済みの拍手）', '<拍手>'),
        ],
    }

    def __init__(
        self,
        lang: str = "en",
        lowercase: bool = True,
        remove_punc: bool = False,
        normalize_num: bool = True,
        mapping_path: Path = None,
        escape: bool = True
    ):
        try:
            # pylint: disable=import-outside-toplevel,unused-import
            from normalize_japanese import normalize as normalize_ja  # noqa: F401
            from sacremoses.normalize import MosesPunctNormalizer
        except ImportError as e:
            raise ImportError from e

        self.moses = MosesPunctNormalizer(lang)
        self.lowercase = lowercase
        self.remove_punc = remove_punc
        self.normalize_num = normalize_num
        self.lang = lang

        if normalize_num:
            try:
                import inflect  # pylint: disable=import-outside-toplevel
                self.inflect = inflect.engine()
            except ImportError as e:
                raise ImportError from e

        self.escape = self.ESCAPE[lang] if escape else None
        self.mapping = self.MAPPING[lang]
        if mapping_path:
            self.mapping_num = {}
            with Path(mapping_path).open('r', encoding="utf8") as f:
                for line in f.readlines():
                    l = line.strip().split('\t')  # noqa: E741
                    self.mapping_num[l[0]] = l[1]
        # mapping.txt (one word per line)
        # ---------- format:
        # orig_surface [TAB] replacement
        # ---------- examples:
        # g7	g seven
        # 11pm	eleven pm
        # 6am	six am
        # ----------

    def _years(self, word):
        num_word = word
        s_flag = False
        if num_word.endswith("'s"):
            s_flag = True
            num_word = num_word[:-2]
        elif num_word.endswith('s'):
            s_flag = True
            num_word = num_word[:-1]

        if len(num_word) in [1, 3, 5]:
            num_word = self.inflect.number_to_words(num_word)
            if s_flag:  # 1s or 100s or 10000s
                num_word += ' s'
            s_flag = False

        if len(num_word) == 2:  # 50s
            try:
                w = int(num_word)
                num_word = self.inflect.number_to_words(w)
            except:  # pylint: disable=bare-except # noqa: E722
                s_flag = False

        elif len(num_word) == 4:
            try:
                w = int(num_word)

                if word.endswith('000'):
                    num_word = self.inflect.number_to_words(num_word)
                elif num_word.endswith('00'):
                    w1 = int(num_word[:2])
                    num_word = f"{self.inflect.number_to_words(w1)} hundred"
                elif 2000 < w < 2010:
                    num_word = self.inflect.number_to_words(num_word, andword="")
                else:
                    num_word = self.inflect.number_to_words(num_word, group=2)
            except:  # pylint: disable=bare-except # noqa: E722
                s_flag = False

        if s_flag:
            w = num_word.rsplit(' ', 1)
            num_word = self.inflect.plural(w[-1])
            if len(w) > 1:
                num_word = f"{w[0]} {num_word}"

        return num_word.lower() if self.lowercase else num_word

    def __call__(self, orig_utt):
        # pylint: disable=too-many-branches
        utt = orig_utt.lower() if self.lowercase else orig_utt
        utt = self.moses.normalize(utt)

        for k, v in self.mapping.items():
            utt = utt.replace(k, f" {v} ")

        if self.normalize_num and self.lang == "en":
            utt = utt.replace('-', ' ')
            matched_iter = re.finditer(r'([^ ]*\d+[^ ]*)', utt)

            try:
                first_match = next(matched_iter)
            except StopIteration:
                pass  # if no digits, do nothing
            else:
                current_position = 0
                utterance = []

                for m in itertools.chain([first_match], matched_iter):
                    start = m.start()
                    word = m.group().strip(string.punctuation)
                    before = utt[current_position:start]
                    if len(before) > 0:
                        utterance.append(before)

                    if word in self.mapping_num:
                        num_word = self.mapping_num[word]
                    else:
                        num_word = self._years(word)
                        if num_word == word:
                            num_word = self.inflect.number_to_words(
                                num_word, andword=""
                            )

                    if len(utterance) > 0 and not utterance[-1].endswith(' '):
                        num_word = ' ' + num_word
                    utterance.append(num_word)
                    current_position += start + len(word)

                if current_position < len(utt):
                    utterance.append(utt[current_position:])
                utt = ''.join(utterance)

        if self.escape is not None:
            for k, v in self.escape:
                utt = utt.replace(k, v)

            utt = re.sub(r'\([^)]+\)', self.escape[0][1], utt)
            utt = re.sub(r'\[[^\]]+\]', self.escape[1][1], utt)

        utt = re.sub(r'(\([^)]+\)|\[[^\]]+\])', ' ', utt)

        if self.lang == 'ja':
            return normalize_ja(utt)  # pylint: disable=undefined-variable # noqa: F821

        valid_char = ' a-z'
        if self.lang == 'de':
            valid_char += 'äöüß'

        if not self.normalize_num:
            valid_char += '0-9'

        if not self.lowercase:
            valid_char += 'A-Z'
            if self.lang == 'de':
                valid_char += 'ÄÖÜ'

        if self.remove_punc:
            valid_char += '\''
        else:
            # ascii punctuations only
            valid_char += string.punctuation
            # unicode punctuations
            # valid_char += ''.join[chr(i) for i in range(sys.maxunicode)
            #    if unicodedata.category(chr(i)).startswith('P')]

        if self.escape is not None:
            valid_char += '<>'
        utt = re.sub(r'[^' + valid_char + ']', ' ', utt)
        utt = re.sub(r'( )+', ' ', utt)

        if self.lowercase:
            utt.lower()
        return utt.strip()
