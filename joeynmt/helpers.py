# coding: utf-8
"""
Collection of helper functions
"""
from __future__ import annotations

import copy
import functools
import logging
import operator
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import packaging.version
import pkg_resources
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from joeynmt.plotting import plot_heatmap

np.set_printoptions(linewidth=sys.maxsize)  # format for printing numpy array


def make_model_dir(model_dir: Path, overwrite: bool = False):
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"Model directory {model_dir} exists "
                                  f"and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)  # create model_dir recursively


def make_logger(log_dir: Path = None, mode: str = "train") -> str:
    """
    Create a logger for logging the training/testing process.

    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: joeynmt version number
    """
    logger = logging.getLogger("")  # root logger
    version = pkg_resources.require("joeynmt")[0].version

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f"{mode}.log"

                fh = logging.FileHandler(log_file.as_posix(), encoding="utf-8")
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        logger.addHandler(sh)
        logger.info("Hello! This is Joey-NMT (version %s).", version)

    return version


def check_version(pkg_version: str, cfg_version: str) -> None:
    """
    Check joeynmt version

    :param pkg_version: package version number
    :param cfg_version: version number specified in config
    """
    joeynmt_version = packaging.version.parse(pkg_version)
    config_version = packaging.version.parse(cfg_version)
    # check if the major version number matches
    # pylint: disable=use-maxsplit-arg
    assert joeynmt_version.major == config_version.major, (
        f"You are using JoeyNMT version {str(joeynmt_version)}, "
        f'but {str(config_version)} is expected in the given config.')


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return: cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def write_list_to_file(output_path: Path, array: List[Any]) -> None:
    """
    Write list of str to file in `output_path`.

    :param output_path: output file path
    :param array: list of strings
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            opened_file.write(f"{entry}\n")


def read_list_from_file(input_path: Path) -> List[str]:
    """
    Read list of str from file in `input_path`.

    :param input_path: input file path
    :return: list of strings
    """
    if input_path is None:
        return []
    return [
        line.rstrip("\n")
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]


def save_hypothese(output_path: Path, hypotheses: List[str], n_best: str = 1) -> None:
    """
    Save list hypothese to file.

    :param output_path: output file path
    :param hypotheses: hypothese to write
    :param n_best: n_best size
    """
    if n_best > 1:
        for n in range(n_best):
            write_list_to_file(
                output_path.parent / f"{output_path.stem}-{n}.{output_path.suffix}",
                [hypotheses[i] for i in range(n, len(hypotheses), n_best)],
            )
    else:
        write_list_to_file(output_path, hypotheses)


def store_attention_plots(
    attentions: np.ndarray,
    targets: List[List[str]],
    sources: List[List[str]],
    output_prefix: str,
    indices: List[int],
    tb_writer: Optional[SummaryWriter] = None,
    steps: int = 0,
) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = f"{output_prefix}.{i}.png"
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(
                scores=attention_scores,
                column_labels=trg,
                row_labels=src,
                output_path=plot_file,
                dpi=100,
            )
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(
                    scores=attention_scores,
                    column_labels=trg,
                    row_labels=src,
                    output_path=None,
                    dpi=50,
                )
                tb_writer.add_figure(f"attention/{i}.", fig, global_step=steps)
        except Exception:  # pylint: disable=broad-except
            print(f"Couldn't plot example {i}: "
                  f"src len {len(src)}, trg len {len(trg)}, "
                  f"attention scores shape {attention_scores.shape}")
            continue


def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Returns the latest checkpoint (by creation time, not the steps number!)
    from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = ckpt_dir.glob("*.ckpt")
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=lambda f: f.stat().st_ctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in directory {ckpt_dir}.")
    return latest_checkpoint


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param device: cuda device name or cpu
    :return: checkpoint (dict)
    """
    logger = logging.getLogger(__name__)
    assert path.is_file(), f"Checkpoint {path} not found."
    checkpoint = torch.load(path, map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return checkpoint


def resolve_ckpt_path(load_model: Path, model_dir: Path) -> Path:
    """
    Resolve checkpoint path

    :param load_model: config entry (cfg['training']['load_model']) or CLI arg (--ckpt)
    :param model_dir: Path(cfg['training']['model_dir'])
    :return: resolved checkpoint path
    """
    if load_model is None:
        if (model_dir / "best.ckpt").is_file():
            load_model = model_dir / "best.ckpt"
        else:
            load_model = get_latest_checkpoint(model_dir)
    assert load_model.is_file(), load_model
    return load_model


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    # yapf: disable
    x = (x.view(batch, -1)
         .transpose(0, 1)
         .repeat(count, 1)
         .transpose(0, 1)
         .contiguous()
         .view(*out_size))
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def delete_ckpt(to_delete: Path) -> None:
    """
    Delete checkpoint

    :param to_delete: checkpoint file to be deleted
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("delete %s", to_delete.as_posix())
        to_delete.unlink()

    except FileNotFoundError as e:
        logger.warning(
            "Wanted to delete old checkpoint %s but "
            "file does not exist. (%s)",
            to_delete,
            e,
        )


def symlink_update(target: Path, link_name: Path) -> Optional[Path]:
    """
    This function finds the file that the symlink currently points to, sets it
    to the new target, and returns the previous target if it exists.

    :param target: A path to a file that we want the symlink to point to.
                    no parent dir, filename only, i.e. "10000.ckpt"
    :param link_name: This is the name of the symlink that we want to update.
                    link name with parent dir, i.e. "models/my_model/best.ckpt"

    :return:
        - current_last: This is the previous target of the symlink, before it is
            updated in this function. If the symlink did not exist before or did
            not have a target, None is returned instead.
    """
    if link_name.is_symlink():
        current_last = link_name.resolve()
        link_name.unlink()
        link_name.symlink_to(target)
        return current_last
    link_name.symlink_to(target)
    return None


def flatten(array: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().

    :param array: a nested list
    :return: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])


def expand_reverse_index(reverse_index: List[int], n_best: int = 1) -> List[int]:
    """
    Expand resort_reverse_index for n_best prediction

    ex. 1) reverse_index = [1, 0, 2] and n_best = 2, then this will return
    [2, 3, 0, 1, 4, 5].

    ex. 2) reverse_index = [1, 0, 2] and n_best = 3, then this will return
    [3, 4, 5, 0, 1, 2, 6, 7, 8]

    :param reverse_index: reverse_index returned from batch.sort_by_src_length()
    :param n_best:
    :return: expanded sort_reverse_index
    """
    if n_best == 1:
        return reverse_index

    resort_reverse_index = []
    for ix in reverse_index:
        for n in range(0, n_best):
            resort_reverse_index.append(ix * n_best + n)
    assert len(resort_reverse_index) == len(reverse_index) * n_best
    return resort_reverse_index


def remove_extra_spaces(s: str) -> str:
    """
    Remove extra spaces
    - used in pre_process() / post_process() in tokenizer.py

    :param s: input string
    :return: string w/o extra white spaces
    """
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)

    s = s.replace(" ?", "?")
    s = s.replace(" !", "!")
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(" :", ":")
    return s.strip()


def unicode_normalize(s: str) -> str:
    """
    apply unicodedata NFKC normalization
    - used in pre_process() in tokenizer.py

    :param s: input string
    :return: normalized string
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'")
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    return s
