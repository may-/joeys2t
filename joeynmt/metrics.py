# coding: utf-8
"""
Evaluation metrics
"""
from inspect import getfullargspec
from typing import Callable, List

import jiwer
from sacrebleu.metrics import BLEU, CHRF

from joeynmt.helpers_for_ddp import get_logger
from joeynmt.tokenizers import BasicTokenizer

logger = get_logger(__name__)


def chrf(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Character F-score from sacrebleu

    .. seealso::
        https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: character f-score (0 <= chf <= 1)
             see Breaking Change in sacrebleu v2.0
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(CHRF).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = CHRF(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    # log sacrebleu signature
    logger.info(metric.get_signature())
    return score / 100


def bleu(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    .. seealso::
        https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: bleu score
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(BLEU).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = BLEU(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    # log sacrebleu signature
    logger.info(metric.get_signature())
    return score


def token_accuracy(
    hypotheses: List[str], references: List[str], tokenizer: Callable
) -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    We lookup the references before one-hot-encoding, that is, UNK generation in
    hypotheses is always evaluated as incorrect.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: token accuracy (float)
    """
    correct_tokens = 0
    all_tokens = 0
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        hyp = tokenizer(hyp)
        ref = tokenizer(ref)
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp, ref):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    We lookup the references before one-hot-encoding, that is, hypotheses with UNK
    are always evaluated as incorrect.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: sequence accuracy (float)
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([
        1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref
    ])
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


class ReduceToListOfListOfWords(jiwer.ReduceToListOfListOfWords):
    """
    Define transformation

    .. seealso::
        https://github.com/jitsi/jiwer/blob/master/jiwer/transforms.py

    """

    def __init__(self, tokenizer: BasicTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def process_string(self, s: str):
        return [self.tokenizer(s)]


def get_transform(tokenizer) -> jiwer.Compose:
    """
    Get transformation

    .. seealso::
        https://github.com/jitsi/jiwer/blob/master/jiwer/transformations.py

    """
    return jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        # jiwer.ReduceToListOfListOfWords(),
        ReduceToListOfListOfWords(tokenizer),
    ])


def wer(hypotheses, references, transform) -> float:
    """
    Compute word error rate in corpus-level

    .. seealso::
        https://github.com/jitsi/jiwer/blob/master/jiwer/measures.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param transform: tokenizer
    :return: normalized word error rate
    """
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    ) * 100.0
