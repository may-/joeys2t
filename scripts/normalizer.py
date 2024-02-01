#!/usr/bin/env python3
# coding: utf-8

# Taken from
# https://github.com/openai/whisper/tree/main/whisper/normalizers

import itertools
import json
import re
import string
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Match, Optional, Union

import truecase
from more_itertools import windowed

from joeynmt.helpers import remove_extra_spaces

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c if c in keep else ADDITIONAL_DIACRITICS[c] if c in
        ADDITIONAL_DIACRITICS else "" if unicodedata.category(c) ==
        "Mn" else " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


def normalize_unicode(template: str, s: str) -> str:
    """
    normalize unicode
    """
    # pylint: disable=consider-using-f-string
    pt = re.compile("([{}]+)".format(template))

    def _norm(c):
        return unicodedata.normalize("NFKC", c) if pt.match(c) else c

    s = "".join(_norm(x) for x in re.split(pt, s))
    return s


class BasicTextNormalizer:

    def __init__(
        self,
        remove_diacritics: bool = False,
        split_letters: bool = False,
        lower: bool = False
    ) -> str:
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters
        self.lower = lower
        self.escape = None

    def __call__(self, s: str):
        s = s.strip()

        if self.escape is not None:
            for a, b, _ in self.escape:
                s = s.replace(a, b)

        s = re.sub(r"[<][^>]*[>]", "", s)  # remove words between brackets
        s = re.sub(r"[\[][^\]]*[\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis

        if self.escape is not None:
            for _, b, c in self.escape:
                s = s.replace(b, c)

        s = self.clean(s)

        if self.lower:
            s = s.lower()

        if self.split_letters:
            s = " ".join(re.findall(r"\X", s, re.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s.strip()

    def _escape(self, s: str) -> str:
        if self.escape is not None:
            for k, v in self.escape:
                s = s.replace(k, v)
        return s


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million`
        -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    # pylint: disable=too-many-instance-attributes, consider-using-set-comprehension
    def __init__(self):
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        # yapf: disable
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items() if (value > 3
                                                         and value != 5
                                                         and value != 12)
            },
        }
        # yapf: enable
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (value, "s")
            for name, value in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (value, "th")
            for name, value in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (value, "s")
            for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (value, "th")
            for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(
            list(self.preceding_prefixers.values())
            + list(self.following_prefixers.values())
        )
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = set([
            key for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ] for key in mapping
        ])
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        # pylint: disable=too-many-branches, too-many-statements
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next_ in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next_ is not None and re.match(r"^\d+(\.\d+)?$", next_)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    # pylint: disable=no-else-continue
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if (
                        prev in self.tens and ones < 10
                    ):  # replace the last zero with the digit
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next_ in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next_ in suffix:
                            yield output(str(value) + suffix[next_])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next_ not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current in ["double", "triple"]:
                    if next_ in self.ones or next_ in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next_, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next_ in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):

        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self):
        mapping_path = Path(__file__).parent / "english.json"
        self.mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:

    def __init__(self, lower: bool = True):
        self.lower = lower
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<][^>]*[>]", "", s)  # remove words between brackets

        if self.escape:
            for k, v in self.escape:
                s = s.replace(k, v)

        s = re.sub(r"[\[][^\]]*[\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis

        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        if not self.lower:
            s = truecase.get_true_case(s)

        return s.strip()


class JapaneseNormalizer:

    def __init__(self, lower: bool = True):
        self.lower = lower

    def __call__(self, s: str) -> str:
        s = s.strip()
        s = re.sub("\t", " ", s)
        s = normalize_unicode("０-９Ａ-Ｚａ-ｚ｡-ﾟ", s)

        def _maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub("[˗֊‐‑‒–⁃⁻₋−]+", "-", s)  # normalize hyphens
        s = re.sub("[﹣－ｰ—―─━ー]+", "ー", s)  # normalize choonpus
        s = re.sub("[~∼∾〜〰～]+", "〜", s)  # normalize tildes (modified by Isao Sonobe)
        s = s.translate(
            _maketrans(
                "!\"#$%&'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣",
                "！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」",
            )
        )

        s = remove_extra_spaces(s)
        s = normalize_unicode("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜", s)  # keep ＝,・,「,」
        s = re.sub("[’]", "'", s)
        s = re.sub("[”]", '"', s)
        s = re.sub("[“]", '"', s)
        return s


class Normalizer:
    """
    OBSOLATE: Text normalizer
    used in the experiments for https://arxiv.org/abs/2210.02545
    """
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
            ('(Geräusch)', '<noise>'),
            ('[unklar]', '<unclear>'),
            ('(Lachen)', '<laughter>'),
            ('(Lacht)', '<laughter>'),
            ('(lacht)', '<laughter>'),
            ('(Gelächter)', '<laughter>'),
            ('(Gelaechter)', '<laughter>'),
            ('(Applaus)', '<applause>'),
            ('(Applause)', '<applause>'),
            ('(Beifall)', '<applause>'),
        ],
        'ja': [
            ('（ため息）', '<noise>'),
            ('(笑)', '<applause>'),
            ('（笑）', '<applause>'),
            ('（笑い）', '<applause>'),
            ('（笑い声）', '<applause>'),
            ('（歌う）', '<music>'),
            ('（音楽）', '<music>'),
            ('（ヒーローの音楽）', '<music>'),
            ('（大音量の音楽）', '<music>'),
            ('(ビデオ)', '<music>'),
            ('（ビデオ）', '<music>'),
            ('（映像と音楽）', '<music>'),
            ('(映像)', '<music>'),
            ('（映像）', '<music>'),
            ('(拍手)', '<applause>'),
            ('（拍手）', '<applause>'),
            ('（録音済みの拍手）', '<applause>'),
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
