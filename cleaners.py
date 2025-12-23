""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

# import phonemizer
from unidecode import unidecode

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
# global_phonemizer = phonemizer.backend.EspeakBackend(
#     language="en-us",
#     preserve_punctuation=True,
#     with_stress=True,
#     language_switch="remove-flags",
#     logger=critical_logger,
# )


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = re.sub(r'(?<!s)x', 'h', text)
    text = re.sub(r'(?<!s)X', 'H', text)
    text = text.replace('‘', "'").replace('’', "'")
    # Remove all punctuation except ,.?!'
    allowed_punct = ",.?!'"
    text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in allowed_punct)
    text = convert_numbers_in_string_to_uzbek(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


def number_to_uzbek(num: int) -> str:
    """
    Convert an integer (0 <= num <= 999,999,999,999) into its Uzbek textual form.
    """
    # Handle zero explicitly
    if num == 0:
        return "nol"

    # Define small number mappings
    ones = {
        0: "",
        1: "bir",
        2: "ikki",
        3: "uch",
        4: "to'rt",
        5: "besh",
        6: "olti",
        7: "yetti",
        8: "sakkiz",
        9: "to'qqiz",
    }

    tens = {
        10: "o'n",
        20: "yigirma",
        30: "o'ttiz",
        40: "qirq",
        50: "ellik",
        60: "oltmish",
        70: "yetmish",
        80: "sakson",
        90: "to'qson",
    }

    # Larger scale names in Uzbek
    scales = [
        (10**9, "milliard"),
        (10**6, "million"),
        (10**3, "ming"),
        (1, ""),  # for loop consistency
    ]

    def three_digit_to_uzbek(n: int) -> str:
        """
        Convert an integer 0 <= n < 1000 into its Uzbek textual representation.
        """
        assert 0 <= n < 1000
        parts = []

        # Hundreds
        hundreds = n // 100
        remainder = n % 100

        if hundreds > 0:
            if hundreds == 1:
                parts.append("yuz")  # "1 yuz" = "yuz"
            else:
                parts.append(ones[hundreds] + " yuz")

        # Tens & ones
        if remainder != 0:
            if remainder < 10:
                parts.append(ones[remainder])
            elif remainder in tens:
                parts.append(tens[remainder])
            else:
                ten_part = (remainder // 10) * 10
                one_part = remainder % 10
                if ten_part > 0:
                    parts.append(tens[ten_part])
                if one_part > 0:
                    parts.append(ones[one_part])

        return " ".join(parts).strip()

    words = []
    current = num

    # Break the number down by large scales (milliard, million, ming, ...)
    for scale_value, scale_name in scales:
        if current == 0:
            break

        chunk = current // scale_value
        current = current % scale_value

        if chunk == 0:
            continue

        chunk_text = three_digit_to_uzbek(chunk)
        if scale_name:
            chunk_text += " " + scale_name
        words.append(chunk_text)

    return " ".join(words).strip()


def convert_numbers_in_string_to_uzbek(text: str) -> str:
    """
    Find all numeric substrings in the given text and replace each with its
    Uzbek textual representation.
    """

    # This function (replacer) will be called for each match of the \d+ pattern.
    # match.group(0) is the full numeric substring.
    def replacer(match):
        numeric_str = match.group(0)
        # Safely convert to int and pass through number_to_uzbek
        # If it's too large or invalid, we could handle that, but here we assume
        # it fits into an int and is valid.
        numeric_value = int(numeric_str)
        return number_to_uzbek(numeric_value)

    # Use re.sub with a function that replaces each \d+ match
    new_text = re.sub(r"\d+", replacer, text)
    return new_text
