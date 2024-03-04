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

import phonemizer
import piper_phonemize
from unidecode import unidecode

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

arabic_phonemizer = phonemizer.backend.EspeakBackend(
    language="ar",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

additional_language_utf8_ranges = {
    'arabic': [u'\u0600', u'\u06ff']
}

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
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


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return [text], None


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return [text], None


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return [phonemes], None


def english_cleaners_piper(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
    phonemes = collapse_whitespace(phonemes)
    return [phonemes], None


def arabic_cleaners(text):
    """Pipeline for Arabic text, including abbreviation expansion. + punctuation + stress"""
    phonemes = arabic_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return [phonemes], None


def arabic_cleaners_piper(text):
    """Pipeline for Arabic text, including abbreviation expansion. + punctuation + stress"""
    phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="ar")[0])
    phonemes = collapse_whitespace(phonemes)
    return [phonemes], None


def get_utf8_range(c):
    for lang in additional_language_utf8_ranges.keys():
        range_check = additional_language_utf8_ranges[lang][0] <= c <= additional_language_utf8_ranges[lang][1]
        if range_check:
            break
    return int(range_check)


def get_cs_list(text):
    cs_seq_list = []
    if len(text.split()) == 1:
        cs_seq_list.append(text)
    else:
        cs_seq_list.append(text.split()[0])
        for word in text.split()[1:]:
            if get_utf8_range(word[0]) == get_utf8_range(cs_seq_list[-1][0]):
                cs_seq_list[-1] += ' ' + word
            else:
                cs_seq_list.append(word)
    return cs_seq_list


def cs_eng_ara_cleaners(text):
    """Pipeline for text to handle English and Arabic, including abbreviation expansion. + punctuation + stress"""
    phonemes = []
    cs_seq_list = get_cs_list(text)
    if get_utf8_range(cs_seq_list[0][0]) == 1:
        for i, phrase in enumerate(cs_seq_list):
            if i%2 == 0:
                ar_phoneme = arabic_phonemizer.phonemize([phrase], strip=True, njobs=1)[0]
                ar_phoneme = collapse_whitespace(ar_phoneme)
                ar_phoneme.strip()
                phonemes.append(ar_phoneme)
            else:
                phrase = convert_to_ascii(phrase)
                phrase = lowercase(phrase)
                phrase = expand_abbreviations(phrase)
                en_phoneme = global_phonemizer.phonemize([phrase], strip=True, njobs=1)[0]
                en_phoneme = collapse_whitespace(en_phoneme)
                en_phoneme.strip()
                phonemes.append(en_phoneme)
        return phonemes, "ar"
    
    else:
        for i, phrase in enumerate(cs_seq_list):
            if i%2 == 0:
                phrase = convert_to_ascii(phrase)
                phrase = lowercase(phrase)
                phrase = expand_abbreviations(phrase)
                en_phoneme = global_phonemizer.phonemize([phrase], strip=True, njobs=1)[0]
                en_phoneme = collapse_whitespace(en_phoneme)
                en_phoneme.strip()
                phonemes.append(en_phoneme) 
            else:
                ar_phoneme = arabic_phonemizer.phonemize([phrase], strip=True, njobs=1)[0]
                ar_phoneme = collapse_whitespace(ar_phoneme)
                ar_phoneme.strip()
                phonemes.append(ar_phoneme)
        return  phonemes, "en"


def cs_eng_ara_cleaners_piper(text):
    """Pipeline for text to handle English and Arabic, including abbreviation expansion. + punctuation + stress"""
    phonemes = []
    cs_seq_list = get_cs_list(text)
    if get_utf8_range(cs_seq_list[0][0]) == 1:
        for i, phrase in enumerate(cs_seq_list):
            if i%2 == 0:
                ar_phoneme = "".join(piper_phonemize.phonemize_espeak(text=phrase, voice="ar")[0])
                ar_phoneme = collapse_whitespace(ar_phoneme)
                ar_phoneme.strip()
                phonemes.append(ar_phoneme)
            else:
                phrase = convert_to_ascii(phrase)
                phrase = lowercase(phrase)
                phrase = expand_abbreviations(phrase)
                en_phoneme = "".join(piper_phonemize.phonemize_espeak(text=phrase, voice="en-US")[0])
                en_phoneme = collapse_whitespace(en_phoneme)
                en_phoneme.strip()
                phonemes.append(en_phoneme)
        return phonemes, "ar"
    
    else:
        for i, phrase in enumerate(cs_seq_list):
            if i%2 == 0:
                phrase = convert_to_ascii(phrase)
                phrase = lowercase(phrase)
                phrase = expand_abbreviations(phrase)
                en_phoneme = "".join(piper_phonemize.phonemize_espeak(text=phrase, voice="en-US")[0])
                en_phoneme = collapse_whitespace(en_phoneme)
                en_phoneme.strip()
                phonemes.append(en_phoneme) 
            else:
                ar_phoneme = "".join(piper_phonemize.phonemize_espeak(text=phrase, voice="ar")[0])
                ar_phoneme = collapse_whitespace(ar_phoneme)
                ar_phoneme.strip()
                phonemes.append(ar_phoneme)
        return phonemes, "en"

