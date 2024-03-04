""" from https://github.com/keithito/tacotron """
from matcha.text import cleaners
from matcha.text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text, starts_with = _clean_text(text, cleaner_names)
    space_id = _symbol_to_id[" "]
    for i, phoneme_phrase in enumerate(clean_text):
        phrase_seq = []
        for symbol in phoneme_phrase:
            symbol_id = _symbol_to_id[symbol]
            phrase_seq += [symbol_id]
        
        if i != len(clean_text)-1:
            phrase_seq += [space_id]
        sequence.append(phrase_seq)
    
    return sequence, starts_with


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text, starts_with = cleaner(text)
    return text, starts_with
