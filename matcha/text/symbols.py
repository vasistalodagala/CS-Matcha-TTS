""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_additional_pad = "^"  # padding symbol when more than one language is in place. Eg: A code-switching case
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_additional_arabic_phone = '̪'
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


# Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
symbols = [_pad] + [_additional_pad] + list(_punctuation) + list(_letters) + list(_additional_arabic_phone) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")
