# Inference Commands

Support for `Arabic` language added as a part of this repository.

The `file` parameter should be pointing to the text file containing the lines to be infered.
The `language` parameter supports `english` and `arabic` as the possible options.
The `checkpoint_path` parameter should be pointing to the Matcha-TTS checkpoint.
The `custom_vocoder_path` parameter should be pointing to the HiFi-GAN generator checkpoint.
The `output_folder` parameter is the location where the synthesized audios are stored.
The `output_file` parameter is the output file name for the synthesized audio during single sentence inference.
The `cpu` parameter indicates to run the synthesis on CPU.


To synthesize text from a file:
```bash
matcha-tts \
--file <path-to-the-text-file> \
--language arabic \
--checkpoint_path <path-to-the-matcha_tts-checkpoint> \
--custom_vocoder_path <path-to-the-HiFi_GAN-checkpoint> \
--output_folder <path-to-the-output-folder>
```


To synthesize a single sentence:
```bash
matcha-tts \
--text "وَطَارَتْ أُخْرَى إِلَى دَاخِلِ حَقِيبَةِ أُخْتِ حَنِينْ، الَّتِي تَعْمَلُ فِي مَصْنَعٍ يَعْصِرُ الزَّيْتُونَ لِيَصْنَعَ مِنْهُ الصَّابُونْ." \
--language arabic \
--checkpoint_path <path-to-the-matcha_tts-checkpoint> \
--custom_vocoder_path <path-to-the-HiFi_GAN-checkpoint> \
--output_folder <path-to-the-output-folder> \
--output_file <name-of-the-utterance-file-without-extension>
```

To synthesize a code-mixed sentence:
```bash
matcha-tts \
--text "Even in Italy most of the theological and  إيهْ أُغْنيِّةْ اِلْخِتامْ دي law books were printed in Gothic letter  بِتِعْمِلوا كِدَهْ ليهْ في اِلزَّمالِكْ" \
--language code_mixed \
--checkpoint_path <path-to-the-matcha_tts-checkpoint> \
--custom_vocoder_path <path-to-the-HiFi_GAN-checkpoint> \
--output_folder <path-to-the-output-folder> \
--output_file <name-of-the-utterance-file-without-extension>
```