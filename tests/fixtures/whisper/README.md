Real voice checks from 2026-11-13: the narration of videos 7 and 8 (ElevenLabs,
eleven_multilingual_v2) and what faster-whisper small.en heard, as saved in
voice_transcript.json (`[word, start, end, probability]`).

Recorded with the old comma-separated vocabulary prompt: whisper put a comma after
most words and was unsure (p < 0.4) of correct sentence starters ("On", "Not") and
of "fifty". The old rule dropped those words: 97.0 % and 97.7 %; the new one counts
unsure words unless they are prompt words (which an echo could produce).
