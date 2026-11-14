Real voice checks from 2026-11-13: the narration of videos 7 and 8 (ElevenLabs,
eleven_multilingual_v2) and what faster-whisper small.en heard, as saved in
voice_transcript.json (`[word, start, end, probability]`).

Recorded with the old comma-separated vocabulary prompt: whisper put a comma after
most words and was unsure (p < 0.4) of correct sentence starters ("On", "Not") and
of "fifty". The old rule dropped those words: 97.0 % and 97.7 %; the new one counts
unsure words unless they are prompt words (which an echo could produce).

`video6_take1.json`, `video6_take2.json`: two takes of video 6 rejected at 92.7 % and
93.6 %. The voice was fine: whisper wrote "85 %" with a space (the "%" was not read
as "percent") and skipped "Number three." / "Number two." / "Number one.", each
between two pauses, stretching the next word over it. In take 1 the audio has
speech at about 2.6-3.3 s, 15.6-16.3 s and 24.1-24.7 s, where those phrases are.
