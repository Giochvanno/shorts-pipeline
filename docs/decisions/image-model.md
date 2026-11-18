# ADR: image model and prompt for scene frames

- **Status:** accepted, 2026-11-18
- **Decision:** Nano Banana 2.1 (`google/nano-banana-2.1` on fal) draws every frame;
  FLUX.2 [dev] (`fal-ai/flux-2`) is the fallback. The style line drops "natural history",
  the palette goes into the prompt in words as accents, and each scene brings its own light.
- **Where:** `models.images` in `config/channel.yaml`, `visual_bible` in
  `config/series/fact-of-day.yaml`, `build_image_prompt` in `pipeline/steps/visuals.py`.

## Context

A frame is the first frame of an animated clip. A wrong animal, a fake signature or
a mark that reads as blood gets animated by the video model and paid for again at
ten times the price of the frame. So accuracy decides; the price of a frame counts
only between models that are equally accurate (one frame costs cents, a clip of it
a dollar or two).

Up to video 10 the frames came from FLUX.1 [dev] (August 2024) with a "vintage
naturalist illustration" style. The wombat of video 10 looked like a hamster, a
frame carried a fake signature, and the subjects sat small on blank paper.

## Method

`python -m pipeline.compare_images` draws the 7 scenes of video 10 ("why wombats
make cube-shaped droppings") with each candidate: same prompts, same seed, frames
cached, spend in the ledger, a grid and a report per run. Every frame was scored by
eye: right subject, text or caption, style, light, and whether it would make a good
first frame for a video (a clear subject, a readable foreground and background).
The reports are `runs/10/compare/report.md` and `runs/10/compare-style3/report.md`.

| Round | What changed                                         | Frames | Cost   |
| ----- | ---------------------------------------------------- | ------ | ------ |
| 1     | 4 models, new style "clean digital ... painting"     | 28     | $0.749 |
| 2     | 2 newer models (the round-1 set was a year old)      | 14     | $0.560 |
| 3     | the chosen model, three prompt changes               | 7      | $0.350 |

## Results: the model (rounds 1 and 2)

| Model            | Right subject (yes / partly / no) | Text or caption | Good first frames | One style | 7 frames |
| ---------------- | --------------------------------- | --------------- | ----------------- | --------- | -------- |
| FLUX.1 [dev]     | 1 / 3 / 3                         | 1               | 1                 | partly    | $0.175   |
| FLUX.2 [dev]     | 5 / 2 / 0                         | 1               | 4                 | yes       | $0.084   |
| FLUX.2 [pro]     | 4 / 3 / 0                         | 0 (+1 plate)    | 3                 | partly    | $0.210   |
| Nano Banana      | 6 / 1 / 0                         | 0               | 5                 | yes       | $0.280   |
| Nano Banana 2.1  | **7 / 0 / 0**                     | 1 (+1 partly)   | 5                 | yes       | ~$0.28   |
| Seedream 5.5     | 4 / 3 / 0                         | 1 (heavy)       | 5                 | partly    | $0.210   |

FLUX.1 [dev] in the new style next to the video's own frames showed what the style
alone changes: half the signatures, cleaner pictures, but still no wombat. The model
was the main problem.

## Results: the prompt (round 3)

Same model and seed, so the differences come from the prompt alone.

| Nano Banana 2.1 | Right subject | Text or caption | Light matches the scene | Good first frames | One style |
| --------------- | ------------- | --------------- | ----------------------- | ----------------- | --------- |
| Round 2 prompt  | 7 / 0 / 0     | 1 + 1 partly    | 6 of 7                  | 5 of 7            | yes       |
| Round 3 prompt  | 7 / 0 / 0     | 0 + 1 partly    | 7 of 7                  | 6 of 7            | yes       |

- **No "natural history" in the style line.** It invited plate captions: Nano Banana
  2.1 printed "Lasiorhinus latifrons (Wombat) scat" under one frame, Seedream wrote
  "Vombatus ursinus". Without it the caption is gone (one frame: thin evidence; the
  frame review stays).
- **Light from the scene.** The bible's "soft diffused daylight" overrode the script's
  "moonlit": most models drew scene 2 by day. The script now gives each scene its
  light; the bible's is only the default. Scene 2 became night.
- **Colours in words.** Hex codes are text to an image model. Written as a plain list,
  though, the colours were painted literally: rose eucalyptus, a rose boulder, a pink
  armchair. So they go in as accents: "natural colours with accents of deep slate
  blue, warm sand and muted rose". To check on the next real video: natural colours of
  leaves, stones and furniture; text on backgrounds.

## Why Nano Banana 2.1

The only model with the right subject in all 7 scenes, and the closest to each
prompt: the intestine with alternating rings and the cube inside its last section,
the cube-shaped trophy, exactly as written. Its weak spot is text (the caption above,
scribbled papers behind a lab bench); the frame review in the visuals step rejects
such frames, and the retry draws with the next seed.

At $0.05 per frame in the ledger (about $0.04 billed), 7 frames plus a retry cost
about $0.40 a video: small next to $1-2 for the clips, and a wrong first frame
means a clip paid for twice.

## Why FLUX.2 [dev] as the fallback

For when the main model is unavailable, never mixed into one video (models differ
in style; a retry is the same model with the next seed). FLUX.2 [dev] is 5 of 7 right
at $0.012 a frame. The original Nano Banana scored higher (6 of 7) but is not the
fallback: on fal every Nano Banana is a proxy to Google, so one outage takes them
all out, while fal runs FLUX.2 [dev] from its open weights.

## Rejected

- **Seedream 5.5:** four labels written into one frame, two landscapes stacked with
  a seam in another, and no seed parameter: a good frame cannot be drawn again.
- **FLUX.2 [pro]:** no better than [dev] at 2.5 times the price and twice as slow;
  red patches like blood on the wombat, one frame drawn as a framed plate.
- **FLUX.1 [dev]:** a hamster, a kangaroo and a bear figurine instead of a wombat and
  a dropping.
- **Nano Banana (original):** a close second, kept out by the shared outage risk above.
- **Not tried:** Imagen 4 (being retired by Google), Nano Banana Pro and Nano Banana 2
  (over the $1 limit of round 1), GPT Image (about $0.21 a frame).

## Licence

- fal: the model page of `google/nano-banana-2.1` carries the "Commercial use" badge
  (checked 2026-11-16).
- Google, Gemini API Additional Terms of Service (effective 2026-03-23, checked
  2026-11-17): Google does not claim ownership of generated content; using it lawfully
  is on us. The services may not be part of an app directed at or likely used by people
  under 18: the channel is not made for kids.
- Every frame carries Google's invisible SynthID mark; uploads are disclosed as
  synthetic media (`upload.synthetic_media`).

## Seed

`tests/live/test_fal_live.py -k seed` draws one prompt twice with one seed
(2026-11-18):

| Model           | Same bytes | Equal values | Largest difference |
| --------------- | ---------- | ------------ | ------------------ |
| Nano Banana 2.1 | no         | 64.6 %       | 2 of 255           |
| FLUX.2 [dev]    | no         | 100 %        | 0                  |

Nano Banana 2.1 redraws the same picture with ±2 noise in the pixel values (99.4 % within
1): the same frame by eye and for the video model. So a good frame can be drawn again,
the comparison rounds compare prompts and not chance, and the cache (keyed by the
request, not by the output bytes) is unaffected.

## Consequences

- Frames cost about twice as much as with FLUX.1 [dev]; the stage limits allow it:
  $0.55 cash for images and $0.75 cash per video.
- The fallback is in config but not wired yet: today a video uses the main model only.
- Composition is no longer fixed by prompt (no model keeps the subject in the upper two
  thirds): captions get their own safe zone, and cropping by code may come later.
