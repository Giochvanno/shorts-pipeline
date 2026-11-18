# Image model: Nano Banana 2.1, fallback FLUX.2 [dev]

Decided 2026-11-18. Config: `models.images` in `config/channel.yaml`.

## Why it matters

A frame is the first frame of an animated clip. A wrong animal, a fake signature or
a mark that reads as blood gets animated by the video model and paid for again at
ten times the price. So accuracy comes first; the price of a frame counts only
between models that are equally accurate.

## How it was chosen

`python -m pipeline.compare_images` drew the 7 scenes of video 10 ("why wombats make
cube-shaped droppings") with each model, same prompts, same seed; every frame was
scored by eye (`runs/10/compare/report.md`).

| Model            | Right subject (yes / partly / no) | Text or caption | Good first frames | One style | 7 frames |
| ---------------- | --------------------------------- | --------------- | ----------------- | --------- | -------- |
| FLUX.1 [dev]     | 1 / 3 / 3                         | 1               | 1                 | partly    | $0.175   |
| FLUX.2 [dev]     | 5 / 2 / 0                         | 1               | 4                 | yes       | $0.084   |
| FLUX.2 [pro]     | 4 / 3 / 0                         | 0 (+1 plate)    | 3                 | partly    | $0.210   |
| Nano Banana      | 6 / 1 / 0                         | 0               | 5                 | yes       | $0.280   |
| Nano Banana 2.1  | **7 / 0 / 0**                     | 1 (+1 partly)   | 5                 | yes       | ~$0.28   |
| Seedream 5.5     | 4 / 3 / 0                         | 1 (heavy)       | 5                 | partly    | $0.210   |

- **Nano Banana 2.1** is the only model with the right subject in all 7 scenes and the
  closest to each prompt. Its weak spot is text (a Latin-name caption in one frame):
  the frame review (`pipeline/adapters/vision/`) rejects such frames, and the retry
  draws with the next seed.
- **FLUX.2 [dev]** is the fallback, for when the main model is unavailable; one video
  never mixes models. Not the original Nano Banana, although it scored higher: on fal
  every Nano Banana is a proxy to Google, so one outage takes them all out, while
  fal runs FLUX.2 [dev] from its open weights.
- Seedream 5.5 is out: four labels in one frame, two landscapes stacked in another,
  and no seed, so a good frame cannot be drawn again. FLUX.2 [pro] is not better than
  [dev] at 2.5 times the price.

## Licence

- fal: the model page of `google/nano-banana-2.1` carries the "Commercial use" badge
  (checked 2026-11-16).
- Google, Gemini API Additional Terms of Service (effective 2026-03-23, checked
  2026-11-17): Google does not claim ownership of generated content; using it lawfully
  is on us. The services may not be part of an app directed at or likely used by people
  under 18: the channel is not made for kids.
- Every frame carries Google's invisible SynthID mark; uploads are disclosed as
  synthetic media.

## Seed

`tests/live/test_fal_live.py -k seed` draws one prompt twice with one seed
(2026-11-18):

| Model           | Same bytes | Equal values | Largest difference |
| --------------- | ---------- | ------------ | ------------------ |
| Nano Banana 2.1 | no         | 64.6 %       | 2 of 255           |
| FLUX.2 [dev]    | no         | 100 %        | 0                  |

Nano Banana 2.1 redraws the same picture with ±2 noise in the pixel values (99.4 % within
1): the same frame by eye and for the video model. So a good frame can be drawn again,
and the cache (keyed by the request, not by the output bytes) is unaffected.

## Open

- The fallback is in config but not wired yet: today a video uses the primary model only.
- Round 3 (style without "natural history", colours in words, light from the scene)
  decides the prompt, not the model.
