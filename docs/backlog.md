# Backlog

Decided to do later, not now. Each item says why it waits and what would start it.

## Frame review on a cheaper Claude model

The review runs on the heavy model: about $0.03 of subscription credit per frame,
about $0.34 per video (video 12: 11 frames). Reviewing video clips will add more.

Experiment, free of image cost: run the saved frames (accepted ones from `frames/`,
turned-down ones from `rejected_frames/`, each with the heavy model's verdict in its
meta or json) through a cheaper model and compare the verdicts.

Switch only if both hold:
- the verdicts agree on at least 95 % of frames;
- the cheaper model turns down every frame the heavy one turned down for text or
  unwanted content. Most frames are clean, so 95 % agreement alone could still miss
  the rare frames the review exists for (small garbled shop signs in video 12).

Start after about 10 videos with the review, when there are enough turned-down frames
to compare (`pipeline status` counts them by reason).

## Scenes drawn in parallel

Visuals take about 270 s for 8 scenes (video 12): about 19 s to draw a frame, 6 s to
review it, scene after scene. Videos are made at night, so speed does not matter yet.
When it does: draw and review several scenes at once, keeping one seed per video and
the per-scene retries; the image limit per video stays the guard.
