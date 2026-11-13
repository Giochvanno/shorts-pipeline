"""Word-by-word captions as ASS subtitles, drawn by libass inside FFmpeg.

ASS (Advanced SubStation Alpha) is the karaoke/fansub format: styled text with an
outline, exact placement, and colour changes inside a line. Each caption chunk
("Honey outlives pharaohs.") becomes one event per word; in each event the word
being spoken is coloured with the series' highlight colour:

    0.00-0.40  [Honey] outlives pharaohs.
    0.40-0.80  Honey [outlives] pharaohs.
    0.80-1.40  Honey outlives [pharaohs.]

Pure functions only: the render step writes the result next to FFmpeg.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.steps.captions import CaptionWord, Chunk

# A chunk stays up this long after its last word, unless the next chunk starts sooner:
# no flicker between chunks, no stale text over long pauses.
LINGER_S = 0.3
_MIN_EVENT_S = 0.05


@dataclass(frozen=True, slots=True)
class CaptionStyle:
    font: str  # family name inside the font file, e.g. "Montserrat ExtraBold"
    size: int
    color: str  # "#RRGGBB"
    highlight: str  # "#RRGGBB", the word being spoken
    outline: int
    margin_v: int  # px from the bottom edge, clear of YouTube's own buttons
    margin_h: int
    width: int = 1080
    height: int = 1920


def ass_color(hex_rgb: str) -> str:
    """ "#C1666B" -> "&H006B66C1&": ASS writes colours as alpha, blue, green, red."""
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H00{b}{g}{r}&".upper()


def ass_time(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.cc (centiseconds)."""
    cs = max(round(seconds * 100), 0)
    h, rest = divmod(cs, 360_000)
    m, rest = divmod(rest, 6_000)
    s, cs = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    """Braces start override tags and a backslash starts escapes; neither belongs in captions."""
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def header(style: CaptionStyle) -> str:
    color = ass_color(style.color)
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {style.width}",
            f"PlayResY: {style.height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
            " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
            " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{style.font},{style.size},{color},{color},&H00000000&,&H00000000&,"
            f"0,0,0,0,100,100,0,0,1,{style.outline},0,2,"
            f"{style.margin_h},{style.margin_h},{style.margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )


def build_ass(
    words: Sequence[CaptionWord],
    chunks: Sequence[Chunk],
    style: CaptionStyle,
    *,
    first_word: int = 0,
) -> str:
    """The whole subtitle file. Words before `first_word` are left out (the hook,
    which is already on screen in large type)."""
    visible = [(max(c.first, first_word), c.last) for c in chunks if c.last >= first_word]
    on, off = ass_color(style.highlight), ass_color(style.color)
    lines = [header(style)]
    for n, (first, last) in enumerate(visible):
        next_start = words[visible[n + 1][0]].start if n + 1 < len(visible) else float("inf")
        chunk_end = min(next_start, words[last].end + LINGER_S)
        for k in range(first, last + 1):
            start = words[k].start
            end = words[k + 1].start if k < last else chunk_end
            end = max(end, start + _MIN_EVENT_S)
            text = " ".join(
                f"{{\\1c{on}}}{escape(w.text)}{{\\1c{off}}}" if i == k else escape(w.text)
                for i, w in enumerate(words[first : last + 1], start=first)
            )
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"
