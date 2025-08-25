import time
import hmac
import hashlib
import random
from typing import Tuple

from llm_guard.util import get_logger

from .base import Scanner


LOGGER = get_logger()


HOMOGLYPHS = {
    "a": ["а", "ɑ", "α"],
    "e": ["е", "ҽ", "ℯ"],
    "i": ["і", "ι", "ӏ"],
    "o": ["о", "ο", "օ"],
    "c": ["с", "ϲ"],
    "p": ["р", "ρ"],
    "x": ["х", "ꭓ"],
    "y": ["у", "γ"],
    "A": ["Α", "А"],
    "E": ["Ε", "Е"],
    "I": ["Ι", "І"],
    "O": ["Ο", "О"],
    "C": ["С", "Ϲ"],
    "P": ["Ρ", "Р"],
    "X": ["Χ", "Х"],
    "Y": ["Υ", "У"],
}


INVISIBLE_CHARS = [
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
]


class TimeLockObfuscator(Scanner):
    """
    Obfuscates prompts for LLM providers using:
    - low-distortion homoglyph substitutions
    - sparse zero-width characters
    - a time-lock header with TTL and HMAC tag (for audit/verification)

    This keeps prompts readable to LLMs while degrading exact string matching
    at the provider side and attaching ephemeral validity metadata.
    """

    def __init__(
        self,
        *,
        secret: str | None = None,
        ttl_seconds: int = 300,
        time_step_seconds: int = 60,
        homoglyph_probability: float = 0.12,
        zero_width_probability: float = 0.08,
        tag_prefix: str = "<tl:v1",
    ) -> None:
        self.secret = (secret or "changeme_timelock_secret").encode("utf-8")
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.time_step_seconds = max(1, int(time_step_seconds))
        self.homoglyph_probability = max(0.0, min(1.0, float(homoglyph_probability)))
        self.zero_width_probability = max(0.0, min(1.0, float(zero_width_probability)))
        self.tag_prefix = tag_prefix

    def _current_window(self) -> int:
        return int(time.time() // self.time_step_seconds)

    def _compute_tag(self, window: int, expires_at: int) -> str:
        msg = f"{window}:{expires_at}".encode("utf-8")
        digest = hmac.new(self.secret, msg, hashlib.sha256).hexdigest()
        return digest[:16]

    def _embed_header(self, text: str) -> str:
        now = int(time.time())
        window = self._current_window()
        expires_at = now + self.ttl_seconds
        tag = self._compute_tag(window, expires_at)
        header = f"{self.tag_prefix} w={window} exp={expires_at} tag={tag}>\n"
        return header + text

    def _obfuscate_text(self, text: str) -> str:
        if not text:
            return text

        chars = []
        for ch in text:
            # Occasionally inject zero-width char before spaces/punctuation
            if ch.isspace() and random.random() < self.zero_width_probability:
                chars.append(random.choice(INVISIBLE_CHARS))

            # Replace with homoglyph with small probability when available
            if ch in HOMOGLYPHS and random.random() < self.homoglyph_probability:
                chars.append(random.choice(HOMOGLYPHS[ch]))
            else:
                chars.append(ch)

            # Occasionally inject zero-width char after alphanumerics
            if ch.isalnum() and random.random() < self.zero_width_probability:
                chars.append(random.choice(INVISIBLE_CHARS))

        return "".join(chars)

    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        if prompt is None or prompt == "":
            return prompt, True, -1.0

        try:
            obfuscated = self._obfuscate_text(prompt)
            with_header = self._embed_header(obfuscated)
            return with_header, True, -1.0
        except Exception as exc:
            LOGGER.error("TimeLockObfuscator error", error=str(exc))
            return prompt, True, -1.0

