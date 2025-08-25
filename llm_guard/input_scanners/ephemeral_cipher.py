import time
import hmac
import hashlib
import random
import string
from typing import Dict, Tuple

from llm_guard.util import get_logger

from .base import Scanner


LOGGER = get_logger()


def _derive_seed(secret: bytes, window: int) -> int:
    msg = f"{window}".encode("utf-8")
    digest = hmac.new(secret, msg, hashlib.sha256).digest()
    # Use first 8 bytes as integer seed
    return int.from_bytes(digest[:8], "big", signed=False)


class EphemeralSubstitutionObfuscator(Scanner):
    """
    Time-rotating monoalphabetic substitution for human/LLM-decodable encryption.

    - Builds a character-level substitution map per time window using HMAC(secret, window)
    - Encodes the prompt and embeds minimal instructions + mapping for the LLM to decode
    - Includes TTL metadata in header

    Security note: this is NOT cryptographic secrecy. It resists trivial keyword
    matching and enables ephemeral keys, but the mapping is present so the model
    (and provider) can decode.
    """

    def __init__(
        self,
        *,
        secret: str | None = None,
        time_step_seconds: int = 300,
        ttl_seconds: int = 300,
        alphabet: str | None = None,
        include_header: bool = True,
    ) -> None:
        self.secret = (secret or "changeme_ephemeral_secret").encode("utf-8")
        self.time_step_seconds = max(1, int(time_step_seconds))
        self.ttl_seconds = max(1, int(ttl_seconds))
        # Default alphabet: visible ASCII plus basic punctuation likely safe for LLMs
        self.alphabet = (
            alphabet
            or (string.ascii_letters + string.digits + string.punctuation + " ")
        )
        # Remove backticks to avoid breaking Markdown code fences unexpectedly
        self.alphabet = self.alphabet.replace("`", "")
        self.include_header = include_header

    def _current_window(self) -> int:
        return int(time.time() // self.time_step_seconds)

    def _build_mapping(self, window: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        seed = _derive_seed(self.secret, window)
        rng = random.Random(seed)
        src = list(self.alphabet)
        dst = src.copy()
        rng.shuffle(dst)
        fwd = {s: d for s, d in zip(src, dst)}
        rev = {d: s for s, d in zip(src, dst)}
        return fwd, rev

    def _encode(self, text: str, fwd: Dict[str, str]) -> str:
        return "".join(fwd.get(ch, ch) for ch in text)

    def _render_instructions(self, window: int, expires_at: int, rev: Dict[str, str], ciphertext: str) -> str:
        # Compact instruction block to keep tokens low and decoding reliable
        lines = []
        if self.include_header:
            lines.append(f"<epc:v1 w={window} exp={expires_at}>")
        lines.append(
            "Assistant: Decode the ciphertext by replacing characters using the table (cipher->plain). "
            "Apply only to listed characters; keep others as-is. Then answer the decoded prompt and do not mention decoding."
        )
        lines.append("Mapping (cipher -> plain):")
        # Emit mapping in compact chunks
        chunk = []
        count = 0
        for ciph, plain in rev.items():
            pair = f"{ciph}:{plain}"
            chunk.append(pair)
            count += 1
            if count % 40 == 0:
                lines.append(", ".join(chunk))
                chunk = []
        if chunk:
            lines.append(", ".join(chunk))
        lines.append("Ciphertext:")
        lines.append(ciphertext)
        return "\n".join(lines)

    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        if prompt is None or prompt == "":
            return prompt, True, -1.0

        try:
            window = self._current_window()
            expires_at = int(time.time()) + self.ttl_seconds
            fwd, rev = self._build_mapping(window)
            ciphertext = self._encode(prompt, fwd)
            sanitized = self._render_instructions(window, expires_at, rev, ciphertext)
            return sanitized, True, -1.0
        except Exception as exc:
            LOGGER.error("EphemeralSubstitutionObfuscator error", error=str(exc))
            return prompt, True, -1.0

