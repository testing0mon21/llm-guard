import time
import base64
import hashlib
from typing import Tuple

from .base import Scanner


class TimeBasedEncryption(Scanner):
    """Simple time-bucket based prompt obfuscator.

    The prompt is XOR-encrypted with a key derived from the current time
    bucket.  The bucket width (``ttl``) controls for how long the same key
    remains valid.  After the bucket expires the ciphertext cannot be
    decrypted with a newly derived key which effectively limits the time
    window in which the data can be recovered.

    NOTE: This implementation focuses on integration with the LLM-Guard
    scanner pipeline and **is NOT meant for production-grade security**.  It
    purposefully avoids external dependencies (e.g. ``cryptography``) to
    keep the core package lightweight.  If strong cryptographic guarantees
    are required you should replace the XOR logic with a well-vetted
    algorithm such as AES-GCM/Fernet.
    """

    def __init__(self, ttl: int = 300):
        """Create a new scanner.

        Parameters
        ----------
        ttl : int, optional
            Time-to-live for a key in **seconds**.  Prompts encrypted in the
            same *time bucket* (``floor(time() / ttl)``) use the same key.
        """
        self._ttl = max(1, int(ttl))

    # ------------------------------------------------------------------
    # Public helpers (can be used by downstream services)
    # ------------------------------------------------------------------

    def decrypt(self, ciphertext_b64: str) -> str:
        """Attempt to decrypt *ciphertext_b64*.

        The method derives the key for **all** buckets that are still valid
        (current and previous) in order to tolerate clock skew of at most
        one bucket.
        """
        data = base64.urlsafe_b64decode(ciphertext_b64.encode())
        # Try current bucket first, then previous one to mitigate small skew
        for bucket in (self._current_bucket(), self._current_bucket() - 1):
            key = self._derive_key(bucket, len(data))
            plain_bytes = bytes(b ^ k for b, k in zip(data, key))
            try:
                return plain_bytes.decode()
            except UnicodeDecodeError:
                continue
        # If both attempts fail, raise error
        raise ValueError("Unable to decrypt ciphertext with available keys")

    # ------------------------------------------------------------------
    # Scanner interface implementation
    # ------------------------------------------------------------------

    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        """Encrypt *prompt* and return a self-contained instruction block.

        The returned *sanitized_prompt* contains:
          • The base64 ciphertext
          • Natural-language instructions describing how to derive the key and
            decrypt the ciphertext.  These instructions are intended to be
            sent verbatim to the downstream LLM so that it can recover the
            original prompt *only* during the validity window defined by
            ``ttl``.
        """
        bucket = self._current_bucket()
        key = self._derive_key(bucket, len(prompt.encode()))
        cipher_bytes = bytes(b ^ k for b, k in zip(prompt.encode(), key))
        cipher_b64 = base64.urlsafe_b64encode(cipher_bytes).decode()

        instructions = (
            "You will receive an encrypted prompt that must be decrypted **inside the model** "
            "before any further reasoning.  The ciphertext was produced using an XOR with a key "
            "derived from the current *time bucket* (UNIX epoch seconds divided by {ttl}).\n\n"
            "Decryption procedure:\n"
            "1. Let `bucket = floor(current_time / {ttl})`.\n"
            "2. Compute `digest = SHA-256( str(bucket) )` (32-byte array).\n"
            "3. Repeat `digest` until its length matches the ciphertext length to obtain `key`.\n"
            "4. XOR each ciphertext byte with the corresponding key byte.\n"
            "5. Interpret the result as UTF-8 to recover the original prompt.\n"
            "If UTF-8 decoding fails, retry with `bucket-1` to tolerate small clock skew.\n"
        ).format(ttl=self._ttl)

        wrapped_prompt = (
            "## TIME-LOCKED PROMPT ##\n"
            f"TTL_SECONDS={self._ttl}\n"
            f"CIPHERTEXT_B64={cipher_b64}\n\n"
            "### DECRYPTION INSTRUCTIONS ###\n" + instructions + "\n"
            "### END ###"
        )

        return wrapped_prompt, True, 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_bucket(self) -> int:
        return int(time.time() // self._ttl)

    @staticmethod
    def _derive_key(bucket: int, length: int) -> bytes:
        # Derive a pseudo-random key from the bucket identifier
        digest = hashlib.sha256(str(bucket).encode()).digest()
        # Repeat the digest as necessary to match desired length
        repeats = (length + len(digest) - 1) // len(digest)
        return (digest * repeats)[:length]