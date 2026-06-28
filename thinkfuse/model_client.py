import abc
import hashlib
import math

import numpy as np


class ModelClient(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, text):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, ids, skip_special=True):
        raise NotImplementedError

    @abc.abstractmethod
    def batch_decode(self, id_lists):
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, context_ids, n_tokens, top_k):
        raise NotImplementedError

    @abc.abstractmethod
    def perplexity(self, context_ids, segment_text):
        raise NotImplementedError


def _seeded_unit(*parts):
    digest = hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


_AUX_PHRASES = [
    "wait, let me reconsider the previous step ",
    "alternatively we can rewrite the equation ",
    "double-checking the arithmetic here gives ",
    "another way to see this is to factor out ",
    "so the intermediate result should equal ",
]


class MockModelClient(ModelClient):

    def __init__(self, name, script=None, top_k=5):
        self._name = name
        self._script = script
        self._top_k = top_k
        self._prompt_len = None

    @property
    def name(self):
        return self._name

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special=True):
        return "".join(chr(i) for i in ids if i >= 0)

    def batch_decode(self, id_lists):
        return [self.decode(ids) for ids in id_lists]

    def _logprobs_for(self, char, position):
        conf = 0.35 + 0.6 * _seeded_unit(self._name, char, position)
        rest = (1.0 - conf) / (self._top_k - 1)
        probs = [conf] + [rest] * (self._top_k - 1)
        return {f"<{i}>": math.log(max(p, 1e-12)) for i, p in enumerate(probs)}

    def _generate_from_script(self, context_ids, n_tokens):
        if self._prompt_len is None:
            self._prompt_len = len(context_ids)
        cursor = max(len(context_ids) - self._prompt_len, 0)
        chunk = self._script[cursor:cursor + n_tokens]
        eos = cursor + n_tokens >= len(self._script)
        new_ids = self.encode(chunk)
        logprobs = [self._logprobs_for(c, cursor + i) for i, c in enumerate(chunk)]
        return new_ids, eos, logprobs

    def _generate_alternative(self, context_ids, n_tokens):
        seed = _seeded_unit(self._name, len(context_ids))
        phrase = _AUX_PHRASES[int(seed * len(_AUX_PHRASES)) % len(_AUX_PHRASES)]
        offset = int(_seeded_unit("offset", self._name, len(context_ids)) * len(phrase))
        chunk = (phrase * 2)[offset:offset + n_tokens]
        new_ids = self.encode(chunk)
        logprobs = [self._logprobs_for(c, len(context_ids) + i) for i, c in enumerate(chunk)]
        return new_ids, False, logprobs

    def generate(self, context_ids, n_tokens, top_k):
        if self._script is not None:
            return self._generate_from_script(context_ids, n_tokens)
        return self._generate_alternative(context_ids, n_tokens)

    def perplexity(self, context_ids, segment_text):
        if not segment_text:
            return float("inf")
        jitter = 0.5 * _seeded_unit(self._name, segment_text)
        if self._script is not None and self._prompt_len is not None:
            cursor = max(len(context_ids) - self._prompt_len, 0)
            expected = self._script[cursor:cursor + len(segment_text)]
            compatible = 1.0 if segment_text == expected else 0.0
            return 1.5 + 6.0 * (1.0 - compatible) + jitter
        tail = self.decode(context_ids[-16:]) if context_ids else ""
        return 1.0 + 8.0 * _seeded_unit(self._name, tail, segment_text) + jitter
