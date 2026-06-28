import re


THINK_TAG_MAP = {
    "exaone-deep": ("<thought>", "</thought>"),
}


def get_think_tags(model_name):
    name_lower = model_name.lower()
    for key, tags in THINK_TAG_MAP.items():
        if key in name_lower:
            return tags
    return None


def normalize_think_tags(text, model_name):
    tags = get_think_tags(model_name)
    if tags is None:
        return text
    open_tag, close_tag = tags
    return text.replace(open_tag, "<think>").replace(close_tag, "</think>")


def adapt_think_tags_for_model(text, model_name):
    tags = get_think_tags(model_name)
    if tags is None:
        return text
    open_tag, close_tag = tags
    return text.replace("<think>", open_tag).replace("</think>", close_tag)


class PhaseDetector:

    THINKING = "thinking"
    ANSWER = "answer"

    _OPEN_PATTERN = re.compile(r"<think>", re.IGNORECASE)
    _CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)

    def detect(self, text):
        if not text:
            return self.ANSWER
        n_open = len(self._OPEN_PATTERN.findall(text))
        n_close = len(self._CLOSE_PATTERN.findall(text))
        if n_open > 0 and n_open > n_close:
            return self.THINKING
        return self.ANSWER
