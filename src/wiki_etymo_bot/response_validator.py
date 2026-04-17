import re


class ResponseValidator:
    def __init__(self, negative_patterns: set[str]):
        self._negative_patterns = set(map(re.compile, negative_patterns))

    def __call__(self, response: str):
        for negative_pattern in self._negative_patterns:
            if negative_pattern.search(response):
                return False
        return True
