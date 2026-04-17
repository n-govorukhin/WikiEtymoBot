from pydantic import BaseModel


class Word(BaseModel):
    text: str
    language_code: str


class SearchPlannerOutput(BaseModel):
    words: list[Word]


class LanguageRecognizerOutput(BaseModel):
    language: str


class ArticleSelectorOutput(BaseModel):
    article_id: int


class LinkClassifierOutput(BaseModel):
    answer: int
