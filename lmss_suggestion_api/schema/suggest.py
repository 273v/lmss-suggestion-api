"""lmss_suggestion_api.schemas.core provides pydantic schemas for basic API endpoint info."""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# pylint: disable=E0611,R0903

# package imports
from pydantic import BaseModel


class Suggestion(BaseModel):
    """
    Suggestion request
    """

    iri: str
    label: str
    alt_labels: list[str]
    hidden_labels: list[str]
    definitions: list[str]
    parents: list[str]
    children: list[str]
    exact: bool
    substring: bool
    distance: float
    score: float
    source: str
    startswith: bool
    match: str


class SuggestResponse(BaseModel):
    """
    Suggestion response
    """

    suggestions: list[Suggestion]
    response_time: float = 0.0
    error: str | None = None
