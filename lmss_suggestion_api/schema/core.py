"""lmss_suggestion_api.schema.core provides pydantic schema for basic API endpoint info."""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# pylint: disable=R0903

# package imports
from pydantic import BaseModel  # pylint: disable=E0611


class HealthStatusResponse(BaseModel):
    """
    Health status response model
    """

    status: str
    error: str | None = None


class VersionResponse(BaseModel):
    """
    Version response model
    """

    version: str
    error: str | None = None


class VersionListResponse(BaseModel):
    """
    Version response model
    """

    version_list: list[str]
    error: str | None = None


class StatsResponse(BaseModel):
    """
    API Statistics response model
    """

    # object statistics
    objects: dict

    # request statistics
    requests: dict

    error: str | None = None
