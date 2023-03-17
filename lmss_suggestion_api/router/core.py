"""lmss_suggestion_api.routers.core provides basic API endpoints for the application related to
health checks and versioning."""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT


# package imports
from fastapi import APIRouter, Request

# pydantic models
from lmss_suggestion_api.schema.core import (
    HealthStatusResponse,
    VersionResponse,
    VersionListResponse,
    StatsResponse,
)
from lmss_suggestion_api.api_settings import (
    API_VERSION,
    VERSION_001,
    VERSIONS_OFFERED,
)

# implement health check API endpoint with a version
router = {
    VERSION_001: APIRouter(
        prefix=f"/{VERSION_001}/core",
        tags=["core", VERSION_001],
        responses={404: {"description": "Unknown request path"}},
    ),
    "default": APIRouter(
        prefix="/core",
        tags=["core"],
        responses={404: {"description": "Unknown request path"}},
    ),
}


@router["default"].get("/status", response_model=HealthStatusResponse)
@router[VERSION_001].get("/status", response_model=HealthStatusResponse)
async def method_health_check(request: Request) -> HealthStatusResponse:
    """Health check API endpoint."""
    # get the status
    request.app.state.request_stats["core/status"] += 1
    return HealthStatusResponse(status="up")


@router["default"].get("/version", response_model=VersionResponse)
@router[VERSION_001].get("/version", response_model=VersionResponse)
async def method_version_check(request: Request) -> VersionResponse:
    """Version check API endpoint."""
    # return the response
    request.app.state.request_stats["core/method_version_check"] += 1
    return VersionResponse(version=API_VERSION)


@router["default"].get("/versions-offered", response_model=VersionListResponse)
@router[VERSION_001].get(
    "/versions-offered",
    response_model=VersionListResponse,
)
async def method_versions_offered(request: Request) -> VersionListResponse:
    """Versions offered API endpoint."""
    # return the response
    request.app.state.request_stats["core/method_versions_offered"] += 1
    return VersionListResponse(version_list=VERSIONS_OFFERED)


# return statistics about the API
@router["default"].get("/stats", response_model=StatsResponse)
@router[VERSION_001].get("/stats", response_model=StatsResponse)
async def method_stats(
    request: Request,
) -> StatsResponse:
    """API statistics API endpoint."""
    # get the statistics
    request.app.state.request_stats["core/method_stats"] += 1
    try:
        # get the request count dictionary
        request_stats = request.app.state.request_stats

        # return the response
        return StatsResponse(
            objects={},
            requests=request_stats,
            error=None,
        )
    except Exception as error:  # pylint: disable=W0718
        return StatsResponse(
            objects={},
            requests={},
            error=str(error),
        )
