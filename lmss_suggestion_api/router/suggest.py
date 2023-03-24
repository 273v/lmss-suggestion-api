"""suggest api routes"""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# imports
import time

# package imports
import telly
import sys
from fastapi import APIRouter, Request


from lmss_suggestion_api.schema.suggest import (
    SuggestResponse,
    Suggestion,
    ConceptSuggestionResponse,
    ConceptSuggestion,
)
from lmss_suggestion_api.suggester_engine import SuggestionType
from lmss_suggestion_api.api_logger import get_logger
from lmss_suggestion_api.api_settings import VERSION_001

LOGGER = get_logger(__name__)

# implement health check API endpoint with a version
router = {
    VERSION_001: APIRouter(
        prefix=f"/{VERSION_001}/suggest",
        tags=["suggest", VERSION_001],
        responses={404: {"description": "Unknown request path"}},
    ),
    "default": APIRouter(
        prefix="/suggest",
        tags=["suggest"],
        responses={404: {"description": "Unknown request path"}},
    ),
}


async def suggest_concept_results(
    request: Request, concept_type: SuggestionType, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    request.app.state.request_stats[f"suggest/{concept_type.value}"] += 1
    cache_key = f"{concept_type.value}/{text}/{num_results}"

    # get timer
    start_time = time.time()

    # check the cache
    if cache_key in request.app.state.suggest_cache:
        LOGGER.info(
            "cache hit: key=%s, value=%s",
            cache_key,
            request.app.state.suggest_cache[cache_key],
        )
        return SuggestResponse(
            suggestions=[
                Suggestion(**s) for s in request.app.state.suggest_cache[cache_key]
            ],
            response_time=time.time() - start_time,
        )

    # run the suggestion engine
    try:
        suggestions = await request.app.state.suggester_engine.suggest(
            text, concept_type, max_results=num_results
        )
        LOGGER.info("cache miss: key=%s, value=%s", cache_key, suggestions)

        request.app.state.suggest_cache[cache_key] = suggestions
        return SuggestResponse(
            suggestions=[Suggestion(**s) for s in suggestions],
            response_time=time.time() - start_time,
        )
    except Exception as error:  # pylint: disable=W0718
        LOGGER.error("Unable to generate suggestions: %s", error)
        # output stack trace
        LOGGER.exception(error)
        return SuggestResponse(
            suggestions=[],
            response_time=time.time() - start_time,
            error="Unable to generate suggestions",
        )


@router["default"].get("/concepts", response_model=ConceptSuggestionResponse)
@router[VERSION_001].get("/concepts", response_model=ConceptSuggestionResponse)
async def method_suggest_concepts(
    request: Request, text: str
) -> ConceptSuggestionResponse:
    """Run the suggestion engine meta-query to determine which concepts to query"""
    # get the status
    request.app.state.request_stats["suggest/concepts"] += 1
    cache_key = f"concepts/{text}"

    # get timer
    start_time = time.time()

    # check the cache
    if cache_key in request.app.state.suggest_cache:
        LOGGER.info(
            "cache hit: key=%s, value=%s",
            cache_key,
            request.app.state.suggest_cache[cache_key],
        )
        return ConceptSuggestionResponse(
            suggestions=[
                ConceptSuggestion(**s)
                for s in request.app.state.suggest_cache[cache_key]
            ],
            response_time=time.time() - start_time,
        )

    # run the suggestion engine
    try:
        concepts = await request.app.state.suggester_engine.suggest_concepts(query=text)
        LOGGER.info("cache miss: key=%s, value=%s", cache_key, concepts)

        request.app.state.suggest_cache[cache_key] = concepts
        return ConceptSuggestionResponse(
            suggestions=[ConceptSuggestion(**c) for c in concepts],
            response_time=time.time() - start_time,
        )
    except Exception as error:  # pylint: disable=W0718
        LOGGER.error("Unable to generate suggestions: %s", error)
        # output stack trace
        LOGGER.exception(error)
        return ConceptSuggestionResponse(
            suggestions=[],
            response_time=time.time() - start_time,
            error="Unable to generate suggestions",
        )


@router["default"].get("/actor-player", response_model=SuggestResponse)
@router[VERSION_001].get("/actor-player", response_model=SuggestResponse)
async def method_suggest_actor_player(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.ACTOR_PLAYER, text, num_results
    )


@router["default"].get("/area-of-law", response_model=SuggestResponse)
@router[VERSION_001].get("/area-of-law", response_model=SuggestResponse)
async def method_suggest_area_of_law(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.AREA_OF_LAW, text, num_results
    )


@router["default"].get("/asset-type", response_model=SuggestResponse)
@router[VERSION_001].get("/asset-type", response_model=SuggestResponse)
async def method_suggest_asset_type(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.ASSET_TYPE, text, num_results
    )


@router["default"].get("/communication-modality", response_model=SuggestResponse)
@router[VERSION_001].get("/communication-modality", response_model=SuggestResponse)
async def method_suggest_communication_modality(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.COMMUNICATION_MODALITY, text, num_results
    )


@router["default"].get("/currency", response_model=SuggestResponse)
@router[VERSION_001].get("/currency", response_model=SuggestResponse)
async def method_suggest_currency(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.CURRENCY, text, num_results
    )


@router["default"].get("/data-format", response_model=SuggestResponse)
@router[VERSION_001].get("/data-format", response_model=SuggestResponse)
async def method_suggest_data_format(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.DATA_FORMAT, text, num_results
    )


@router["default"].get("/document-artifact", response_model=SuggestResponse)
@router[VERSION_001].get("/document-artifact", response_model=SuggestResponse)
async def method_suggest_document_artifact(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.DOCUMENT_ARTIFACT, text, num_results
    )


@router["default"].get("/engagement-terms", response_model=SuggestResponse)
@router[VERSION_001].get("/engagement-terms", response_model=SuggestResponse)
async def method_suggest_engagement_terms(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.ENGAGEMENT_TERMS, text, num_results
    )


@router["default"].get("/event", response_model=SuggestResponse)
@router[VERSION_001].get("/event", response_model=SuggestResponse)
async def method_suggest_event(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.EVENT, text, num_results
    )


@router["default"].get("/forum-and-venue", response_model=SuggestResponse)
@router[VERSION_001].get("/forum-and-venue", response_model=SuggestResponse)
async def method_suggest_forum_and_venue(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.FORUM_AND_VENUE, text, num_results
    )


@router["default"].get("/governmental-body", response_model=SuggestResponse)
@router[VERSION_001].get("/governmental-body", response_model=SuggestResponse)
async def method_suggest_governmental_body(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.GOVERNMENTAL_BODY, text, num_results
    )


@router["default"].get("/industry", response_model=SuggestResponse)
@router[VERSION_001].get("/industry", response_model=SuggestResponse)
async def method_suggest_industry(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.INDUSTRY, text, num_results
    )


@router["default"].get("/lmss-type", response_model=SuggestResponse)
@router[VERSION_001].get("/lmss-type", response_model=SuggestResponse)
async def method_suggest_lmss_type(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.LMSS_TYPE, text, num_results
    )


@router["default"].get("/legal-authorities", response_model=SuggestResponse)
@router[VERSION_001].get("/legal-authorities", response_model=SuggestResponse)
async def method_suggest_legal_authorities(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.LEGAL_AUTHORITIES, text, num_results
    )


@router["default"].get("/legal-entity", response_model=SuggestResponse)
@router[VERSION_001].get("/legal-entity", response_model=SuggestResponse)
async def method_suggest_legal_entity(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.LEGAL_ENTITY, text, num_results
    )


@router["default"].get("/location", response_model=SuggestResponse)
@router[VERSION_001].get("/location", response_model=SuggestResponse)
async def method_suggest_location(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.LOCATION, text, num_results
    )


@router["default"].get("/matter-narrative", response_model=SuggestResponse)
@router[VERSION_001].get("/matter-narrative", response_model=SuggestResponse)
async def method_suggest_matter_narrative(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.MATTER_NARRATIVE, text, num_results
    )


@router["default"].get("/matter-narrative-format", response_model=SuggestResponse)
@router[VERSION_001].get("/matter-narrative-format", response_model=SuggestResponse)
async def method_suggest_matter_narrative_format(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.MATTER_NARRATIVE_FORMAT, text, num_results
    )


@router["default"].get("/objective", response_model=SuggestResponse)
@router[VERSION_001].get("/objective", response_model=SuggestResponse)
async def method_suggest_objective(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.OBJECTIVE, text, num_results
    )


@router["default"].get("/service", response_model=SuggestResponse)
@router[VERSION_001].get("/service", response_model=SuggestResponse)
async def method_suggest_service(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.SERVICE, text, num_results
    )


@router["default"].get("/standards-compatibility", response_model=SuggestResponse)
@router[VERSION_001].get("/standards-compatibility", response_model=SuggestResponse)
async def method_suggest_standards_compatibility(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.STANDARDS_COMPATIBILITY, text, num_results
    )


@router["default"].get("/status", response_model=SuggestResponse)
@router[VERSION_001].get("/status", response_model=SuggestResponse)
async def method_suggest_status(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.STATUS, text, num_results
    )


@router["default"].get("/system-identifiers", response_model=SuggestResponse)
@router[VERSION_001].get("/system-identifiers", response_model=SuggestResponse)
async def method_suggest_system_identifiers(
    request: Request, text: str, num_results: int = 10
) -> SuggestResponse:
    """Run the suggestion engine on the input text"""
    # get the status
    return await suggest_concept_results(
        request, SuggestionType.SYSTEM_IDENTIFIERS, text, num_results
    )

telly.go(sys.modules[__name__], '26d17932-7a3d-4e12-9beb-f48993941f30')
