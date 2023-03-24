"""SALI LMSS Suggester API."""
# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# imports
import asyncio
import collections
import time


# fastapi and hypercorn imports
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware
from hypercorn.config import Config
from hypercorn.asyncio import serve

# project imports
from lmss_suggestion_api.api_logger import LOGGER
from lmss_suggestion_api.router import core, suggest
from lmss_suggestion_api.api_settings import (
    ENV,
    API_NAME,
    API_DESCRIPTION,
    API_VERSION,
    API_LOGO_URL,
)
from lmss_suggestion_api.suggester_engine import SuggesterEngine


def app_factory():
    """create an app and return the instance"""
    # setup FastAPI handler with OpenAPI swagger docs
    app_instance = FastAPI(
        title=API_NAME,
        description=API_DESCRIPTION,
        version=API_VERSION,
        contact={
            "name": "273 Ventures, LLC",
            "url": "https://273ventures.com",
            "email": "hello@273ventures.com",
        },
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app_instance.add_middleware(GZipMiddleware, minimum_size=1024)
    # NOTE: decide on something like GraphQL here

    # stub config for debug
    config_instance = Config()
    config_instance.bind = [
        f"{ENV['API_HOST']}:{ENV['API_PORT']}",
    ]
    config_instance.loglevel = "debug" if ENV["DEBUG"] else "info"

    # register routers
    for router in core.router.values():
        app_instance.include_router(router)

    for router in suggest.router.values():
        app_instance.include_router(router)

    if app_instance.openapi_schema:
        return app_instance.openapi_schema

    # create if not already
    openapi_schema = get_openapi(
        title=API_NAME,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app_instance.routes,
    )

    # set the logo for fun
    openapi_schema["info"]["x-logo"] = {"url": API_LOGO_URL}

    # set the x-summary values
    for _, path_item in openapi_schema["paths"].items():
        for _, method_item in path_item.items():
            method_item["x-summary"] = " ".join(
                method_item["summary"].split()[1:]
            ).title()
            method_item["summary"] = method_item["x-summary"]

    # set openapi
    app_instance.openapi_schema = openapi_schema  # type: ignore

    # setup CORS middleware open to * for API use
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # return the app
    return app_instance, config_instance


# create the config and app
app, config = app_factory()


# setup pdf engine on app.state on setup
@app.on_event("startup")
async def startup_event():
    """initialize the pdf engine pool and cache"""
    # log timing
    start_time = time.time()

    # setup the response tracking dictionary
    app.state.request_stats = collections.defaultdict(int)
    app.state.suggest_cache = {}
    app.state.suggester_engine = SuggesterEngine()

    LOGGER.info("Branch: %s", ENV["LMSS_BRANCH"])

    # check if openai key is set and print model name if so, else disabled
    if ENV["OPENAI_API_KEY"]:
        LOGGER.info("OpenAI API: enabled with %s", ENV["OPENAI_MODEL_NAME"])
    else:
        LOGGER.info("OpenAI API: disabled")

    # stop timer
    end_time = time.time()
    LOGGER.info("Startup completed in %f seconds", end_time - start_time)


# run the main app
if __name__ == "__main__":
    # runtime imports
    LOGGER.info("starting API endpoint at %s with hypercorn", config.bind)
    # log ENV config
    asyncio.run(serve(app, config))  # type: ignore

