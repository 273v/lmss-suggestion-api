"""This module loads settings from .env or defaults"""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT


# imports
import json
import os
from pathlib import Path

# load .env as json
try:
    DOTENV_PATH = Path(__file__).parent.parent / ".env.json"
    if DOTENV_PATH.exists():
        with DOTENV_PATH.open("rt", encoding="utf-8") as f:
            ENV = json.load(f)
    else:
        ENV = {}

    # ensure that DEBUG is set to boolean
    if "DEBUG" in ENV:
        if ENV["DEBUG"] in [1, "1", "t", "T", "true", "True", "TRUE", "y", "Y", "yes", "Yes", "YES"]:
            ENV["DEBUG"] = True
        else:
            ENV["DEBUG"] = False
except Exception as error:
    # pylint: disable=W0707
    raise RuntimeError(f"Failed to load .env.json: {error}", error)

# set defaults for missing keys
if "OPENAI_API_KEY" not in ENV or ENV["OPENAI_API_KEY"] in [None, ""]:
    if "OPENAI_API_KEY" in os.environ:
        ENV["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

# check if it's an empty string
if ENV.get("OPENAI_API_KEY", None) is None:
    raise RuntimeWarning(
        "OPENAI_API_KEY is empty in .env or available as environment variable; LLM Suggestion API will not work."
    )

# setup API config
API_NAME = "SALI LMSS Suggester API"
API_VERSION = "0.0.1"
API_DESCRIPTION = "Suggester API for SALI LMSS"
API_LOGO_URL = "https://www.sali.org/resources/assets/svg/SALIalliance_tag.svg"

VERSION_001 = "v0.0.1"

VERSIONS_OFFERED = [
    VERSION_001,
]


if __name__ == "__main__":
    # imports
    import argparse

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # output keys and values
    if args.json:
        # output as JSON
        print(json.dumps(ENV, indent=4))
    else:
        # output keys and values for debugging as k=v
        for k, v in ENV.items():
            print(f"{k}={v}")
