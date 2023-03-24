# SALI LMSS Suggestion API

**Description**: A RESTful API for searching and tagging text with the [SALI Legal Matter Standard Specification (LMSS) ontology](https://sali.org/).

---- 
  * **Project Name:** SALI LMSS Suggestion API (lmss-suggestion-api)
 * **Project Authors:** Michael Bommarito ([273 Ventures](https://273ventures.com)), Damien Riehl ([SALI](https://sali.org/))
 * **Support:** [info@sali.org](mailto:info@sali.org)

----
## Distributions
* [Source](https://github.com/273v/lmss-suggestion-api)
* [Docker Hub](https://hub.docker.com/r/273ventures/lmss-suggestion-api)
----

### Run from Source
* Prerequisites:
  * Python 3.10+
  * [Poetry](https://python-poetry.org/)
* Clone the repository
* Install environment with poetry
  * `$ poetry install`
* Configure the API, including OpenAI credentials
  * `$ cp .env.json.template .env.json`
  * Edit to set values - typically `OPENAI_API_KEY`
* Run the API
  * `PYTHONPATH=. poetry run python lmss_suggestion_api/api.py`

### Run from Docker
* Prerequisites:
  * Docker
* Pull the image
  * `$ docker pull 273ventures/lmss-suggestion-api`
* Create .docker.env file with contents:
  * ```OPENAI_API_KEY=...```
* Run the image (on http tcp/8888 by default)
```
$ docker run --publish 8888:8888 \
    --name lmss-suggestion-api \
    --env-file .docker.env \
    273ventures/lmss-suggestion-api:latest
```

----

## API Documentation

  * Swagger/OpenAPI docs: You can find the Swagger page at `/docs`.
  * Redoc docs: Additional documentation will be available at `/redoc` in the future.

----

## Telemetry

We want to make our software more useful and reliable. In order to do that, we're using Telly, a privacy-friendly usage statistics and support tool. Telly may collect basic information about your Python and OS environment, allowing us to understand things like which operating systems and Python versions to support. In some cases, Telly may also collect SHA-256-hashed representations of information that allows us to estimate our unique number of users. These hashes are only used temporarily to assign a temporary ID for aggregate reporting; no plaintext information is ever transmitted or visible to humans and even hashes are only stored during a temporary, rolling analytics window.

You can learn more about Telly's privacy functionality here: https://gotelly.io/.

To disable Telly's data collection, you can either:
* set the TELLY_DISABLE environment variable to a truthy value like Y or 1
* create a file named .telly_disable in your home directory
 
----

## License

This project is licensed under the terms of the [MIT license](/LICENSE).

----
