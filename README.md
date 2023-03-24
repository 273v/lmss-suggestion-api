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

## License

This project is licensed under the terms of the [MIT license](/LICENSE).

----
