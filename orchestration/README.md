This folder contains example orchestration workflows used to connect the
services in this repository.  The workflows are designed for the
[n8n](https://n8n.io/) automation tool, but the concepts are applicable
to any orchestration engine.

The `example_workflow.json` file demonstrates a simple end‑to‑end flow:

1. Trigger: a scheduler fires daily at a configured time.
2. Fetch demand history: call the backend `/forecasts/{sku_id}` endpoint to obtain a forecast for a specific SKU.
3. Compute procurement: call the `/procure/recommendations` endpoint with the forecast.
4. Explain: optionally call the LLM service to produce a human‑friendly explanation.
5. Notify: send the recommendation and explanation to a human approver via email or chat.
6. Await response: pause until the human approves or rejects the recommendation.
7. Store: record the decision in a database or data warehouse.

You can import the JSON file into n8n to see the structure and customise it for your use case.