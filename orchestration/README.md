This folder contains example orchestration workflows for connecting the backend API to a daily agent flow using **n8n**.

### Import
1. Launch n8n (Compose brings it up at http://localhost:5678).
2. In n8n → **Workflows** → **Import from file** → select `n8n_workflows/example_workflow.json`.

### Configure
- Set env `API_URL` inside n8n (or edit HTTP Request node URLs):
  - Compose: `http://backend:8000/api/v1`
  - Local dev: `http://localhost:8000/api/v1`
  - Render: `https://<your-api>.onrender.com/api/v1`
- Change `sku_id` in the first Set node.

### Flow
1. Schedule (daily 06:00)
2. Forecast → `GET /forecasts/{sku_id}`
3. Procure → `POST /procure/recommendations`
4. Explain (optional)
5. Notify approver
6. Await approval
7. Store audit record
