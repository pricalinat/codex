from .api import OrchestratorService, create_app


service = OrchestratorService(max_rounds=100)
service.register_builtin_agents()
app = create_app(service)


if __name__ == "__main__":
    if app is None:
        raise SystemExit("FastAPI is not installed. Install requirements to run the API server.")

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
