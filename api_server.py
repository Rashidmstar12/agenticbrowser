from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from browser_agent import BrowserAgent
from task_planner import STEP_SCHEMA, summarize_results, validate_task

app = FastAPI(title="agenticbrowser API")


class StepRequest(BaseModel):
    action: str
    params: dict[str, Any] = {}


class TaskRequest(BaseModel):
    name: str
    steps: list[StepRequest]


class TaskRunResponse(BaseModel):
    name: str
    success: bool
    results: list[dict]
    summary: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"actions": STEP_SCHEMA}


@app.post("/validate")
def validate_endpoint(task: TaskRequest):
    errors = validate_task(task.model_dump())
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return {"valid": True}


@app.post("/task/run", response_model=TaskRunResponse)
async def task_run(task: TaskRequest):
    agent = BrowserAgent(headless=True)
    try:
        await agent.start()
        steps = [{"action": s.action, **s.params} for s in task.steps]
        results = await agent.run_task(steps)
        if any(not r.get("success") for r in results):
            raise HTTPException(status_code=422, detail="One or more steps failed")
        summary = summarize_results(results)
        return TaskRunResponse(name=task.name, success=True, results=results, summary=summary)
    finally:
        await agent.stop()


@app.get("/tasks/schema")
def tasks_schema():
    return {"step_actions": STEP_SCHEMA, "count": len(STEP_SCHEMA)}
