import asyncio
import json

import typer

from browser_agent import BrowserAgent
from task_planner import summarize_results, validate_task

app = typer.Typer(help="agenticbrowser CLI")


@app.command()
def run(
    task_file: str = typer.Argument(..., help="Path to JSON task file"),
    headless: bool = typer.Option(True, help="Run browser in headless mode"),
    output: str | None = typer.Option(None, help="Output results to JSON file"),
):
    """Run a task from a JSON file."""
    with open(task_file) as f:
        task_data = json.load(f)

    errors = validate_task(task_data)
    if errors:
        typer.echo(f"Task validation failed: {errors}", err=True)
        raise typer.Exit(1)

    async def _run():
        agent = BrowserAgent(headless=headless)
        await agent.start()
        try:
            steps = []
            for s in task_data["steps"]:
                step = {"action": s["action"]}
                step.update(s.get("params", {}))
                steps.append(step)
            results = await agent.run_task(steps)
            return results
        finally:
            await agent.stop()

    results = asyncio.run(_run())
    summary = summarize_results(results)
    typer.echo(json.dumps({"results": results, "summary": summary}, indent=2))

    if output:
        with open(output, "w") as f:
            json.dump({"results": results, "summary": summary}, f, indent=2)

    if summary["failed"] > 0:
        raise typer.Exit(1)


@app.command()
def version():
    """Show version."""
    typer.echo("agenticbrowser 0.1.0")


@app.command()
def validate(task_file: str = typer.Argument(..., help="Path to JSON task file")):
    """Validate a task file without running it."""
    with open(task_file) as f:
        task_data = json.load(f)

    errors = validate_task(task_data)
    if errors:
        typer.echo(f"Validation errors: {errors}", err=True)
        raise typer.Exit(1)

    typer.echo("Task is valid.")


if __name__ == "__main__":
    app()
