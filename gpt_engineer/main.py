import sys
# sys.path.append('C:\\Users\\User\\Documents\\GitHub\\gpt-engineer') # Path to gpt-engineer folder 
sys.path.append('C:/Users/abfernan/CrossCanFloodMapping/TestingNew/gpt-engineer/gpt_engineer/')

import logging
from pathlib import Path
import typer 

import gpt_engineer.db as db
from gpt_engineer.ai import AI, fallback_model
from gpt_engineer.collect import collect_learnings
from db import archive
from db import DB, DBs
import gpt_engineer.learning as learning
from learning import collect_consent
from gpt_engineer.steps import STEPS, Config as StepsConfig

app = typer.Typer()


@app.command()
def main(
    project_path: str = typer.Argument("projects/example", help="path"),
    model: str ="gpt-3.5-turbo",
    temperature: float = 0.1,
    steps_config: StepsConfig = typer.Option(
        StepsConfig.DEFAULT, "--steps", "-s", help="decide which steps to run"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    model = fallback_model(model)
    ai = AI(model_name = model, temperature=temperature)

    input_path = Path(project_path).absolute()
    memory_path = input_path / "memory"
    workspace_path = input_path / "workspcace"
    archive_path = input_path / "archive"

    dbs = DBs(
        memory=DB(memory_path),
        logs=DB(memory_path / "logs"),
        input=DB(input_path),
        workspace=DB(workspace_path),
        preprompts=DB(Path(__file__).parent / "preprompts"),
        archive=DB(archive_path),
    )

    if steps_config not in [
        StepsConfig.EXECUTE_ONLY,
        StepsConfig.USE_FEEDBACK,
        StepsConfig.EVALUATE,
    ]:
        archive(dbs)

    steps = STEPS[steps_config]
    for step in steps:
        messages = step(ai, dbs)
        dbs.logs[step.__name__] = AI.serialize_messages(messages)

    if collect_consent():
        collect_learnings(model, temperature, steps, dbs)

    dbs.logs["token_usage"] = ai.format_token_usage_log()


if __name__ == "__main__":
    app()
