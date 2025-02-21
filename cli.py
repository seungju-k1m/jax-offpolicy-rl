import click
from td7.run import run_td7


@click.command("td7")
@click.option(
    "--env-id", type=str, required=True, help="Environment ID (e.g., 'HalfCheetah-v4')"
)
@click.option(
    "--total-timesteps",
    type=int,
    default=5_000_000,
    show_default=True,
    help="Total training timesteps",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
@click.option(
    "--save-path",
    type=str,
    default="save/TD7/",
    show_default=True,
    help="Path to save checkpoints",
)
@click.option(
    "--discount-factor",
    type=float,
    default=0.99,
    show_default=True,
    help="Discount factor (gamma)",
)
@click.option(
    "--target-update-rate",
    type=int,
    default=250,
    show_default=True,
    help="Target network update frequency",
)
@click.option(
    "--policy-freq",
    type=int,
    default=2,
    show_default=True,
    help="Policy network update frequency",
)
@click.option(
    "--target-noise",
    type=float,
    default=0.2,
    show_default=True,
    help="Target policy noise",
)
@click.option(
    "--exploration-timesteps",
    type=int,
    default=25_000,
    show_default=True,
    help="Timesteps for exploration before training starts",
)
@click.option(
    "--max-episodes-per-ckpt",
    type=int,
    default=20,
    show_default=True,
    help="Max episodes per checkpoint update",
)
@click.option(
    "--init-episodes-per-ckpt",
    type=int,
    default=1,
    show_default=True,
    help="Initial episodes per checkpoint update",
)
@click.option(
    "--batch-size",
    type=int,
    default=256,
    show_default=True,
    help="Batch size for training",
)
@click.option(
    "--reset-weight",
    type=float,
    default=0.9,
    show_default=True,
    help="Weight for resetting best min return",
)
@click.option(
    "--update-timestep",
    type=int,
    default=int(75e4),
    show_default=True,
    help="Timesteps before increasing checkpoint episodes",
)
@click.option(
    "--eval-period",
    type=int,
    default=5_000,
    show_default=True,
    help="Evaluation frequency (in timesteps)",
)
@click.option("--use-progressbar", is_flag=True, help="Show training progress bar")
@click.option("--deterministic", is_flag=True, help="Use deterministic evaluation")
def cli_td7(*args, **kwargs):
    """Run TD7 Algorithm with specified hyperparameters"""
    run_td7(*args, **kwargs)


@click.group()
def cli():
    """Command Line Interface."""


cli.add_command(cli_td7)

if __name__ == "__main__":
    cli()
