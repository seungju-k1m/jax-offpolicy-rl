import click
from td7.run import run_td7, args_for_td7
from sale_tqc.run import run_sale_tqc, args_for_sale_tqc
from simba.run import run_simba, args_for_simba


@click.command("td7")
@args_for_td7
def cli_td7(*args, **kwargs):
    """Run TD7 Algorithm with specified hyperparameters"""
    run_td7(*args, **kwargs)


@click.command("sale-tqc")
@args_for_sale_tqc
def cli_sale_tqc(*args, **kwargs):
    """RUN SALE-TQC."""
    run_sale_tqc(*args, **kwargs)


@click.command("simba")
@args_for_simba
def cli_simba(*args, **kwargs):
    """RUN SIMBA."""
    run_simba(*args, **kwargs)


@click.group()
def cli():
    """Command Line Interface."""


cli.add_command(cli_td7)
cli.add_command(cli_sale_tqc)
cli.add_command(cli_simba)

if __name__ == "__main__":
    cli()
