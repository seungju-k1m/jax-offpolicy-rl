ENV_ID="Humanoid-v4"
rye run python cli.py simba --env-id $ENV_ID --seed 1 --use-progressbar &&
rye run python cli.py simba --env-id $ENV_ID --seed 2 --use-progressbar &&
rye run python cli.py simba --env-id $ENV_ID --seed 3 --use-progressbar &&
rye run python cli.py simba --env-id $ENV_ID --seed 4 --use-progressbar &&
rye run python cli.py simba --env-id $ENV_ID --seed 5 --use-progressbar &&