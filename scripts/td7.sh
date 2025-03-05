ENV_ID="Humanoid-v4"
rye run python cli.py td7 --env-id $ENV_ID --save-path "save/TD7" --seed 1 --use-progressbar &&
rye run python cli.py td7 --env-id $ENV_ID --save-path "save/TD7" --seed 2 --use-progressbar &&
rye run python cli.py td7 --env-id $ENV_ID --save-path "save/TD7" --seed 3 --use-progressbar &&
rye run python cli.py td7 --env-id $ENV_ID --save-path "save/TD7" --seed 4 --use-progressbar &&
rye run python cli.py td7 --env-id $ENV_ID --save-path "save/TD7" --seed 5 --use-progressbar &&