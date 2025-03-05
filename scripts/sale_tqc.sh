ENV_ID="Humanoid-v4"
rye run python cli.py sale-tqc --env-id $ENV_ID --save-path "save/SALE-TQC" --seed 1 --use-progressbar  &&
rye run python cli.py sale-tqc --env-id $ENV_ID --save-path "save/SALE-TQC" --seed 2 --use-progressbar  &&
rye run python cli.py sale-tqc --env-id $ENV_ID --save-path "save/SALE-TQC" --seed 3 --use-progressbar  &&
rye run python cli.py sale-tqc --env-id $ENV_ID --save-path "save/SALE-TQC" --seed 4 --use-progressbar  &&
rye run python cli.py sale-tqc --env-id $ENV_ID --save-path "save/SALE-TQC" --seed 5 --use-progressbar  &&