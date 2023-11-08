# python scripts/daamwandb.py --prompt "attractive woman in the street" --group 1 --words attractive --wandb --group article
# python scripts/daamwandb.py --prompt "attractive woman in the street" --negative_prompt unattractive --group 1 --words attractive unattractive --wandb --group article
# python scripts/daamwandb.py --prompt "an attractive woman in the street" --group 1 --words an attractive --wandb --group article
# python scripts/daamwandb.py --prompt "an attractive woman in the street" --negative_prompt unattractive --group 1 --words an attractive unattractive --wandb --group article

python scripts/daamwandb.py --prompt "fresh orange in the plate" --words fresh orange --wandb --group article_1 --tag nooverlay
python scripts/daamwandb.py --prompt "fresh orange in the plate" --negative stale --words fresh orange stale --wandb --group article_1 --tag nooverlay
python scripts/daamwandb.py --prompt "a fresh orange in the plate" --words a fresh orange --wandb --group article_1 --tag nooverlay
python scripts/daamwandb.py --prompt "a fresh orange in the plate" --negative stale --words a fresh orange stale --wandb --group article_1 --tag nooverlay
python scripts/daamwandb.py --prompt "fresh oranges in the plate" --words fresh oranges --wandb --group article_1 --tag nooverlay
python scripts/daamwandb.py --prompt "fresh oranges in the plate" --negative stale --words fresh oranges stale --wandb --group article_1 --tag nooverlay