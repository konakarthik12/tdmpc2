#!/nfshomes/kkona/local/bin/fish
mkdir -p $SLURM_WORK_DIR
cd $SLURM_WORK_DIR


nvidia-smi

set repo_url $argv[1]
set commit_hash $argv[2]

echo "Repo url: $repo_url"
echo "Commit hash: $commit_hash"
echo "Overrides: $argv"

mkdir tdmpc2

git -C tdmpc2 init
git -C tdmpc2 remote add origin $repo_url
git -C tdmpc2 fetch --depth 1 origin $commit_hash
git -C tdmpc2 checkout FETCH_HEAD


. tdmpc2/setup_cluster.fish $SLURM_WORK_DIR

