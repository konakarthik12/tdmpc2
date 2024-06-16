#!/nfshomes/kkona/local/bin/fish
mkdir -p $SLURM_WORK_DIR
cd $SLURM_WORK_DIR


nvidia-smi

set repo_url $argv[1]
set commit_hash $argv[2]

echo "Repo url: $repo_url"
echo "Commit hash: $commit_hash"
echo "Overrides: $argv"

set repo_dir $SLURM_WORK_DIR/tdmpc2
mkdir $repo_dir

git -C $repo_dir init
git -C $repo_dir remote add origin $repo_url
git -C $repo_dir fetch --depth 1 origin $commit_hash
git -C $repo_dir checkout FETCH_HEAD


. $repo_dir/slurm_scripts/setup_cluster.fish $SLURM_WORK_DIR

