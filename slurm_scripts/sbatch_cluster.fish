#!/nfshomes/kkona/local/bin/fish

cd (dirname (status -f))


echo "Starting job..."
echo "with job id: $SLURM_JOB_ID"
echo "with args: $argv"
echo "Running on $(hostname)"

set slurm_dir $SCRATCH_DIR/slurm_runs/$SLURM_JOB_ID/
mkdir -p $slurm_dir
nvidia-smi

set repo_url $argv[1]
set commit_hash $argv[2]
set argv $argv[3..-1]
echo "Repo url: $repo_url"
echo "Commit hash: $commit_hash"
echo "Overrides: $argv"

. setup_cluster.fish $slurm_dir

cd $slurm_dir

mkdir tdmpc2

git -C tdmpc2 init
git -C tdmpc2 remote add origin $repo_url
git -C tdmpc2 fetch --depth 1 origin $commit_hash
git -C tdmpc2 checkout FETCH_HEAD

echo $slurm_dir