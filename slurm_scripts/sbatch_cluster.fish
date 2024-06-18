#!/nfshomes/kkona/local/bin/fish
mkdir -p $SLURM_WORK_DIR
cd $SLURM_WORK_DIR


nvidia-smi

set -x REPO_URL $argv[1]
set -x COMMIT_SHA $argv[2]

echo "Repo url: $REPO_URL"
echo "Commit hash: $COMMIT_SHA"
echo "Overrides: $argv"

set repo_dir $SLURM_WORK_DIR/tdmpc2
mkdir $repo_dir

git -C $repo_dir init
git -C $repo_dir remote add origin $REPO_URL
git -C $repo_dir fetch --depth 1 origin $COMMIT_SHA
git -C $repo_dir checkout FETCH_HEAD &> /tmp/checkout.lot || begin; cat /tmp/checkout.log >&2; exit 1; end


. $repo_dir/slurm_scripts/setup_cluster.fish $SLURM_WORK_DIR


cd "$SLURM_WORK_DIR"

echo "Running task for real..."
set work_dir "$SLURM_WORK_DIR/work_dir"
mkdir "$work_dir"
cd "$work_dir"
echo "Working directory: $(pwd)"

echo "Starting job..."
echo "with job id: $SLURM_JOB_ID"
echo "in repo: $argv[1]"
echo "with commit: $argv[2]"
echo "with args: $argv[3..-1]"
echo "Running on $(hostname)"
echo "In directory: $(pwd)"
conda run -n tdmpc2 --live-stream python "$SLURM_WORK_DIR/tdmpc2/tdmpc2/train.py" $argv[3..-1]

echo "Task complete"