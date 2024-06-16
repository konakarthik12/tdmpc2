#!/nfshomes/kkona/local/bin/fish

cd (dirname (status -f))
. ./utils.fish
. /fs/nexus-projects/KGB-MBRL/scratch_hold/init_scratch_dir.fish
. /fs/nexus-projects/KGB-MBRL/scratch_hold/setup_conda.fish

set repo_url (git remote get-url origin)

if ! set commit_hash (verify_changes_pushed)
    echo $commit_hash
    exit 1
end
echo "Repo url: $repo_url"

echo "Commit hash: $commit_hash"
if test (count $argv) -le 1
    echo "Usage: submit_cluster.fish <task_name> <task_args...>"
    exit 1
end

if ! set task_name (python3 extract_task_name.py $argv)
    echo "Failed to extract task name"
    exit 1
end
set full_name "tdmpc2-$task_name"
sbatch --job-name="$full_name" sbatch_cluster.sh $repo_url $commit_hash $argv
echo "Job submitted"
