#!/usr/bin/env fish
set setup_dir $SCRATCH_DIR
if test (count $argv) -ge 1
    set setup_dir $argv[1]
end
echo "Setting up cluster in $(realpath $argv[1])"
. /fs/nexus-projects/KGB-MBRL/scratch_hold/setup_all.fish $argv[1]

pushd (dirname (status -f))
cd ..

echo "Generating tdmpc2 conda environment"
conda env update -n tdmpc2 -f environment.yaml -q

popd
echo "Cluster is ready to run tdmpc2"