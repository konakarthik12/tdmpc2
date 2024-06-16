#!/usr/bin/env fish

echo "Setting up cluster in $(realpath $argv[1])"
. /fs/nexus-projects/KGB-MBRL/scratch_hold/setup_all.fish $argv[1]

pushd (dirname (status -f))
cd ..

mamba env update -n tdmpc2 -f environment.yaml -q


mamba run -n tdmpc2 pip install "git+https://github.com/StanfordVL/OmniGibson@v1.0.0#egg=omnigibson" -q
mamba run -n tdmpc2 pip install -r tdmpc2_requirements.txt -q
popd