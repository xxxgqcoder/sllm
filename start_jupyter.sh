#!/usr/bin/env bash
set -e
cd $(dirname "$0")

port_num=2003
jupyter_log_file='.logs/jupyter.logs'

ps ax | grep -E "jupyter-lab --port=${port_num}" | grep -v "grep" | awk '{print $1}' | xargs -I {} kill -9 {}

rm -r ${jupyter_log_file} || True

mkdir -p $(dirname ${jupyter_log_file})

touch ${jupyter_log_file}

nohup jupyter-lab --port=${port_num}> ${jupyter_log_file} 2>&1 &

echo 'jupyter-lab is running'

