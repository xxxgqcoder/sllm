#!/usr/bin/env bash
set -e
cd $(dirname "$0")


train_log='pretrain.log'

rm $train_log || True
touch $train_log

nohup python -u pretrain.py > $train_log 2>&1 &

echo "pretrain is running"