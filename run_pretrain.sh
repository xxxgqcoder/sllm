#!/usr/bin/env bash
set -e
cd $(dirname "$0")


train_log='pretrain.log'

touch $train_log || true

nohup python pretrain.py > $train_log 2>&1 &

echo "pretrain is running"