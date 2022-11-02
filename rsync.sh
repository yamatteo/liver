#!/bin/bash

source_path="/home/yamatteo/hogwild_liver"
target_path="uiu95bi@jean-zay.idris.fr:/gpfswork/rech/otc/uiu95bi"

while inotifywait -r -e modify,create,delete $source_path
do
    rsync -azh -e 'ssh -A -J cpenzo@styx.obspm.fr,cpenzo@145.238.179.16' $source_path $target_path \
          --progress \
          --delete --force \
          --exclude=".idea" \
          --exclude="__pycache__" \
          --exclude="venv"
done

