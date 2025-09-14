#!/bin/bash

# SSH Configuration
SSH_HOST="103.196.86.163"
SSH_PORT="16392"
SSH_USER="root"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_PATH="/workspace/"

# Copy files to remote server
scp -P ${SSH_PORT} -i ${SSH_KEY} train.py ${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}
scp -P ${SSH_PORT} -i ${SSH_KEY} requirements.txt ${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}
scp -P ${SSH_PORT} -i ${SSH_KEY} install_dependencies.sh ${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}
