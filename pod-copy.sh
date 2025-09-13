#!/bin/bash
scp -P 46885 -i ~/.ssh/id_ed25519 train.py root@62.169.158.26:/workspace/
scp -P 46885 -i ~/.ssh/id_ed25519 requirements.txt root@62.169.158.26:/workspace/
