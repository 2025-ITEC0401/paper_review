#!/bin/bash
# 로컬 -> 원격 서버로 Merit 폴더 동기화

REMOTE_HOST="lab-server"
REMOTE_PATH="~/Merit"
LOCAL_PATH="/Users/isangmin/Desktop/종합설계프로젝트/paper_review/Merit"

echo "Syncing Merit from local to remote server..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '/models' \
    ${LOCAL_PATH}/ ${REMOTE_HOST}:${REMOTE_PATH}/

echo "Sync completed!"
