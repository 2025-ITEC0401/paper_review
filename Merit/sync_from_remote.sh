#!/bin/bash
# 원격 서버 -> 로컬로 Merit 폴더 동기화

REMOTE_HOST="lab-server"
REMOTE_PATH="~/Merit"
LOCAL_PATH="/Users/isangmin/Desktop/종합설계프로젝트/paper_review/Merit"

echo "Syncing Merit from remote server to local..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'models' \
    ${REMOTE_HOST}:${REMOTE_PATH}/ ${LOCAL_PATH}/

echo "Sync completed!"
