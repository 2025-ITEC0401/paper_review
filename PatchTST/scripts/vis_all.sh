# 1. 시각화 일괄 실행 스크립트 생성 (vis_all.sh)
cat > vis_all.sh << 'EOF'
#!/bin/bash

# 시각화할 데이터셋 목록
datasets=(
    "AtrialFibrillation"
    "StandWalkJump"
    "ArticularyWordRecognition"
    "NATOPS"
    "PenDigits"
    "UWaveGestureLibrary"
    "PEMS-SF"
)

echo "=== 모든 데이터셋 t-SNE 시각화 시작 ==="

for dataset in "${datasets[@]}"; do
    echo "------------------------------------------------"
    echo "Processing: $dataset"
    # vis.py 실행 (mode: tsne)
    python vis.py --mode tsne --dataset "$dataset"
done

echo "------------------------------------------------"
echo "=== 모든 작업 완료 ==="
echo "생성된 이미지 파일 목록:"
ls tsne_*.png
EOF

# 2. 실행 권한 부여
chmod +x vis_all.sh

# 3. 스크립트 실행
./vis_all.sh