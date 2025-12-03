# 1. 스크립트 파일 생성
cat > run_vis_all.sh << 'EOF'
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

echo "========================================================"
echo " [t-SNE & Clustering Evaluation (K-Means vs Spectral)] "
echo "========================================================"

for dataset in "${datasets[@]}"; do
    echo ""
    echo ">>> Processing: $dataset"
    
    # Python 스크립트 실행
    python vis.py --mode tsne --dataset "$dataset"
    
    echo "--------------------------------------------------------"
done

echo ""
echo "=== 모든 작업 완료 ==="
echo "생성된 이미지 파일들:"
ls tsne_*.png
EOF

# 2. 실행 권한 부여
chmod +x run_vis_all.sh

# 3. 바로 실행
./run_vis_all.sh