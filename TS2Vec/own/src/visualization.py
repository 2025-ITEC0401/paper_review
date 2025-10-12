# -*- coding: utf-8 -*-
"""
시각화 관련 함수들을 관리하는 모듈
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from .config import N_SAMPLE_PLOTS

# 한글 폰트 설정 (경고 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_sample_data(data, n_samples=N_SAMPLE_PLOTS, save_path="result/sample_data.png"):
    """샘플 데이터를 시각화하는 함수"""
    plt.figure(figsize=(12, 8))
    
    # 처음 n_samples개의 시계열을 플롯
    for i in range(min(n_samples, data.shape[0])):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(data[i, :, 0])
        plt.title("Sample " + str(i + 1))
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("샘플 데이터 그래프를 " + save_path + "에 저장했습니다.")
    plt.close()

def plot_loss_curve(loss_log, save_path="result/loss_curve.png"):
    """손실률 그래프를 그리는 함수"""
    if not loss_log:
        print("손실 로그가 비어있습니다.")
        return
        
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_log) + 1)
    plt.plot(epochs, loss_log, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title("Training Loss Over Time", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 최소 손실값 표시
    min_loss_idx = np.argmin(loss_log)
    plt.plot(epochs[min_loss_idx], loss_log[min_loss_idx], 'ro', markersize=8)
    plt.annotate("Min Loss: {:.4f}".format(loss_log[min_loss_idx]), 
                xy=(epochs[min_loss_idx], loss_log[min_loss_idx]),
                xytext=(epochs[min_loss_idx] + len(epochs)*0.1, loss_log[min_loss_idx]),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 시작과 끝 손실값 표시
    plt.text(0.02, 0.98, "Initial Loss: {:.4f}".format(loss_log[0]), 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(0.02, 0.85, "Final Loss: {:.4f}".format(loss_log[-1]), 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("손실률 그래프를 " + save_path + "에 저장했습니다.")
    plt.close()
