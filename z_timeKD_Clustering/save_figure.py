import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = ['ETTh1', 'exchange_rate', 'traffic', 'electricity', 'HVAC']
OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
TYPE = ['train', 'val']
CSV_DIR = './Result/csv'
FIG_DIR = './Result/figure'

os.makedirs(FIG_DIR, exist_ok=True)

print("\n\n============= Save Figure =============")

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        for tp in TYPE:
            csv_file = f'{CSV_DIR}/{ds}_o{output_len}_{tp}_res.csv'
            fig_path = f'{FIG_DIR}/{ds}_o{output_len}_{tp}_res.png'
            
            if not os.path.exists(csv_file):
                print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST) * len(TYPE)}) File not found: {csv_file}")
                idx += 1
                continue
            
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST) * len(TYPE)}) File: {fig_path}... ", end="")
            idx += 1
            
            df_plot_pd = pd.read_csv(csv_file)
            unique_labels = df_plot_pd['cluster_label'].unique()
            n_cluster = len(unique_labels)

            plt.figure(figsize=(14, 10))
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue="cluster_label",
                palette=sns.color_palette("hsv", n_cluster),
                data=df_plot_pd,
                legend="full",
                alpha=0.7
            )

            plt.title(f"{ds}({tp}), Output Length: {output_len}")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.grid(True)

            plt.savefig(fig_path, dpi=300, bbox_inches='tight')

            print("Completed.")

            plt.close()