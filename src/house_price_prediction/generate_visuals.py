import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.edgecolor'] = '#4e4151'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'sans-serif'
sns.set_palette(["#a02933", "#dbba78", "#4e4151"])

def generate_plots(data_path="data/raw/train.csv", output_dir="src/webapp/static/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # 1. SalePrice Distribution
    plt.figure()
    sns.histplot(df['SalePrice'], kde=True, color="#a02933")
    plt.title("Distribution of Sale Prices", fontsize=16, fontweight='bold', color="#4e4151")
    plt.xlabel("Sale Price ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.savefig(os.path.join(output_dir, "price_dist.png"), transparent=True)
    plt.close()
    
    # 2. Overall Quality vs SalePrice
    plt.figure()
    sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette="YlOrRd")
    plt.title("Overall Quality vs Sale Price", fontsize=16, fontweight='bold', color="#4e4151")
    plt.xlabel("Overall Quality (1-10)", fontsize=12)
    plt.ylabel("Sale Price ($)", fontsize=12)
    plt.savefig(os.path.join(output_dir, "quality_price.png"), transparent=True)
    plt.close()
    
    # 3. Top Correlations
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    top_corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(10)
    sns.barplot(x=top_corr.values, y=top_corr.index, palette="flare")
    plt.title("Top Features Correlated with Price", fontsize=16, fontweight='bold', color="#4e4151")
    plt.xlabel("Correlation Coefficient", fontsize=12)
    plt.savefig(os.path.join(output_dir, "correlations.png"), transparent=True)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    generate_plots()
