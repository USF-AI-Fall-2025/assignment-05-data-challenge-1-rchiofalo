import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# load the training data
data_path = Path('../data/earnings_train.csv')
df = pd.read_csv(data_path)

print("=" * 60)
print("DATA EXPLORATION SUMMARY")
print("=" * 60)

# data types
print("\n1. DATA TYPES:")
print("-" * 60)
print(df.dtypes)

# missing data
print("\n\n2. MISSING DATA:")
print("-" * 60)
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("No missing data found in any column!")
else:
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"{col}: {missing_data[col]} missing ({missing_data[col]/len(df)*100:.2f}%)")

# categorical columns
print("\n\n3. CATEGORICAL COLUMNS - UNIQUE VALUES:")
print("-" * 60)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"\n{col}: {unique_count} unique values")
    if unique_count <= 10:
        print(f"  Values: {df[col].unique().tolist()}")

# numeric columns
print("\n\n4. NUMERIC COLUMNS - RANGES:")
print("-" * 60)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
    print(f"  Non-zero: {(df[col] != 0).sum()} ({(df[col] != 0).sum()/len(df)*100:.1f}%)")

# wage columns analysis
wage_cols = [col for col in numeric_cols if 'WAGE' in col]
print("\n\n5. WAGE COLUMNS ANALYSIS:")
print("-" * 60)
for col in wage_cols:
    non_zero = df[df[col] > 0][col]
    if len(non_zero) > 0:
        print(f"\n{col} (non-zero values only):")
        print(f"  Count: {len(non_zero)}")
        print(f"  Mean: {non_zero.mean():.2f}")
        print(f"  Std: {non_zero.std():.2f}")

# generate visualizations
print("\n\n6. GENERATING VISUALIZATIONS...")
print("-" * 60)

# wage distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
for idx, col in enumerate(wage_cols):
    non_zero_data = df[df[col] > 0][col]
    axes[idx].hist(non_zero_data, bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col} (non-zero values)')
    axes[idx].set_xlabel('Wage')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/wage_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wage_distributions.png")

# wage correlation
wage_df = df[wage_cols]
correlation_matrix = wage_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Between Wage Years')
plt.tight_layout()
plt.savefig('../data/wage_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wage_correlation.png")

# wage by award category
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for idx, col in enumerate(wage_cols):
    non_zero = df[df[col] > 0]
    award_wages = non_zero.groupby('AWARD_CATEGORY')[col].mean().sort_values(ascending=False)
    axes[idx].barh(range(len(award_wages)), award_wages.values)
    axes[idx].set_yticks(range(len(award_wages)))
    axes[idx].set_yticklabels(award_wages.index, fontsize=9)
    axes[idx].set_xlabel('Average Wage')
    axes[idx].set_title(f'Average {col} by Award Category')
    axes[idx].grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../data/wage_by_award.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wage_by_award.png")

print("\n\nAnalysis complete!")

