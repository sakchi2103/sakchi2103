import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the Excel File
df = pd.read_excel(r"C:\Users\Sakchi\Downloads\INT327.xlsx")

# Data CLEANING
print("Initial Shape:", df.shape)

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing pollutant data
df_clean = df.dropna(subset=['pollutant_min', 'pollutant_max', 'pollutant_avg']).copy() 

# Check data types and info
print("\nCleaned Data Info:")
print(df_clean.info())

# Define a consistent color palette
base_color = 'skyblue'
highlight_color = 'salmon'
text_color = 'dimgray'
grid_color = 'lightgray'

# STATES WITH HIGHEST AVERAGE POLLUTION
top_states = df_clean.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
top_states_df = top_states.reset_index()
top_states_df.columns = ['state', 'pollutant_avg']

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_states_df,
    x='pollutant_avg',
    y='state',
    hue='state',
    palette='viridis', 
    legend=False,
    color=base_color 
)
plt.title("Top 10 Polluted States (Avg Level)", color=text_color)
plt.xlabel("Average Pollution", color=text_color)
plt.ylabel("State", color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='x', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# BASIC STATISTICS
print("\nSummary Statistics:")
print(df_clean.describe(include='all'))

# POLLUTANT DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_clean,
    y='pollutant_id',
    hue='pollutant_id',
    order=df_clean['pollutant_id'].value_counts().index,
    palette='Set2', # Changed to a different palette
    legend=False,
    color=base_color # Added base color for consistency
)
plt.title('Pollutant Count Distribution', color=text_color)
plt.xlabel('Count', color=text_color)
plt.ylabel('Pollutant ID', color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='x', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# CORRELATION BETWEEN POLLUTANT VALUES
corr = df_clean[['pollutant_min', 'pollutant_max', 'pollutant_avg']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, linecolor='black') # Kept coolwarm for correlation
plt.title('Correlation between Pollutant Values', color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.tight_layout()
plt.show()

# AVERAGE POLLUTION LEVEL BY POLLUTANT TYPE
avg_pollutant = df_clean.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_pollutant.values, y=avg_pollutant.index, hue=avg_pollutant.index, palette='plasma', legend=False, color=base_color)
plt.title("Average Pollution Level by Pollutant", color=text_color)
plt.xlabel("Average Value", color=text_color)
plt.ylabel("Pollutant", color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='x', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# TOP 10 POLLUTED CITIES (by average pollutant level)
top_cities = df_clean.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, hue=top_cities.index, palette='magma', legend=False, color=base_color)
plt.title("Top 10 Polluted Cities (Avg Level)", color=text_color)
plt.xlabel("Average Pollution", color=text_color)
plt.ylabel("City", color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='x', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# Distribution of Average Pollution Levels
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['pollutant_avg'], bins=30, color=base_color, edgecolor='black', kde=True, line_kws={'color': highlight_color})
plt.title('Distribution of Average Pollution Levels', color=text_color)
plt.xlabel('Pollutant Avg', color=text_color)
plt.ylabel('Frequency', color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='y', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# Boxplot: Pollution by State
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_clean, x='state', y='pollutant_avg', hue='state', palette='Paired', legend=False) 
plt.xticks(rotation=90, color=text_color)
plt.yticks(color=text_color)
plt.title('Pollution Level Distribution by State', color=text_color)
plt.xlabel('State', color=text_color)
plt.ylabel('Pollutant Avg', color=text_color)
plt.grid(axis='y', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# Violin Plot: Pollution by Pollutant Type
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_clean, x='pollutant_id', y='pollutant_avg', hue='pollutant_id', palette='Set3', legend=False) 
plt.title('Pollution Levels by Pollutant Type', color=text_color)
plt.xlabel('Pollutant Type', color=text_color)
plt.ylabel('Average Level', color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.grid(axis='y', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# Pairplot: Relationships Between Pollution Metrics
sns.pairplot(df_clean[['pollutant_min', 'pollutant_max', 'pollutant_avg']], corner=True, diag_kind='kde', plot_kws={'color': base_color, 'alpha': 0.7}) # Removed redundant diag_kws color
plt.suptitle("Pairwise Relationships Between Pollutant Measures", y=1.02, color=text_color)
plt.tight_layout()
plt.show()

# Swarmplot: Pollutant Averages per City (Top 10 Cities by frequency)
top_cities_list = df_clean['city'].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.swarmplot(
    data=df_clean[df_clean['city'].isin(top_cities_list)],
    x='city',
    y='pollutant_avg',
    hue='city',
    palette='Spectral', # Changed palette
    legend=False,
    size=3 # Further decreased size to reduce overlap warnings
)
plt.title('Pollution Distribution in Top 10 Cities', color=text_color)
plt.xlabel('City', color=text_color)
plt.ylabel('Pollutant Avg', color=text_color)
plt.xticks(rotation=45, color=text_color)
plt.yticks(color=text_color)
plt.grid(axis='y', linestyle='--', color=grid_color)
plt.tight_layout()
plt.show()

# Heatmap: Pollution by State vs Pollutant Type
pivot_table = df_clean.pivot_table(values='pollutant_avg', index='state', columns='pollutant_id', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=0.5, linecolor='white') 
plt.title('Average Pollution by State and Pollutant Type', color=text_color)
plt.xlabel('Pollutant Type', color=text_color)
plt.ylabel('State', color=text_color)
plt.tick_params(axis='both', colors=text_color)
plt.tight_layout()
plt.show()

# HYPOTHESIS TESTING

print("\n--- HYPOTHESIS TESTING ---")

# Z-test (comparing mean pollution of two states)
print("\n--- Z-test: Comparing Average Pollution of Two States ---")
state1 = 'California'
state2 = 'Texas'

if state1 in df_clean['state'].unique() and state2 in df_clean['state'].unique():
    poll_state1 = df_clean[df_clean['state'] == state1]['pollutant_avg']
    poll_state2 = df_clean[df_clean['state'] == state2]['pollutant_avg']

    n1 = len(poll_state1)
    n2 = len(poll_state2)

    if n1 > 0 and n2 > 0:
        mean1 = np.mean(poll_state1)
        mean2 = np.mean(poll_state2)
        std1 = np.std(poll_state1, ddof=1) # sample standard deviation
        std2 = np.std(poll_state2, ddof=1)

        # Pooled standard deviation (assuming equal variances for simplicity)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        standard_error_diff = pooled_std * np.sqrt(1/n1 + 1/n2)

        if standard_error_diff > 0:
            z_statistic = (mean1 - mean2) / standard_error_diff
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic))) 

            alpha = 0.05
            print(f"Comparing average pollution of {state1} (mean={mean1:.2f}, n={n1}) and {state2} (mean={mean2:.2f}, n={n2})")
            print(f"Z-statistic: {z_statistic:.2f}")
            print(f"P-value: {p_value:.3f}")
            if p_value < alpha:
                print(f"Reject the null hypothesis: There is a significant difference in average pollution between {state1} and {state2}.")
            else:
                print(f"Fail to reject the null hypothesis: There is no significant difference in average pollution between {state1} and {state2}.")
        else:
            print("Standard error is zero, cannot perform z-test.")
    else:
        print(f"Not enough data for either {state1} or {state2} to perform z-test.")
else:
    print(f"One or both of the specified states ({state1}, {state2}) not found in the data.")

# Chi-Square Test (independence of state and a binary pollutant level category)
print("\n--- Chi-Square Test: Independence of State and High/Low Pollution ---")
pollutant_threshold = df_clean['pollutant_avg'].median()
df_clean.loc[:, 'pollution_level'] = np.where(df_clean['pollutant_avg'] > pollutant_threshold, 'High', 'Low') 

contingency_table = pd.crosstab(df_clean['state'], df_clean['pollution_level'])
print("\nContingency Table:")
print(contingency_table)

chi2_statistic, p_value, dof, expected_frequencies = stats.chi2_contingency(contingency_table)

alpha = 0.05
print(f"\nChi-square statistic: {chi2_statistic:.2f}")
print(f"P-value: {p_value:.3f}")
print(f"Degrees of freedom: {dof}")
print("\nExpected Frequencies:")
print(pd.DataFrame(expected_frequencies, index=contingency_table.index, columns=contingency_table.columns))

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant association between the state and the pollution level (High/Low).")
else:
    print("Fail to reject the null hypothesis: There is no significant association between the state and the pollution level (High/Low).")

# INSIGHT SUMMARY
print("\n--- INSIGHTS ---")
print(f"Most common pollutant: {df_clean['pollutant_id'].value_counts().idxmax()}")
print(f"Total stations: {df_clean['station'].nunique()}")
print(f"Cities monitored: {df_clean['city'].nunique()}")
print(f"States monitored: {df_clean['state'].nunique()}")
print(f"Highest average pollution level: {df_clean['pollutant_avg'].max()}")
