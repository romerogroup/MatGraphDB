import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data from the JSON file
database_name='main-06242024'
with open(f'examples/scripts/{database_name}.json', 'r') as f:
    data = json.load(f)

# Prepare data for plotting
all_data = []
for property_name, models in data.items():
    for model_name, metrics in models.items():
        for metric_name, metric_value in metrics.items():
            all_data.append({
                'Property': property_name,
                'Model': model_name,
                'Metric': metric_name,
                'Value': metric_value
            })

# Convert data to a DataFrame
df = pd.DataFrame(all_data)
df = df[df['Value'].notnull()]  # Filter out null values

# Define the directory to save the plots
save_dir = os.path.join('examples','scripts','databases',database_name,'property_plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the aesthetic style
sns.set_style("whitegrid")


# Create plots for each property
for property_name, property_df in df.groupby('Property'):
    plt.figure(figsize=(12, 8))  # Adjust size as necessary
    metrics = property_df['Metric'].unique()
    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(2, 3, i)  # Adjust grid size based on the number of metrics
        sns.barplot(data=property_df[property_df['Metric'] == metric], x='Model', y='Value', hue='Model')
        plt.title(f"{property_name} - {metric}")
        plt.xticks(rotation=45)
        plt.legend().remove()  # This line removes the legend from the subplot
        plt.tight_layout()

    # Save the plot with a filename based on the property
    plt.savefig(f"{save_dir}/{property_name.replace(' ', '_')}.png")
    plt.close()  # Close the figure after saving to free up memory