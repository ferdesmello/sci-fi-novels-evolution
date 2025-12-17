"""
This script process questions and answers data, test some hypotheses, and makes figures.

Modules:
    - pandas
    - matplotlib.pyplot
    - seaborn
    - scipy.stats
    - numpy
    - plotly
    - typing
    - pathlib
    - sys
"""

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import plotly.express as px
from typing import List

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED, ANSWERS, FIGURES, VARIABILITY_IN_ANSWERS

#---------------------------------------------------------------------------------------------------
print("Reading and processing tha data...")

# Read the data
df_filtered = pd.read_csv(FILTERED / "sci-fi_novels_FILTERED.csv", sep=";", encoding="utf-8-sig")
df_top = pd.read_csv(FILTERED / "sci-fi_novels_TOP_Wiki.csv", sep=";", encoding="utf-8-sig")
df_top_AI = pd.read_csv(ANSWERS / "sci-fi_novels_AI_ANSWERS.csv", sep=";", encoding="utf-8-sig")
df_top_AI_gender = pd.read_csv(ANSWERS / "sci-fi_novels_AI_ANSWERS_GENDER.csv", sep=";", encoding="utf-8-sig")

#print(df_top_AI.info())
#print(df.head())

#---------------------------------------------------------------------------------------------------
# Exclude novels of before 1860 (allmost none)

mask_all = df_filtered['decade'] >= 1860
df_filtered = df_filtered[mask_all]

mask_top = df_top['decade'] >= 1860
df_top = df_top[mask_top]

mask_top_AI = df_top_AI['decade'] >= 1860
df_top_AI = df_top_AI[mask_top_AI]

#---------------------------------------------------------------------------------------------------
# Include author gender to the main dataframe

# Create a dictionary from df_top_AI_gender
author_gender_dict = df_top_AI_gender.set_index('author')["gender"].to_dict()

# Map the "author gender" based on 'author' column from df_top_AI
df_top_AI["author gender"] = df_top_AI['author'].map(author_gender_dict).fillna('Uncertain')

#---------------------------------------------------------------------------------------------------
# Process for the complex figures
#---------------------------------------------------------------------------------------------------
# Author and protagonist gender

df_top_AI_new = df_top_AI.copy()
df_top_AI_new["genders"] = df_top_AI_new["author gender"] + " / " + df_top_AI_new['18 protagonist gender']

# Count the occurrences of each category per decade
category_counts_genders = pd.crosstab(df_top_AI_new['decade'], df_top_AI_new["genders"])

#------------------------------------
# Define a list of the specific 'Other' columns you want to combine
other_columns_to_combine = ['Other / Male',
                            'Other / Female',
                            'Other / Other',
                            'Other / Uncertain',
                            'Other / Not applicable']

# Create the 'Other / All combined' column, handling missing columns gracefully
category_counts_genders['Other / All combined'] = 0
for col in other_columns_to_combine:
    if col in category_counts_genders.columns:
        category_counts_genders['Other / All combined'] = category_counts_genders['Other / All combined'] + category_counts_genders[col]

# Drop the original 'Other' columns, handling missing columns gracefully
columns_to_drop = other_columns_to_combine
columns_to_drop_existing = [col for col in columns_to_drop if col in category_counts_genders.columns]
category_counts_genders = category_counts_genders.drop(columns=columns_to_drop_existing, errors='ignore')

#------------------------------------
# Define a list of the specific 'Uncertain' columns you want to combine
uncertain_columns_to_combine = ['Uncertain / Male',
                                'Uncertain / Female',
                                'Uncertain / Other',
                                'Uncertain / Uncertain',
                                'Uncertain / Not applicable']

# Create the 'Uncertain / All combined' column, handling missing columns gracefully
category_counts_genders['Uncertain / All combined'] = 0
for col in uncertain_columns_to_combine:
    if col in category_counts_genders.columns:
        category_counts_genders['Uncertain / All combined'] = category_counts_genders['Uncertain / All combined'] + category_counts_genders[col]

# Drop the original 'Uncertain' columns, handling missing columns gracefully
columns_to_drop = uncertain_columns_to_combine
columns_to_drop_existing = [col for col in columns_to_drop if col in category_counts_genders.columns]
category_counts_genders = category_counts_genders.drop(columns=columns_to_drop_existing, errors='ignore')

#------------------------------------
# Normalize the counts to get percentages
category_percent_genders = category_counts_genders.div(category_counts_genders.sum(axis=1), axis=0) * 100

#print(category_percent_genders)
print("\ngenders", df_top_AI_new["genders"].unique())
#print("\n")

#---------------------------------------------------------------------------------------------------
# Variation in Answers

# File names in the normally ordered alternatives
file_names = [
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_01.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_02.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_03.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_04.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_05.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_06.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_07.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_08.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_09.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_10.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_11.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_12.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_13.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_14.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_15.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_16.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_17.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_18.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_19.csv",
    VARIABILITY_IN_ANSWERS / "sci-fi_novels_AI_ANSWERS_TEST_20.csv"
    ]

column_order = [
    '1 accuracy',
    '2 discipline',
    '3 light heavy',
    '4 time',
    '5 mood',
    '6 ending',

    '7 social political',
    '8 politically unified',
    '9 on Earth',
    '10 post apocalyptic',
    '11 conflict',

    '12 aliens',
    '13 aliens are',
    '14 robots and AI',
    '15 robots and AI are',

    '16 protagonist',
    '17 protagonist nature',
    '18 protagonist gender',
    '19 protagonist is',

    '20 virtual',
    '21 virtual is',
    '22 biotech',
    '23 biotech is',
    '24 transhuman',
    '25 transhuman is',
    '26 tech and science',

    '27 social issues',
    '28 enviromental'
]

dfs = [] # List of DataFrames

# Process the DataFrames for variation in answers
#for file_name in file_names_s:
for file_name in file_names:
    df_file_name = pd.read_csv(file_name, sep=";", encoding="utf-8-sig")

    df_file_name['year'] = df_file_name['year'].astype('string')
    df_file_name['id'] = df_file_name['title'] + " (" + df_file_name['year'] + ") " + df_file_name['author']
    df_file_name = df_file_name.set_index('id')

    df_file_name = df_file_name.reindex(columns=column_order)

    #print(df_file_name.iloc[18,4])

    dfs.append(df_file_name)

 # Create a 3D array (rows x columns x runs)
data_array = np.stack([df.to_numpy() for df in dfs], axis=-1) # Shape: (rows, columns, runs)

#-------------------------
# Difference two by two of the answers

# Initialize a dataframe to store the sum of differences
comparison_sum = pd.DataFrame(0, index=dfs[0].index, columns=dfs[0].columns)

# Count the number of comparisons made (for averaging later)
num_comparisons = 0

# Compare each pair of dataframes
for i in range(len(dfs)):
    for j in range(i + 1, len(dfs)):
        # Compare dataframes element-wise, convert booleans to integers (1 for different, 0 for same)
        difference = (dfs[i] != dfs[j]).astype(int)
        
        # Add to comparison sum
        comparison_sum += difference
        
        # Increment the comparison count
        num_comparisons += 1

# Calculate the mean difference for each cell
df_mean_difference = comparison_sum / num_comparisons

# Add column and row of means
df_mean_difference['Mean'] = df_mean_difference.mean(axis=1)
df_mean_difference.loc['Mean'] = df_mean_difference.mean(axis=0)

#-----------------------------
# Percent Agreement / Mode Consistency

# Initialize empty arrays to store modes and percent agreements
rows, cols = data_array.shape[0], data_array.shape[1]
modes = np.empty((rows, cols), dtype=object)
percent_agreement = np.empty((rows, cols))

# Compute the mode and agreement level for each cell
for i in range(rows):
    for j in range(cols):
        answers = data_array[i, j, :] # All answers for a particular cell across dataframes
        unique_values, counts = np.unique(answers, return_counts=True)  # Unique values and their counts
        mode_index = np.argmax(counts) # Index of the mode (most frequent value)
        modes[i, j] = unique_values[mode_index] # Mode value
        percent_agreement[i, j] = (counts[mode_index] / len(dfs)) * 100 # Agreement as a percentage

# Convert modes and percent_agreement back to DataFrames for easier interpretation
df_modes = pd.DataFrame(modes, columns=dfs[0].columns, index=dfs[0].index)
# Convert to DataFrames
df_percent_agreement = pd.DataFrame(percent_agreement, index=dfs[0].index, columns=dfs[0].columns)

# Add column and row of means
df_percent_agreement['Mean'] = df_percent_agreement.mean(axis=1)
df_percent_agreement.loc['Mean'] = df_percent_agreement.mean(axis=0)
print(df_percent_agreement)
#-----------------------------
# Shannon Entropy (Diversity Index)
def shannon_entropy(values):
    _, counts = np.unique(values, return_counts=True) # values, counts
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Apply the entropy function along the last axis (runs)
entropy_values = np.apply_along_axis(shannon_entropy, 2, data_array)

# Convert to DataFrames
df_entropy = pd.DataFrame(entropy_values, index=dfs[0].index, columns=dfs[0].columns)

# Add column and row of means
df_entropy['Mean'] = df_entropy.mean(axis=1)
df_entropy.loc['Mean'] = df_entropy.mean(axis=0)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def figure_maker (number: int, 
                  
                  column_X: str,
                  order_X: List[str],
                  column_Y: str,
                  order_Y: List[str],

                  df_A: pd.DataFrame,
                  df_B: pd.DataFrame,

                  title: str, 
                  x_label: str,
                  y_label: str,
                  printing_name: str) -> None:
    
    """
    Function to make heat contingency tables and correlation tests.

    Args:
        number (int): Number of the figure.
        column_X (str): Name of A data column to be used.
        order_X (List[str]): A order of the categories.
        column_Y (str): Name of B data column to be used.
        df_top_A (Dataframe): A Dataframe with all data.
        order_Y (List[str]): B order of the categories.
        df_top_B (Dataframe): B Dataframe with all data.
        title (str): Figure title.
        x_label (str): Figure title.
        y_label (str): Figure title.
        printing_name (str): Name to be printed as the file name and label.

    Returns:
        No return.
    """

    #---------------------------------------------------------------------------------------------------
    # Creates a figure object with size 18x5 inches
    figure = plt.figure(number, figsize = (18, 5))
    gs = figure.add_gridspec(ncols = 2, nrows = 1)

    # Custom dark gray color
    custom_dark_gray = (0.2, 0.2, 0.2)

    #-------------------------------------------
    # Step 1: Create a contingency table (cross-tab)
    contingency_table = pd.crosstab(df_A[column_Y], df_A[column_X])
    contingency_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    # Reindex rows and columns according to the desired order
    contingency_table = contingency_table.reindex(index=order_Y, columns=order_X)
    # Create a formatted string version of the percentages with % symbol for annotation
    annot = contingency_table.map(lambda x: f'{x:.2f}%')

    print("\n")
    print("Contingency Table:")
    print(contingency_table)

    #-------------------------------------------
    # Create the main plot
    ax1 = figure.add_subplot(gs[0])
    #sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
    sns.heatmap(contingency_table, cmap='coolwarm', annot=annot, fmt="")

    # Set the title and labels
    ax1.set_title('Contingency Table Heatmap (All decades)')
    ax1.set_xlabel(f"{x_label}")
    ax1.set_ylabel(f"{y_label}")

    # Rotate and align the labels
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right') # Horizontal alignment (right aligns better for rotated labels)
        tick.set_va('top') # Vertical alignment to ensure no overlap

    #-------------------------------------------
    # Step 2: Perform Chi-Square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Results
    print("\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-Value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print(f"There IS a significant correlation between {column_X} and {column_Y}.")
    else:
        print(f"There is NO significant correlation between {column_X} and {column_Y}.")

    #---------------------------------------------------------------------------------------------------
    # Step 1: Create a contingency table (cross-tab)
    contingency_table_2000 = pd.crosstab(df_B[column_Y], df_B[column_X])
    contingency_table_2000 = contingency_table_2000.div(contingency_table_2000.sum(axis=1), axis=0) * 100
    # Reindex rows and columns according to the desired order
    contingency_table_2000 = contingency_table_2000.reindex(index=order_Y, columns=order_X)
    # Create a formatted string version of the percentages with % symbol for annotation
    annot_2000 = contingency_table_2000.map(lambda x: f'{x:.2f}%')

    print("\n")
    print("Contingency Table 2000s:")
    print(contingency_table_2000)

    #-------------------------------------------
    # Create the main plot
    ax2 = figure.add_subplot(gs[1])
    #sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
    sns.heatmap(contingency_table_2000, cmap='coolwarm', annot=annot_2000, fmt="")

    # Set the title and labels
    ax2.set_title('Contingency Table Heatmap (2000s, 2010s, 2020s)')
    ax2.set_xlabel(f"{x_label}")
    ax2.set_ylabel(f"{y_label}")

    # Rotate and align the labels
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right') # Horizontal alignment (right aligns better for rotated labels)
        tick.set_va('top') # Vertical alignment to ensure no overlap

    #-------------------------------------------
    # Step 2: Perform Chi-Square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table_2000)

    # Results
    print("\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-Value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print(f"There IS a significant correlation between {column_X} and {column_Y}.")
    else:
        print(f"There is NO significant correlation between {column_X} and {column_Y}.")

    # Save image-------------------------------------------
    plt.savefig(FIGURES / "00 {printing_name}.png", bbox_inches = 'tight')
    #plt.savefig(FIGURES / "00 {printing_name}.eps", transparent = True, bbox_inches = 'tight')
    # Transparence will be lost in .eps, save in .svg for transparences
    #plt.savefig(FIGURES / "00 {printing_name}.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
    plt.close(figure)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# Make figures
print("Making the figures...")
#---------------------------------------------------------------------------------------------------
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)

#---------------------------------------------------------------------------------------------------
# Figure 1 - Variation in Answers
print("  Making Variation in Answers...")

#-----------------------------------------
# Creates a figure object with size 14x14 inches
figure_1 = plt.figure(1, figsize = (14, 14))
gs = figure_1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_1.add_subplot(gs[0])

# Heatmap-------------------------------------------
# Set var_flag to the measure of variation in answers that you want:
# 1 Difference two by two of the answers
# 2 Percent Agreement (Mode Consistency)
# 3 Shannon Entropy (Diversity Index)
var_flag = 1

# Define tolerance for near-zero values
TOLERANCE = 1e-6

# Custom format function for decimal number show
def custom_format(val):
    # Treat values within the tolerance as zero
    if abs(val) < TOLERANCE:
        return 0
    elif abs(val) >= 100:
        return f"{val:.0f}" # No decimals for numbers >= 100
    elif abs(val) >= 10:
        return f"{val:.1f}" # One decimal for numbers >= 10 and < 100
    else:
        return f"{val:.2f}" # Two decimals for numbers < 10
        
#-----------------------------
# 1 Difference two by two of the answers
if var_flag == 1:
    sns.heatmap(df_mean_difference, 
                annot=df_mean_difference.map(custom_format), 
                cmap="coolwarm",
                fmt="", 
                cbar_kws={'label': 'Mean Difference Score'},
                annot_kws={"size": 8})

#-----------------------------
# 2 Percent Agreement (Mode Consistency)
elif var_flag == 2:
    sns.heatmap(df_percent_agreement, 
                annot=True, 
                cmap="coolwarm_r", # coolwarm_r: reversed coolwarm
                fmt=".3g",
                cbar_kws={'label': 'Percent agreement for the main answer'},
                annot_kws={"size": 8})

#-----------------------------
# 3 Shannon Entropy (Diversity Index)
elif var_flag == 3:
    sns.heatmap(df_entropy, 
                annot=df_entropy.map(custom_format), 
                cmap="coolwarm",
                fmt="", 
                cbar_kws={'label': 'Shannon index'},
                annot_kws={"size": 8})

#-----------------------------
# Add thicker lines to separate the last row and column

# Horizontal line above mean row
ax1.hlines(len(df_mean_difference) - 1, 
           *ax1.get_xlim(), 
           colors = custom_dark_gray, 
           linewidth=2.0)
# Vertical line before mean column
ax1.vlines(len(df_mean_difference.columns) - 1, 
           *ax1.get_ylim(), 
           colors = custom_dark_gray, 
           linewidth=2.0)

# Design-------------------------------------------
#ax1.set_xlabel("Questions", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Variability in Answers", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Move x-axis labels to the top
ax1.xaxis.tick_top()

# Rotate and align the labels
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('left')  # Horizontal alignment (left aligns better for rotated labels)
    tick.set_va('bottom')  # Vertical alignment to ensure no overlap

# Save image-------------------------------------------
if var_flag == 1:
    plt.savefig(FIGURES / "00 variation difference.png", bbox_inches = 'tight')
elif var_flag == 2:
    plt.savefig(FIGURES / "00 variation mode.png", bbox_inches = 'tight')
elif var_flag == 3:
    plt.savefig(FIGURES / "00 variation entropy.png", bbox_inches = 'tight')

plt.close(figure_1)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Author and protagonist gender
# Figure 2 - Author and protagonist gender
print("  Making author and protagonist gender...")

#------------------------------------------
# Creates a figure object with size 12x6 inches
figure_2 = plt.figure(2, figsize = (12, 6))
gs = figure_2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_genders = ['Male / Male',
                          'Male / Female',
                          'Male / Other',

                          'Male / Uncertain',
                          'Male / Not applicable',
                          
                          'Female / Male',
                          'Female / Female',
                          'Female / Other',

                          'Female / Uncertain',
                          'Female / Not applicable',
                          
                          'Other / All combined',
                          'Uncertain / All combined']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_genders = pd.DataFrame(columns=category_order_genders).astype(float)
category_percent_genders = pd.concat([all_categories_df_genders, category_percent_genders]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_genders = category_percent_genders[category_order_genders]

# Define custom colors for each category
custom_colors_genders = ['#385AC2',
                         '#5580D0',
                         '#6CACEB',

                         '#FFD700',
                         '#D3D3D3',
                         
                         '#AE305D',
                         '#CF5D5F',
                         '#E3937B',

                         '#FFD700',
                         '#D3D3D3',
                         
                         '#8B3FCF',
                         '#FFD700']
# Bar plot-------------------------------------------
category_percent_genders.plot(kind = 'bar',
                              stacked = True,
                              ax = ax1,
                              color = custom_colors_genders,
                              width = 1.0,
                              alpha = 1.0,
                              label = "genders")

ax1.text(17.57, 
         106, 
         r"Author / Protagonist", 
         fontsize = 10.7, 
         zorder = 1,)
         #color = custom_dark_gray)

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the author and the protagonist?", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Format the y-axis to show percentages
ax1.yaxis.set_major_formatter(PercentFormatter())

# Legend-------------------------------------------
# Get handles and labels
handles, labels = ax1.get_legend_handles_labels()

# Reverse the order
handles.reverse()
labels.reverse()

# Pass the reversed handles and labels to the legend
ax1.legend(handles, 
           labels, 
           bbox_to_anchor = (0.99, 0.00, 0.50, 0.95), 
           frameon = False, 
           labelspacing = 1.6,
           loc = 'center left')

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Dark lines-------------------------------------------
# Get number of bars (decades) from x-axis
n_decades = len(category_percent_genders.index)

# Define the boundaries (after which category the separator should be drawn)
boundaries = [4, 9, 11] # Indices between Male, Female, and Other/Uncertain
bar_width = 1.0

for b in boundaries:
    # Cumulative sum up to that boundary
    cumulative = category_percent_genders.iloc[:, :b+1].cumsum(axis=1)
    y_values = cumulative.iloc[:, -1].values # top per bar

    # Build step path
    xs = []
    ys = []
    for x, y in enumerate(y_values):
        # left edge
        xs.append(x - bar_width/2)
        ys.append(y)
        # right edge
        xs.append(x + bar_width/2)
        ys.append(y)

    # Now draw with step connections
    ax1.plot(xs, 
             ys, 
             color=custom_dark_gray, 
             linewidth = 1.2, 
             linestyle='solid')

    # Add vertical connectors between bars
    for i in range(len(y_values) - 1):
        ax1.plot(
            [i + bar_width/2, i + 1 - bar_width/2],
            [y_values[i], y_values[i+1]],
            color = custom_dark_gray,
            linewidth = 1.2,
            linestyle='solid'
        )

# Save image-------------------------------------------
plt.savefig(FIGURES / "00 author and protagonist gender.png", bbox_inches = 'tight')
#plt.savefig(FIGURES / "00 author and protagonist gender.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig(FIGURES / "00 author and protagonist gender.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_2)

#---------------------------------------------------------------------------------------------------
# Figure 3 - protagonist classes
print("  Making protagonist classes (All decades)...")

#------------------------------------------
color_map = {
    "Total": '#D3D3D3',
    "Yes": '#385AC2', 
    "No": '#AE305D', 
    "Uncertain": '#FFD700',
    "Not applicable": '#D3D3D3', 
    "Human": '#385AC2', 
    "Non-human": '#AE305D', 
    "Male": '#385AC2', 
    "Female": '#AE305D', 
    "Other": '#8B3FCF',
    "Good": '#385AC2', 
    "Leaning good": '#5580D0', 
    "Ambivalent": '#8B3FCF', 
    "Leaning bad": '#CF5D5F', 
    "Bad": '#AE305D', 
    "Non-moral": '#008000'
}

path = [
    px.Constant("Total"),
    "author gender",
    "16 protagonist",
    "17 protagonist nature", 
    "18 protagonist gender", 
    "19 protagonist is"
    ]

# List of column names
column_titles = [
    "Author",
    "Protagonist",
    "Nature",
    "Gender",
    "Moral"
]

# Create a dictionary to map each column title to its x-position.
# You will need to adjust these values to perfectly center the labels
# on your specific chart.
x_positions = {
    "Author": 0.25,
    "Protagonist": 0.42,
    "Nature": 0.58,
    "Gender": 0.75,
    "Moral": 0.91
}

#-----------------------------
figure_classes_all = px.icicle(
    df_top_AI,
    path=path,
    color=path[-1],
    color_discrete_map=color_map
)

# Create a list of text colors based on the label's value
text_colors = []
for label in color_map.keys():
    if label == "Not applicable":
        text_colors.append("black") # Use default color
    else:
        text_colors.append('white') # Use white for all other labels

figure_classes_all.update_traces(
    textinfo="label+percent entry",
    root_color="#D3D3D3",
    marker_colors=[color_map.get(label, "#CCCCCC") for label in figure_classes_all.data[0].labels],
    textfont=dict(color=text_colors)
)

#-----------------------------
# Create a list of annotations for each column title
annotations = []
for title in column_titles:
    annotations.append(
        dict(
            text=title,
            x=x_positions.get(title), # Get the x-position from the dictionary
            y=1.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    )

#-----------------------------
# Update layout with all changes
figure_classes_all.update_layout(
    #uniformtext=dict(minsize=5, mode='hide'),
    title_text="Authors and Types of Protagonist (all decades)",
    title_font_size=20,
    title_x=0.5,
    margin=dict(t=50, l=20, r=20, b=20),
    annotations=annotations
)

# Save image-------------------------------------------
figure_classes_all.write_image(FIGURES / "00 protagonist classes (all decades).png", scale=3)

#---------------------------------------------------------------------------------------------------
# Figure 4 - protagonist classes
print("  Making protagonist classes (2000s-2020s)...")

mask = df_top_AI['decade'] >= 2000
df_top_AI_2000 = df_top_AI[mask]

#-----------------------------
figure_classes_2000 = px.icicle(
    df_top_AI_2000,
    path=path,
    color=path[-1],
    color_discrete_map=color_map
)

# Create a list of text colors based on the label's value
text_colors = []
for label in color_map.keys():
    if label == "Not applicable":
        text_colors.append("black") # Use default color
    else:
        text_colors.append('white') # Use white for all other labels

figure_classes_2000.update_traces(
    textinfo="label+percent entry",
    root_color="#D3D3D3",
    marker_colors=[color_map.get(label, "#CCCCCC") for label in figure_classes_2000.data[0].labels],
    textfont=dict(color=text_colors)
)

#-----------------------------
# Create a list of annotations for each column title
annotations = []
for title in column_titles:
    annotations.append(
        dict(
            text=title,
            x=x_positions.get(title), # Get the x-position from the dictionary
            y=1.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    )

#-----------------------------
# Update layout with all changes
figure_classes_2000.update_layout(
    #uniformtext=dict(minsize=5, mode='hide'),
    title_text="Authors and Types of Protagonist (2000s-2020s)",
    title_font_size=20,
    title_x=0.5,
    margin=dict(t=50, l=20, r=20, b=20),
    annotations=annotations
)

# Save image-------------------------------------------
figure_classes_2000.write_image(FIGURES / "00 protagonist classes (2000s-2020s).png", scale=3)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# For the tests and more
#---------------------------------------------------------------------------------------------------
# Figure 5 - Author and protagonist heatmap
print("  Making author and protagonist heatmap...")

#-------------------------------------------
# Define the desired order for Protagonist Gender (x-axis)
order_X = [
    'Male',
    'Female',
    'Other',
    #'Uncertain',
    'Not applicable'
]

# Define the desired order for Author Gender (y-axis)
order_Y = [
    'Male',
    'Female', 
    'Other', 
    'Uncertain'
]

mask = df_top_AI['decade'] >= 2000
df_top_AI_2000 = df_top_AI[mask]

figure_maker (5,
              '18 protagonist gender',
              order_X,
              'author gender',
              order_Y,
              df_top_AI,
              df_top_AI_2000,
              "title", 
              "Protagonist Gender",
              "Author Gender",
              "author and protagonist heatmap")

#---------------------------------------------------------------------------------------------------
# Figure 6 - hypothesis test
print("  Making hypothesis test 1...")

#-------------------------------------------
# Define the desired order for x-axis
order_X = [
    "Utopic", 
    "Leaning utopic", 
    "Balanced", 
    "Leaning dystopic", 
    "Dystopic", 
    #"Uncertain"
    ]

# Define the desired order for y-axis
order_Y = [
    "Yes", 
    "Somewhat", 
    "No", 
    #"Uncertain"
    ]

figure_maker (5,
              '7 social political',
              order_X,
              '10 post apocalyptic',
              order_Y,
              df_top_AI,
              df_top_AI_2000,
              "title", 
              "Social-political scenario",
              "Is it post-apocalyptic?",
              "post apocalyptic and social political")

#---------------------------------------------------------------------------------------------------
# Figure 6 - hypotheses test
print("  Making hypotheses test 2...")

#-------------------------------------------
# Define the desired order for x-axis
order_X = [
    "Distant future", 
    "Far future", 
    "Near future", 
    "Present", 
    "Near past", 
    "Far past", 
    "Distant past", 
    "Multiple timelines", 
    #"Uncertain"
    ]

# Define the desired order for y-axis
order_Y = [
    "Yes", 
    "Somewhat", 
    "No", 
    #"Uncertain"
    ]

figure_maker (6,
              '4 time',
              order_X,
              '10 post apocalyptic',
              order_Y,
              df_top_AI,
              df_top_AI_2000,
              "title", 
              "When does the story take place?",
              "Is it post-apocalyptic?",
              "post apocalyptic and time")

#---------------------------------------------------------------------------------------------------
# Figure 7 - hypotheses test
print("  Making hypotheses test 3...")

#-------------------------------------------
# Define the desired order for x-axis
order_X = [
    "Very optimistic", 
    "Optimistic", 
    "Balanced", 
    "Pessimistic", 
    "Very pessimistic", 
    #"Uncertain"
    ]

# Define the desired order for y-axis
order_Y = [
    "Yes", 
    "Somewhat", 
    "No", 
    #"Uncertain"
    ]

figure_maker (7,
              '5 mood',
              order_X,
              '10 post apocalyptic',
              order_Y,
              df_top_AI,
              df_top_AI_2000,
              "title", 
              "What is the mood of the story?",
              "Is it post-apocalyptic?",
              "post apocalyptic and mood")

#---------------------------------------------------------------------------------------------------
# Figure 8 - Questions Cramer's V heatmap
print("  Making questions Cramer's V heatmap...")

def cramers_v(series_x, series_y):
    mask = series_x.notna() & series_y.notna()
    x = series_x[mask]
    y = series_y[mask]
    if x.empty:
        return np.nan
    table = pd.crosstab(x, y)
    if table.size == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    if denom == 0:
        return np.nan
    return np.sqrt(chi2 / denom)

# build matrix
n = len(column_order)
mat = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(i, n):
        v = cramers_v(df_top_AI[column_order[i]], df_top_AI[column_order[j]])
        mat[i, j] = v
        mat[j, i] = v

mat_df = pd.DataFrame(mat, index=column_order, columns=column_order)

# optional: reorder by hierarchical clustering to show blocks
# (rows are clustered by their pattern of associations)
try:
    D = pdist(mat_df.fillna(0).values, metric='euclidean')
    order = leaves_list(linkage(D, method='average'))
    mat_df = mat_df.iloc[order, order]
except Exception:
    pass

# plot heatmap with matplotlib (no seaborn)
data = mat_df.values
masked = np.ma.masked_invalid(data)

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(masked, interpolation='nearest', aspect='auto')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cramér's V")

ax.set_xticks(np.arange(len(mat_df.columns)))
ax.set_yticks(np.arange(len(mat_df.index)))
ax.set_xticklabels(mat_df.columns, rotation=90, fontsize=8)
ax.set_yticklabels(mat_df.index, fontsize=8)

# annotate values
for i in range(mat_df.shape[0]):
    for j in range(mat_df.shape[1]):
        if not np.isnan(data[i, j]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha='center', va='center', fontsize=6)

plt.tight_layout()
plt.savefig(FIGURES / "00 cramers v heatmap questions.png", bbox_inches = 'tight', dpi=300)
#plt.savefig(FIGURES / "00 cramers v heatmap questions.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig(FIGURES / "00 cramers v heatmap questions.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#---------------------------------------------------------------------------------------------------
# Figure 9 - Alternatives Cramer's V heatmap big
print("  Making alternatives Cramer's V heatmap big...")

# --- Step 1: one-hot encode all answers ---
encoded = pd.get_dummies(df_top_AI[column_order], prefix=column_order, dummy_na=False)

# --- Step 2: phi coefficient function for two binary variables ---
def phi(x, y):
    table = pd.crosstab(x, y)
    if table.shape != (2, 2):
        return np.nan
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    return np.sqrt(chi2 / n)

# --- Step 3: build full matrix ---
alts = encoded.columns
n = len(alts)
mat = np.zeros((n, n), dtype=float)

for i in range(n):
    for j in range(i, n):
        v = phi(encoded.iloc[:, i], encoded.iloc[:, j])
        mat[i, j] = v
        mat[j, i] = v

mat_df = pd.DataFrame(mat, index=alts, columns=alts)

# --- Step 4: reorder by clustering (optional, helps visualization) ---
try:
    D = pdist(mat_df.fillna(0).values, metric='euclidean')
    order = leaves_list(linkage(D, method='average'))
    mat_df = mat_df.iloc[order, order]
except Exception:
    pass

# --- Step 5: plot heatmap ---
fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(mat_df, aspect='auto', interpolation='nearest')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Phi coefficient")

ax.set_xticks(np.arange(len(mat_df.columns)))
ax.set_yticks(np.arange(len(mat_df.index)))
ax.set_xticklabels(mat_df.columns, rotation=90, fontsize=6)
ax.set_yticklabels(mat_df.index, fontsize=6)

plt.tight_layout()
plt.savefig(FIGURES / "00 cramers v heatmap alternatives.png", bbox_inches = 'tight', dpi=300)
#plt.savefig(FIGURES / "00 cramers v heatmap alternatives.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig(FIGURES / "00 cramers v heatmap alternatives.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#---------------------------------------------------------------------------------------------------
print("All done.")

# Show figures-------------------------------------------------------------------------------------------
# plt.show() # Too many figures (> 20), so better not to show them.