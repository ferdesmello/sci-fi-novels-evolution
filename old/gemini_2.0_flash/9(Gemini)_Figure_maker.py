"""
This script process all of the data gathered in the previous steps for Gemini and makes figures to present the data.

Modules:
    - pandas
    - matplotlib.pyplot
    - seaborn
    - scipy.stats
    - numpy
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
import numpy as np

#----------------------------------------------------------------------------------
print("Reading and processing tha data...")

# Read the data
df_filtered = pd.read_csv("../Data/Filtered/sci-fi_books_FILTERED.csv", sep=";", encoding="utf-8-sig")
df_top = pd.read_csv("../Data/Filtered/sci-fi_books_TOP_Wiki.csv", sep=";", encoding="utf-8-sig")
df_top_AI = pd.read_csv("sci-fi_books_AI_ANSWERS_Gemini.csv", sep=";", encoding="utf-8-sig")
df_top_AI_gender = pd.read_csv("sci-fi_books_AI_ANSWERS_GENDER_Gemini.csv", sep=";", encoding="utf-8-sig")

#print(df_top_AI.info())
#print(df.head())

#----------------------------------------------------------------------------------
# Exclude books of before 1860 (allmost none)

mask_all = df_filtered['decade'] >= 1860
df_filtered = df_filtered[mask_all]

mask_top = df_top['decade'] >= 1860
df_top = df_top[mask_top]

mask_top_AI = df_top_AI['decade'] >= 1860
df_top_AI = df_top_AI[mask_top_AI]

#----------------------------------------------------------------------------------
# Include author gender to data

# Create a dictionary from df_top_AI_gender
author_gender_dict = df_top_AI_gender.set_index('author')['gender'].to_dict()

# Map the 'gender' based on 'author' column from df_top_AI
df_top_AI['gender'] = df_top_AI['author'].map(author_gender_dict).fillna('Uncertain')

#----------------------------------------------------------------------------------
# For the boxplots

# Add a column to each DataFrame to label the dataset
df_filtered['dataset'] = 'Filtered sample'
df_top['dataset'] = 'Top sample'

# Concatenate dataframes
df_filtered_200 = pd.concat([df_filtered, df_top])

#----------------------------------------------------------------------------------
# General information of the FILTERED sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED books.")

book_per_decade = df_filtered['decade'].value_counts()
mean_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_1 = {'quantity': book_per_decade,
           'avr rate': mean_per_decade['rate'],
           'std rate': sdv_per_decade['rate'],
           'avr ratings': mean_per_decade['ratings'],
           'std ratings': sdv_per_decade['ratings']}
 
# Create DataFrame by passing Dictionary
df_all = pd.DataFrame(frame_1)
df_all = (df_all
          .reset_index(drop = False)
          .sort_values(by = ['decade'], ascending = True)
          .reset_index(drop = True))

#print(df_all.info())
#print(df_all)

#----------------------------------------------------------------------------------
# General information of the 200 PER DECADE sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE books.")

book_per_decade_200 = df_top['decade'].value_counts()
mean_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade_200.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_2 = {'quantity': book_per_decade_200,
           'avr rate': mean_per_decade_200['rate'],
           'std rate': sdv_per_decade_200['rate'],
           'avr ratings': mean_per_decade_200['ratings'],
           'std ratings': sdv_per_decade_200['ratings']}
 
# Create DataFrame by passing Dictionary
df_all_200 = pd.DataFrame(frame_2)
df_all_200 = (df_all_200
              .reset_index(drop = False)
              .sort_values(by = ['decade'], ascending = True)
              .reset_index(drop = True))

# Part in a series of books------------------------------------------
# Count the occurrences of each category per decade
category_counts_series = pd.crosstab(df_top['decade'], df_top['series'])
# Normalize the counts to get percentages
category_percent_series = category_counts_series.div(category_counts_series.sum(axis = 1), axis = 0) * 100

# Author gender------------------------------------------
# Count the occurrences of each category per decade
category_counts_gender = pd.crosstab(df_top_AI['decade'], df_top_AI['gender'])
# Normalize the counts to get percentages
category_percent_gender = category_counts_gender.div(category_counts_gender.sum(axis = 1), axis = 0) * 100

# (very) hard author gender------------------------------------------
# Define the condition to select rows
mask_1 = df_top_AI['1 soft hard'] == "Hard"
mask_2 = df_top_AI['1 soft hard'] == "Very hard"
# Filter the dataframe to exclude those rows
df_top_AI_gender_b = df_top_AI[mask_1 | mask_2]
# Count the occurrences of each category per decade
category_counts_gender_b = pd.crosstab(df_top_AI_gender_b['decade'], df_top_AI_gender_b['gender'])
# Normalize the counts to get percentages
category_percent_gender_b = category_counts_gender_b.div(category_counts_gender_b.sum(axis = 1), axis = 0) * 100

#----------------------------------------------------------------------------------
# Process for the questions/answers

# 1 soft hard------------------------------------------
print(df_top_AI['1 soft hard'].value_counts())
# Count the occurrences of each category per decade
category_counts_1 = pd.crosstab(df_top_AI['decade'], df_top_AI['1 soft hard'])
# Normalize the counts to get percentages
category_percent_1 = category_counts_1.div(category_counts_1.sum(axis = 1), axis = 0) * 100

# 2 light heavy------------------------------------------
print(df_top_AI['2 light heavy'].value_counts())
# Count the occurrences of each category per decade
category_counts_2 = pd.crosstab(df_top_AI['decade'], df_top_AI['2 light heavy'])
# Normalize the counts to get percentages
category_percent_2 = category_counts_2.div(category_counts_2.sum(axis = 1), axis = 0) * 100

# 3 time------------------------------------------
print(df_top_AI['3 time'].value_counts())
# Count the occurrences of each category per decade
category_counts_3 = pd.crosstab(df_top_AI['decade'], df_top_AI['3 time'])
# Normalize the counts to get percentages
category_percent_3 = category_counts_3.div(category_counts_3.sum(axis = 1), axis = 0) * 100

# 4 mood------------------------------------------
print(df_top_AI['4 mood'].value_counts())
# Count the occurrences of each category per decade
category_counts_4 = pd.crosstab(df_top_AI['decade'], df_top_AI['4 mood'])
# Normalize the counts to get percentages
category_percent_4 = category_counts_4.div(category_counts_4.sum(axis = 1), axis = 0) * 100

# 5 social political------------------------------------------
print(df_top_AI['5 social political'].value_counts())
# Count the occurrences of each category per decade
category_counts_5 = pd.crosstab(df_top_AI['decade'], df_top_AI['5 social political'])
# Normalize the counts to get percentages
category_percent_5 = category_counts_5.div(category_counts_5.sum(axis = 1), axis = 0) * 100

# 6 on Earth------------------------------------------
print(df_top_AI['6 on Earth'].value_counts())
# Count the occurrences of each category per decade
category_counts_6 = pd.crosstab(df_top_AI['decade'], df_top_AI['6 on Earth'])
# Normalize the counts to get percentages
category_percent_6 = category_counts_6.div(category_counts_6.sum(axis = 1), axis = 0) * 100

# 7 post apocalyptic------------------------------------------
print(df_top_AI['7 post apocalyptic'].value_counts())
# Count the occurrences of each category per decade
category_counts_7 = pd.crosstab(df_top_AI['decade'], df_top_AI['7 post apocalyptic'])
# Normalize the counts to get percentages
category_percent_7 = category_counts_7.div(category_counts_7.sum(axis = 1), axis = 0) * 100

# 8 aliens------------------------------------------
print(df_top_AI['8 aliens'].value_counts())
# Count the occurrences of each category per decade
category_counts_8 = pd.crosstab(df_top_AI['decade'], df_top_AI['8 aliens'])
# Normalize the counts to get percentages
category_percent_8 = category_counts_8.div(category_counts_8.sum(axis = 1), axis = 0) * 100

# 9a aliens are------------------------------------------
print(df_top_AI['9 aliens are'].value_counts())
# Count the occurrences of each category per decade
category_counts_9a = pd.crosstab(df_top_AI['decade'], df_top_AI['9 aliens are'])
# Normalize the counts to get percentages
category_percent_9a = category_counts_9a.div(category_counts_9a.sum(axis = 1), axis = 0) * 100

# 9b aliens are------------------------------------------
# Define the condition to exclude rows where the column has a specific value
condition_to_exclude = "Not applicable"
# Filter the dataframe to exclude those rows
df_top_AI_9b = df_top_AI[df_top_AI['9 aliens are'] != condition_to_exclude]
# Count the occurrences of each category per decade
category_counts_9b = pd.crosstab(df_top_AI_9b['decade'], df_top_AI_9b['9 aliens are'])
# Normalize the counts to get percentages
category_percent_9b = category_counts_9b.div(category_counts_9b.sum(axis = 1), axis = 0) * 100

# 10 robots and AI------------------------------------------
print(df_top_AI['10 robots and AI'].value_counts())
# Count the occurrences of each category per decade
category_counts_10 = pd.crosstab(df_top_AI['decade'], df_top_AI['10 robots and AI'])
# Normalize the counts to get percentages
category_percent_10 = category_counts_10.div(category_counts_10.sum(axis = 1), axis = 0) * 100

# 11a robots and AI are------------------------------------------
print(df_top_AI['11 robots and AI are'].value_counts())
# Count the occurrences of each category per decade
category_counts_11a = pd.crosstab(df_top_AI['decade'], df_top_AI['11 robots and AI are'])
# Normalize the counts to get percentages
category_percent_11a = category_counts_11a.div(category_counts_11a.sum(axis = 1), axis = 0) * 100

# 11b robots and AI are------------------------------------------
# Define the condition to exclude rows where the column has a specific value
condition_to_exclude = "Not applicable"
# Filter the dataframe to exclude those rows
df_top_AI_11b = df_top_AI[df_top_AI['11 robots and AI are'] != condition_to_exclude]
# Count the occurrences of each category per decade
category_counts_11b = pd.crosstab(df_top_AI_11b['decade'], df_top_AI_11b['11 robots and AI are'])
# Normalize the counts to get percentages
category_percent_11b = category_counts_11b.div(category_counts_11b.sum(axis = 1), axis = 0) * 100

# 12 protagonist------------------------------------------
print(df_top_AI['12 protagonist'].value_counts())
# Count the occurrences of each category per decade
category_counts_12 = pd.crosstab(df_top_AI['decade'], df_top_AI['12 protagonist'])
# Normalize the counts to get percentages
category_percent_12 = category_counts_12.div(category_counts_12.sum(axis = 1), axis = 0) * 100

# 13a protagonist is------------------------------------------
print(df_top_AI['13 protagonist is'].value_counts())
# Count the occurrences of each category per decade
category_counts_13a = pd.crosstab(df_top_AI['decade'], df_top_AI['13 protagonist is'])
# Normalize the counts to get percentages
category_percent_13a = category_counts_13a.div(category_counts_13a.sum(axis = 1), axis = 0) * 100

# 13b protagonist is------------------------------------------
# Define the condition to exclude rows where the column has a specific value
condition_to_exclude = "Not applicable"
# Filter the dataframe to exclude those rows
df_top_AI_13b = df_top_AI[df_top_AI['13 protagonist is'] != condition_to_exclude]
# Count the occurrences of each category per decade
category_counts_13b = pd.crosstab(df_top_AI_13b['decade'], df_top_AI_13b['13 protagonist is'])
# Normalize the counts to get percentages
category_percent_13b = category_counts_13b.div(category_counts_13b.sum(axis = 1), axis = 0) * 100

# 13c protagonist is (only hard and very hard sci-fi)------------------------------------------
# Define the condition to select rows
mask_1 = df_top_AI['1 soft hard'] == "Hard"
mask_2 = df_top_AI['1 soft hard'] == "Very hard"
# Filter the dataframe to exclude those rows
df_top_AI_13c = df_top_AI[mask_1 | mask_2]
# Count the occurrences of each category per decade
category_counts_13c = pd.crosstab(df_top_AI_13c['decade'], df_top_AI_13c['13 protagonist is'])
# Normalize the counts to get percentages
category_percent_13c = category_counts_13c.div(category_counts_13c.sum(axis = 1), axis = 0) * 100

# 14 virtual------------------------------------------
print(df_top_AI['14 virtual'].value_counts())
# Count the occurrences of each category per decade
category_counts_14 = pd.crosstab(df_top_AI['decade'], df_top_AI['14 virtual'])
# Normalize the counts to get percentages
category_percent_14 = category_counts_14.div(category_counts_14.sum(axis = 1), axis = 0) * 100

# 15 tech and science------------------------------------------
print(df_top_AI['15 tech and science'].value_counts())
# Count the occurrences of each category per decade
category_counts_15 = pd.crosstab(df_top_AI['decade'], df_top_AI['15 tech and science'])
# Normalize the counts to get percentages
category_percent_15 = category_counts_15.div(category_counts_15.sum(axis = 1), axis = 0) * 100

# 16 social issues------------------------------------------
print(df_top_AI['16 social issues'].value_counts())
# Count the occurrences of each category per decade
category_counts_16 = pd.crosstab(df_top_AI['decade'], df_top_AI['16 social issues'])
# Normalize the counts to get percentages
category_percent_16 = category_counts_16.div(category_counts_16.sum(axis = 1), axis = 0) * 100

# 17 enviromental------------------------------------------
print(df_top_AI['17 enviromental'].value_counts())
# Count the occurrences of each category per decade
category_counts_17 = pd.crosstab(df_top_AI['decade'], df_top_AI['17 enviromental'])
# Normalize the counts to get percentages
category_percent_17 = category_counts_17.div(category_counts_17.sum(axis = 1), axis = 0) * 100

#----------------------------------------------------------------------------------
# Process for the complex figures
#----------------------------------------------------------------------------------
# Author and protagonist gender

df_top_AI_new = df_top_AI.copy()
df_top_AI_new["genders"] = df_top_AI_new['gender'] + " / " + df_top_AI_new['13 protagonist is']

# Count the occurrences of each category per decade
category_counts_genders = pd.crosstab(df_top_AI_new['decade'], df_top_AI_new["genders"])

# Define a list of the specific 'Other' columns you want to combine
other_columns_to_combine = ['Other / Male',
                            'Other / Female',
                            'Other / Other',
                            'Other / Non-human',
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

# Normalize the counts to get percentages
category_percent_genders = category_counts_genders.div(category_counts_genders.sum(axis=1), axis=0) * 100

#print(category_percent_genders)
print("\ngenders",df_top_AI_new["genders"].unique())
#print("\n")

#----------------------------------------------------------------------------------
# Variation in Answers

# File names in the normally ordered alternatives
file_names = ["sci-fi_books_AI_ANSWERS_TEST_Gemini_01.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_02.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_03.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_04.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_05.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_06.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_07.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_08.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_09.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_10.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_11.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_12.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_13.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_14.csv",
              "sci-fi_books_AI_ANSWERS_TEST_Gemini_15.csv"]

column_order = ['1 soft hard',
                '2 light heavy',
                '3 time',
                '4 mood',
                '5 social political',
                '6 on Earth',
                '7 post apocalyptic',
                '8 aliens',
                '9 aliens are',
                '10 robots and AI',
                '11 robots and AI are',
                '12 protagonist',
                '13 protagonist is',
                '14 virtual',
                '15 tech and science',
                '16 social issues',
                '17 enviromental',]

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

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Make figures
print("Making the figures...")
#----------------------------------------------------------------------------------
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)
#----------------------------------------------------------------------------------
# Figure 1, Quantities
print("  Making book counts...")

# Creates a figure object with size 12x6 inches
figure_c1 = plt.figure(1, figsize = (12, 6))
gs = figure_c1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c1.add_subplot(gs[0])

# Bar plot all sample
bars = ax1.bar(x = df_all['decade'],
               height = df_all['quantity'],
               width = 9, 
               align = 'center',
               color = "#385AC2",
               alpha = 1.0,
               edgecolor = custom_dark_gray,
               linewidth = 0.0,
               label = "Filtered sample")

# Bar plot top 200
ax1.bar(x = df_all_200['decade'],
        height = df_all_200['quantity'], 
        width = 9, 
        align = 'center', 
        color = "none",
        edgecolor = "#AE305D", 
        hatch = "////",
        linewidth = 0.0,
        label = "Top sample")

# Add the count labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        color=custom_dark_gray
    )

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Quantity", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Number of Sci-fi Books per Decade In The Samples", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Legend-------------------------------------------
ax1.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = False, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
#ax1.spines['right'].set_color(custom_dark_gray)
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Get all decades between the minimum and maximum
all_decades = np.arange(df_all['decade'].min(), df_all['decade'].max() + 10, 10)
# Set x-ticks to show each decade
plt.xticks(all_decades, rotation=90)

# Save image-------------------------------------------
plt.savefig("00 Quantities Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 Quantities Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 Quantities Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 2, Quantities
print("  Making rate and ratings...")

# Creates a figure object with size 14x8 inches
figure_c2 = plt.figure(2, figsize = (14, 8))
gs = figure_c2.add_gridspec(ncols = 2, nrows = 1)

#figure_c2.subplots_adjust(hspace = 0.5)

#-----------------------------------------
# Create the main plot
ax1 = figure_c2.add_subplot(gs[0])

# Specify custom colors for each dataset
custom_palette = {'Filtered sample': '#385AC2',
                  'Top sample': '#AE305D'}

# Create the boxplot with hue
sns.boxplot(x = 'decade', 
            y = 'rate', 
            hue = 'dataset', 
            data = df_filtered_200, 
            palette = custom_palette,
            #width = 0.8,
            #gap = 0.3,
            fliersize = 2.0,
            fill = True,
            linecolor = custom_dark_gray)

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Distribution of Average Rate per Decade", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)
ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 0.5)

# Legend-------------------------------------------
ax1.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax1.spines['right'].set_color(custom_dark_gray)
#ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

#----------------------------------------------------------------------------------
# Create the main plot
ax2 = figure_c2.add_subplot(gs[1])

# Specify custom colors for each dataset
custom_palette = {'Filtered sample': '#385AC2',
                  'Top sample': '#AE305D'}

# Create the boxplot with hue
sns.boxplot(x = 'decade', 
            y = 'ratings', 
            hue = 'dataset', 
            data = df_filtered_200, 
            palette = custom_palette,
            #width = 0.8,
            #gap = 0.3,
            fliersize = 2.0,
            fill = True,
            linecolor = custom_dark_gray,
            log_scale = True)

# Design-------------------------------------------
ax2.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax2.set_ylabel("Number of ratings", fontsize = 12, color = custom_dark_gray)
ax2.set_title("Distribution of Number of Ratings per Decade", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax2.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)
ax2.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 0.5)

# Legend-------------------------------------------
ax2.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)

#ax2.set_yscale('log')

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax2.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax2.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax2.tick_params(axis = 'both', colors = custom_dark_gray)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax2.spines['right'].set_color(custom_dark_gray)
#ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)

# Save image-------------------------------------------
plt.savefig("00 Rates and Ratings Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 Rates and Ratings Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 Rates and Ratings Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 3, Series
print("  Making series...")

# Creates a figure object with size 12x6 inches
figure_c3 = plt.figure(3, figsize = (12, 6))
gs = figure_c3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c3.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_series = ['yes',
                         'no']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_series = category_percent_series[category_order_series]

# Define custom colors for each category
custom_colors_series = ['#AE305D', 
                        '#385AC2']

# Bar plot-------------------------------------------
category_percent_series.plot(kind = 'bar',
                             stacked = True,
                             ax = ax1,
                             color = custom_colors_series,
                             width = 1.0,
                             alpha = 1.0,
                             label = "series")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the novel part of a series?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Save image-------------------------------------------
plt.savefig("00 series Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 series Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 series Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 4, author gender
print("  Making author gender...")

# Creates a figure object with size 12x6 inches
figure_c4 = plt.figure(4, figsize = (12, 6))
gs = figure_c4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c4.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_gender = ['Male',
                         'Female',
                         'Other',
                         'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_gender = pd.DataFrame(columns=category_order_gender).astype(float)
category_percent_gender = pd.concat([all_categories_df_gender, category_percent_gender]).convert_dtypes().convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_gender = category_percent_gender[category_order_gender]

# Define custom colors for each category
custom_colors_gender = ['#385AC2',
                        '#AE305D',
                        '#8B3FCF',
                        '#FFD700']

# Bar plot-------------------------------------------
category_percent_gender.plot(kind = 'bar',
                             stacked = True,
                             ax = ax1,
                             color = custom_colors_gender,
                             width = 1.0,
                             alpha = 1.0,
                             label = "gender")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the author?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Save image-------------------------------------------
plt.savefig("00 author gender Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 author gender Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 author gender Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 5, author gender soft hard
print("  Making author gender soft hard...")

# Creates a figure object with size 12x6 inches
figure_c5 = plt.figure(5, figsize = (12, 6))
gs = figure_c5.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c5.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_gender_b = ['Male',
                           'Female',
                           'Other',
                           'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_gender_b = pd.DataFrame(columns=category_order_gender_b).astype(float)
category_percent_gender_b = pd.concat([all_categories_df_gender_b, category_percent_gender_b]).convert_dtypes().convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_gender_b = category_percent_gender_b[category_order_gender_b]

# Define custom colors for each category
custom_colors_gender_b = ['#385AC2',
                          '#AE305D',
                          '#8B3FCF',
                          '#FFD700']

# Bar plot-------------------------------------------
category_percent_gender_b.plot(kind = 'bar',
                                stacked = True,
                                ax = ax1,
                                color = custom_colors_gender_b,
                                width = 1.0,
                                alpha = 1.0,
                                label = "gender")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the author for hard and very hard sci-fi?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Save image-------------------------------------------
plt.savefig("00 author gender soft hard Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 author gender soft hard Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 author gender soft hard Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# figures for the questions/answers
#----------------------------------------------------------------------------------
# Figure 6 - 1 soft hard
print("  Making 1 soft hard...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure1 = plt.figure(6, figsize = (12, 6))
gs = figure1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure1.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_1 = ['Very hard',
                    'Hard',
                    'Mixed', 
                    'Soft',
                    'Very soft',

                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_1 = pd.DataFrame(columns=category_order_1).astype(float)
category_percent_1 = pd.concat([all_categories_df_1, category_percent_1]).convert_dtypes().convert_dtypes().fillna(0.0)

category_percent_1 = category_percent_1.reindex(columns=category_order_1)
# Reorder the columns in the DataFrame according to the desired category order
category_percent_1 = category_percent_1[category_order_1]

# Define custom colors for each category
custom_colors_1 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_1.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_1,
                        width = 1.0,
                        alpha = 1.0,
                        label = "1 soft hard")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the book considered more soft or hard sci-fi?", fontsize = 14, color = custom_dark_gray)
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
           labelspacing = 4.5,
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
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Save image-------------------------------------------
plt.savefig("01 soft hard Gemini.png", bbox_inches = 'tight')
#plt.savefig("01 soft hard Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("01 soft hard Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 7 - 1 light heavy
print("  Making 2 light heavy...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure2 = plt.figure(7, figsize = (12, 6))
gs = figure2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_2 = ['Very heavy',
                    'Heavy',
                    'Balanced', 
                    'Light',
                    'Very light',

                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_2 = pd.DataFrame(columns=category_order_2).astype(float)
category_percent_2 = pd.concat([all_categories_df_2, category_percent_2]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_2 = category_percent_2[category_order_2]

# Define custom colors for each category
custom_colors_2 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_2.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_2,
                        width = 1.0,
                        alpha = 1.0,
                        label = "2 light heavy")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the book considered more of a light or heavy reading experience?", fontsize = 14, color = custom_dark_gray)
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
           labelspacing = 4.5,
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
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Save image-------------------------------------------
plt.savefig("02 light heavy Gemini.png", bbox_inches = 'tight')
#plt.savefig("02 light heavy Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("02 light heavy Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 8 - 3 time
print("  Making 3 time...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure3 = plt.figure(8, figsize = (12, 6))
gs = figure3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure3.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_3 = ['Distant past',
                    'Far past',
                    'Near past',
                    'Present',
                    'Near future',
                    'Far future',
                    'Distant future',

                    'Multiple timelines',
                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_3 = pd.DataFrame(columns=category_order_3).astype(float)
category_percent_3 = pd.concat([all_categories_df_3, category_percent_3]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_3 = category_percent_3[category_order_3]

# Define custom colors for each category
custom_colors_3 = ['#AE305D', # Distant past
                   '#CF5D5F',
                   '#E3937B',
                   '#8B3FCF',
                   '#6CACEB',
                   '#5580D0',
                   '#385AC2', # Distant future

                   '#008000',     # Multiple timelines
                   '#FFD700'] # Uncertain

# Bar plot-------------------------------------------
category_percent_3.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_3,
                        width = 1.0,
                        alpha = 1.0,
                        label = "3 time")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("When does most of the story take place in relation to the year the book was published?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 2.8,
           loc='center left')

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

# Save image-------------------------------------------
plt.savefig("03 time Gemini.png", bbox_inches = 'tight')
#plt.savefig("03 time Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("03 time Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 9 - 4 mood
print("  Making 4 mood...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure4 = plt.figure(9, figsize = (12, 6))
gs = figure4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure4.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_4 = ['Very pessimistic',
                    'Pessimistic',
                    'Balanced',
                    'Optimistic',
                    'Very optimistic',

                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_4 = pd.DataFrame(columns=category_order_4).astype(float)
category_percent_4 = pd.concat([all_categories_df_4, category_percent_4]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_4 = category_percent_4[category_order_4]

# Define custom colors for each category
custom_colors_4 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_4.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_4,
                        width = 1.0,
                        alpha = 1.0,
                        label = "4 mood")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the mood of the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.5,
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

# Save image-------------------------------------------
plt.savefig("04 mood Gemini.png", bbox_inches = 'tight')
#plt.savefig("04 mood Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("04 mood Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 10 - 5 social political
print("  Making 5 social political...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure5 = plt.figure(10, figsize = (12, 6))
gs = figure5.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure5.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_5 = ['Dystopic',
                    'Leaning dystopic',
                    'Balanced',
                    'Leaning utopic',
                    'Utopic',

                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_5 = pd.DataFrame(columns=category_order_5).astype(float)
category_percent_5 = pd.concat([all_categories_df_5, category_percent_5]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_5 = category_percent_5[category_order_5]

# Define custom colors for each category
custom_colors_5 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_5.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_5,
                        width = 1.0,
                        alpha = 1.0,
                        label = "5 social political")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the social political scenario depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.5,
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

# Save image-------------------------------------------
plt.savefig("05 social political Gemini.png", bbox_inches = 'tight')
#plt.savefig("05 social political Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("05 social political Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 11 - 6 on Earth
print("  Making 6 on Earth...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure6 = plt.figure(11, figsize = (12, 6))
gs = figure6.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure6.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_6 = ['Yes',
                    'No',
                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_6 = pd.DataFrame(columns=category_order_6).astype(float)
category_percent_6 = pd.concat([all_categories_df_6, category_percent_6]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_6 = category_percent_6[category_order_6]

# Define custom colors for each category
custom_colors_6 = ['#385AC2',
                   '#AE305D',
                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_6.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_6,
                        width = 1.0,
                        alpha = 1.0,
                        label = "6 on Earth")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is most of the story set on Earth?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Save image-------------------------------------------
plt.savefig("06 on Earth Gemini.png", bbox_inches = 'tight')
#plt.savefig("06 on Earth Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("06 on Earth Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 12 - 7 post apocalyptic
print("  Making 7 post apocalyptic...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure7 = plt.figure(12, figsize = (12, 6))
gs = figure7.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure7.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_7 = ['Yes',
                    'Somewhat',
                    'No',
                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_7 = pd.DataFrame(columns=category_order_7).astype(float)
category_percent_7 = pd.concat([all_categories_df_7, category_percent_7]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_7 = category_percent_7[category_order_7]

# Define custom colors for each category
custom_colors_7 = ['#AE305D',
                   '#8B3FCF',
                   '#385AC2',
                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_7.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_7,
                        width = 1.0,
                        alpha = 1.0,
                        label = "7 post apocalyptic")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the story set in a post-apocalyptic world (after a civilization-collapsing event)?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Save image-------------------------------------------
plt.savefig("07 post apocalyptic Gemini.png", bbox_inches = 'tight')
#plt.savefig("07 post apocalyptic Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("07 post apocalyptic Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 13 - 8 aliens
print("  Making 8 aliens...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure8 = plt.figure(13, figsize = (12, 6))
gs = figure8.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure8.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_8 = ['Yes',
                    'No',
                    'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_8 = pd.DataFrame(columns=category_order_8).astype(float)
category_percent_8 = pd.concat([all_categories_df_8, category_percent_8]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_8 = category_percent_8[category_order_8]

# Define custom colors for each category
custom_colors_8 = ['#AE305D',
                   '#385AC2',
                   '#FFD700']

# Bar plot-------------------------------------------
category_percent_8.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_8,
                        width = 1.0,
                        alpha = 1.0,
                        label = "8 aliens")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions or mentions of non-terrestrial life forms \n(e.g., aliens, extraterrestrial organisms, creatures not originating from Earth, even if non-sentient) \nor alien technology in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Save image-------------------------------------------
plt.savefig("08 aliens Gemini.png", bbox_inches = 'tight')
#plt.savefig("08 aliens Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("08 aliens Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 14 - 9a aliens are
print("  Making 9a aliens are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure9a = plt.figure(14, figsize = (12, 6))
gs = figure9a.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure9a.add_subplot(gs[0])

# Define the desired order of the categories
category_order_9a = ['Bad', 
                     'Leaning bad',
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain',
                     'Not applicable']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_9a = pd.DataFrame(columns=category_order_9a).astype(float)
category_percent_9a = pd.concat([all_categories_df_9a, category_percent_9a]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_9a = category_percent_9a[category_order_9a]

# Define custom colors for each category
custom_colors_9a = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    '#FFD700',
                    '#D3D3D3']

# Bar plot-------------------------------------------
category_percent_9a.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_9a,
                         width = 1.0,
                         alpha = 1.0,
                         label = "9a aliens are")

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
           labelspacing = 4.0,
           loc = 'center left')

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the non-terrestrial life forms generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Format the y-axis to show percentages
ax1.yaxis.set_major_formatter(PercentFormatter())

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

# Save image-------------------------------------------
plt.savefig("09a aliens are Gemini.png", bbox_inches = 'tight')
#plt.savefig("09a aliens are Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("09a aliens are Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 15 - 9b aliens are
print("  Making 9b aliens are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure9b = plt.figure(15, figsize = (12, 6))
gs = figure9b.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure9b.add_subplot(gs[0])

# Define the desired order of the categories
category_order_9b = ['Bad', 
                     'Leaning bad',
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_9b = pd.DataFrame(columns=category_order_9b).astype(float)
category_percent_9b = pd.concat([all_categories_df_9b, category_percent_9b]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_9b = category_percent_9b[category_order_9b]

# Define custom colors for each category
custom_colors_9b = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_9b.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_9b,
                         width = 1.0,
                         alpha = 1.0,
                         label = "9b aliens are")

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
           labelspacing = 5.0,
           loc = 'center left')

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the non-terrestrial life forms generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Format the y-axis to show percentages
ax1.yaxis.set_major_formatter(PercentFormatter())

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

# Save image-------------------------------------------
plt.savefig("09b aliens are Gemini.png", bbox_inches = 'tight')
#plt.savefig("09b aliens are Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("09b aliens are Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 16 - 10 robots and AI
print("  Making 10 robots and AI...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure10 = plt.figure(16, figsize = (12, 6))
gs = figure10.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure10.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_10 = ['Yes',
                     'No',
                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_10 = pd.DataFrame(columns=category_order_10).astype(float)
category_percent_10 = pd.concat([all_categories_df_10, category_percent_10]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_10 = category_percent_10[category_order_10]

# Define custom colors for each category
custom_colors_10 = ['#AE305D',
                    '#385AC2',
                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_10.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_10,
                         width = 1.0,
                         alpha = 1.0,
                         label = "10 robots and AI")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions of robots or artificial intelligences in the story? \n(just automatic systems, advanced technology, or programs do not count)", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Save image-------------------------------------------
plt.savefig("10 robots and AI Gemini.png", bbox_inches = 'tight')
#plt.savefig("10 robots and AI Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("10 robots and AI Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 17 - 11a robots and AI are
print("  Making 11a robots and AI are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure11a = plt.figure(17, figsize = (12, 6))
gs = figure11a.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure11a.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_11a = ['Bad', 
                      'Leaning bad', 
                      'Ambivalent', 
                      'Leaning good',
                      'Good',
                      
                      'Uncertain',
                      'Not applicable']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_11a = pd.DataFrame(columns=category_order_11a).astype(float)
category_percent_11a = pd.concat([all_categories_df_11a, category_percent_11a]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_11a = category_percent_11a[category_order_11a]

# Define custom colors for each category
custom_colors_11a = ['#AE305D',
                     '#CF5D5F',
                     '#8B3FCF',
                     '#5580D0',
                     '#385AC2',
                     
                     '#FFD700',
                     '#D3D3D3']

# Bar plot-------------------------------------------
category_percent_11a.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors_11a,
                          width = 1.0,
                          alpha = 1.0,
                          label = "11a robots and AI are")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the robots or artificial intelligences generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.0,
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

# Save image-------------------------------------------
plt.savefig("11a robots and AI are Gemini.png", bbox_inches = 'tight')
#plt.savefig("11a robots and AI are Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("11a robots and AI are Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 18 - 11b robots and AI are
print("  Making 11b robots and AI are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure11b = plt.figure(18, figsize = (12, 6))
gs = figure11b.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure11b.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_11b = ['Bad', 
                      'Leaning bad', 
                      'Ambivalent', 
                      'Leaning good',
                      'Good',
                      
                      'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_11b = pd.DataFrame(columns=category_order_11b).astype(float)
category_percent_11b = pd.concat([all_categories_df_11b, category_percent_11b]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_11b = category_percent_11b[category_order_11b]

# Define custom colors for each category
custom_colors_11b = ['#AE305D',
                     '#CF5D5F',
                     '#8B3FCF',
                     '#5580D0',
                     '#385AC2',
                     
                     '#FFD700']

# Bar plot-------------------------------------------
category_percent_11b.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors_11b,
                          width = 1.0,
                          alpha = 1.0,
                          label = "11b robots and AI are")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the robots or artificial intelligences generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 5.0,
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

# Save image-------------------------------------------
plt.savefig("11b robots and AI are Gemini.png", bbox_inches = 'tight')
#plt.savefig("11b robots and AI are Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("11b robots and AI are Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 19 - 12 protagonist
print("  Making 12 protagonist...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure12 = plt.figure(19, figsize = (12, 6))
gs = figure12.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure12.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_12 = ['Yes', 
                     'No',
                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_12 = pd.DataFrame(columns=category_order_12).astype(float)
category_percent_12 = pd.concat([all_categories_df_12, category_percent_12]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_12 = category_percent_12[category_order_12]

# Define custom colors for each category
custom_colors_12 = ['#AE305D',
                    '#385AC2',
                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_12.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_12,
                         width = 1.0,
                         alpha = 1.0,
                         label = "12 protagonist")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is there a single protagonist or main character?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Save image-------------------------------------------
plt.savefig("12 protagonist Gemini.png", bbox_inches = 'tight')
#plt.savefig("12 protagonist Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("12 protagonist Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 20 - 13a protagonist is
print("  Making 13a protagonist is...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure13a = plt.figure(20, figsize = (12, 6))
gs = figure13a.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure13a.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_13a = ['Male', 
                      'Female',
                      
                      'Other',
                      'Non-human',
                      'Uncertain',
                      
                      'Not applicable']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_13a = pd.DataFrame(columns=category_order_13a).astype(float)
category_percent_13a = pd.concat([all_categories_df_13a, category_percent_13a]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_13a = category_percent_13a[category_order_13a]

# Define custom colors for each category
custom_colors_13a = ['#385AC2',
                     '#AE305D',
                     
                     '#8B3FCF',
                     '#008000',
                     '#FFD700',
                     
                     '#D3D3D3']

# Bar plot-------------------------------------------
category_percent_13a.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors_13a,
                          width = 1.0,
                          alpha = 1.0,
                          label = "13a protagonist is")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the single protagonist or main character?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 5.0,
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

# Save image-------------------------------------------
plt.savefig("13a protagonist is Gemini.png", bbox_inches = 'tight')
#plt.savefig("13a protagonist is Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("13a protagonist is Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 21 - 13b protagonist is
print("  Making 13b protagonist is...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure13b = plt.figure(21, figsize = (12, 6))
gs = figure13b.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure13b.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_13b = ['Male', 
                      'Female',
                      
                      'Other',
                      'Non-human',
                      'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_13b = pd.DataFrame(columns=category_order_13b).astype(float)
category_percent_13b = pd.concat([all_categories_df_13b, category_percent_13b]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_13b = category_percent_13b[category_order_13b]

# Define custom colors for each category
custom_colors_13b = ['#385AC2',
                     '#AE305D',
                     
                     '#8B3FCF',
                     '#008000',
                     '#FFD700']

# Bar plot-------------------------------------------
category_percent_13b.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors_13b,
                          width = 1.0,
                          alpha = 1.0,
                          label = "13b protagonist is")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the single protagonist or main character?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 6.0,
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

# Save image-------------------------------------------
plt.savefig("13b protagonist is Gemini.png", bbox_inches = 'tight')
#plt.savefig("13b protagonist is Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("13b protagonist is Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')





#----------------------------------------------------------------------------------
# Figure 22 - 13c protagonist is
print("  Making 13c protagonist is hard and very hard...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure13c = plt.figure(22, figsize = (12, 6))
gs = figure13c.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure13c.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_13c = ['Male', 
                      'Female',
                      
                      'Other',
                      'Non-human',
                      'Uncertain',
                      
                      'Not applicable']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_13c = pd.DataFrame(columns=category_order_13c).astype(float)
category_percent_13c = pd.concat([all_categories_df_13c, category_percent_13c]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_13c = category_percent_13c[category_order_13c]

# Define custom colors for each category
custom_colors_13c = ['#385AC2',
                     '#AE305D',
                     
                     '#8B3FCF',
                     '#008000',
                     '#FFD700',
                     
                     '#D3D3D3']

# Bar plot-------------------------------------------
category_percent_13c.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors_13c,
                          width = 1.0,
                          alpha = 1.0,
                          label = "13b protagonist is")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the single protagonist or main character for hard and very hard sci-fi?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 5.0,
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

# Save image-------------------------------------------
plt.savefig("13c protagonist is hard and very hard Gemini.png", bbox_inches = 'tight')
#plt.savefig("13c protagonist is  hard and very hard Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("13c protagonist is  hard and very hard Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 23 - 14 virtual
print("  Making 14 virtual...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure14 = plt.figure(23, figsize = (12, 6))
gs = figure14.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure14.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_14 = ['Yes', 
                     'Somewhat',
                     'No',
                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_14 = pd.DataFrame(columns=category_order_14).astype(float)
category_percent_14 = pd.concat([all_categories_df_14, category_percent_14]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_14 = category_percent_14[category_order_14]

# Define custom colors for each category
custom_colors_14 = ['#AE305D',
                    '#8B3FCF',
                    '#385AC2',
                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_14.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_14,
                         width = 1.0,
                         alpha = 1.0,
                         label = "14 virtual")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions of virtual reality or immersive digital environments \n(e.g., simulations, augmented reality) in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Save image-------------------------------------------
plt.savefig("14 virtual Gemini.png", bbox_inches = 'tight')
#plt.savefig("14 virtual Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("14 virtual Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 24 - 15 tech and science
print("  Making 15 tech and science...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure15 = plt.figure(24, figsize = (12, 6))
gs = figure15.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure15.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_15 = ['Bad', 
                     'Leaning bad', 
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_15 = pd.DataFrame(columns=category_order_15).astype(float)
category_percent_15 = pd.concat([all_categories_df_15, category_percent_15]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_15 = category_percent_15[category_order_15]

# Define custom colors for each category
custom_colors_15 = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_15.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_15,
                         width = 1.0,
                         alpha = 1.0,
                         label = "15 tech and science")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are technology and science depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.5,
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

# Save image-------------------------------------------
plt.savefig("15 tech and science Gemini.png", bbox_inches = 'tight')
#plt.savefig("15 tech and science Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("15 tech and science Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 25 - 16 social issues
print("  Making 16 social issues...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure16 = plt.figure(25, figsize = (12, 6))
gs = figure16.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure16.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_16 = ['Core', 
                     'Major',
                     'Minor',

                     'Absent',
                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_16 = pd.DataFrame(columns=category_order_16).astype(float)
category_percent_16 = pd.concat([all_categories_df_16, category_percent_16]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_16 = category_percent_16[category_order_16]

# Define custom colors for each category
custom_colors_16 = ['#385AC2',
                    '#5580D0',
                    '#6CACEB',

                    '#AE305D',
                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_16.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_16,
                         width = 1.0,
                         alpha = 1.0,
                         label = "16 social issues")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How central is the critique or reflection of specific social issues \n(e.g., inequality, war, discrimination, political oppression) to the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 6.0,
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

# Save image-------------------------------------------
plt.savefig("16 social issues Gemini.png", bbox_inches = 'tight')
#plt.savefig("16 social issues Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("16 social issues Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 26 - 17 enviromental
print("  Making 17 enviromental...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure17 = plt.figure(26, figsize = (12, 6))
gs = figure17.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure17.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_17 = ['Core', 
                     'Major',
                     'Minor',

                     'Absent',
                     'Uncertain']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_17 = pd.DataFrame(columns=category_order_17).astype(float)
category_percent_17 = pd.concat([all_categories_df_17, category_percent_17]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_17 = category_percent_17[category_order_17]

# Define custom colors for each category
custom_colors_17 = ['#385AC2',
                    '#5580D0',
                    '#6CACEB',

                    '#AE305D',
                    '#FFD700']

# Bar plot-------------------------------------------
category_percent_17.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_17,
                         width = 1.0,
                         alpha = 1.0,
                         label = "17 enviromental")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How central is an ecological or environmental message to the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 6.0,
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

# Save image-------------------------------------------
plt.savefig("17 enviromental Gemini.png", bbox_inches = 'tight')
#plt.savefig("17 enviromental Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("17 enviromental Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# For the Tests and more
#----------------------------------------------------------------------------------
# Figure 27 - Author and protagonist heatmap
print("  Making author and protagonist heatmap...")

#-------------------------------------------
# Creates a figure object with size 18x5 inches
figure_t1 = plt.figure(27, figsize = (18, 5))
gs = figure_t1.add_gridspec(ncols = 2, nrows = 1)

# Define the desired order for Author Gender (y-axis)
author_gender_order = ['Male', 'Female', 'Other']

# Define the desired order for Protagonist Gender (x-axis)
protagonist_gender_order = ['Male', 'Female', 'Other', 'Non-human', 'Uncertain', 'Not applicable']

#-------------------------------------------
# Step 1: Create a contingency table (cross-tab)
contingency_table = pd.crosstab(df_top_AI['gender'], df_top_AI['13 protagonist is'])

contingency_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

# Reindex rows and columns according to the desired order
contingency_table = contingency_table.reindex(index=author_gender_order, columns=protagonist_gender_order)

# Create a formatted string version of the percentages with % symbol for annotation
annot = contingency_table.map(lambda x: f'{x:.2f}%')

print("\n")
print("Contingency Table:")
print(contingency_table)

#-------------------------------------------
# Create the main plot
ax1 = figure_t1.add_subplot(gs[0])
#sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
sns.heatmap(contingency_table, cmap='coolwarm', annot=annot, fmt="")

# Set the title and labels
ax1.set_title('Contingency Table Heatmap (All decades)')
ax1.set_xlabel('Protagonist Gender')
ax1.set_ylabel('Author Gender')

# Rotate and align the labels
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')  # Horizontal alignment (right aligns better for rotated labels)
    tick.set_va('top')  # Vertical alignment to ensure no overlap

#-------------------------------------------
# Step 2: Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Results
print("Chi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant correlation between Author Gender and Protagonist Gender.")
else:
    print("There is no significant correlation between Author Gender and Protagonist Gender.")

#-------------------------------------------
mask = df_top_AI['decade'] >= 2000
df_top_AI_2000 = df_top_AI[mask]

# Step 1: Create a contingency table (cross-tab)
contingency_table_2000 = pd.crosstab(df_top_AI_2000['gender'], df_top_AI_2000['13 protagonist is'])

contingency_table_2000 = contingency_table_2000.div(contingency_table_2000.sum(axis = 1), axis = 0) * 100

# Reindex rows and columns according to the desired order
contingency_table_2000 = contingency_table_2000.reindex(index=author_gender_order, columns=protagonist_gender_order)

# Create a formatted string version of the percentages with % symbol for annotation
annot_2000 = contingency_table_2000.map(lambda x: f'{x:.2f}%')

print("\n")
print("Contingency Table 2000s:")
print(contingency_table_2000)

#-------------------------------------------
# Create the main plot
ax2 = figure_t1.add_subplot(gs[1])
#sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
sns.heatmap(contingency_table_2000, cmap='coolwarm', annot=annot_2000, fmt="")

# Set the title and labels
ax2.set_title('Contingency Table Heatmap (2000s, 2010s, 2020s)')
ax2.set_xlabel('Protagonist Gender')
ax2.set_ylabel('Author Gender')

# Rotate and align the labels
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')  # Horizontal alignment (right aligns better for rotated labels)
    tick.set_va('top')  # Vertical alignment to ensure no overlap

#-------------------------------------------
# Step 2: Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table_2000)

# Results
print("Chi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant correlation between Author Gender and Protagonist Gender.")
else:
    print("There is no significant correlation between Author Gender and Protagonist Gender.")

# Save image-------------------------------------------
plt.savefig("00 author and protagonist heatmap Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 author and protagonist heatmap Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 author and protagonist heatmap Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 28 - Author and protagonist gender
print("  Making author and protagonist heatmap...")

#------------------------------------------
# Creates a figure object with size 12x6 inches
figure_t2 = plt.figure(28, figsize = (12, 6))
gs = figure_t2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_genders = ['Male / Male',
                          'Male / Female',
                          'Male / Other',

                          'Male / Non-human',
                          'Male / Uncertain',
                          'Male / Not applicable',
                          
                          'Female / Male',
                          'Female / Female',
                          'Female / Other',

                          'Female / Non-human',
                          'Female / Uncertain',
                          'Female / Not applicable',
                          
                          'Other / All combined']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_genders = pd.DataFrame(columns=category_order_genders).astype(float)
category_percent_genders = pd.concat([all_categories_df_genders, category_percent_genders]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_genders = category_percent_genders[category_order_genders]

# Define custom colors for each category
custom_colors_genders = ['#385AC2',
                         '#5580D0',
                         '#6CACEB',

                         '#A4C8ED',
                         '#FFD700',
                         '#CFE3F7',
                         
                         '#AE305D',
                         '#CF5D5F',
                         '#E3937B',

                         '#EFB8A7',
                         '#FFD700',
                         '#FBD9CF',
                         
                         '#8B3FCF']

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
           labelspacing = 1.5,
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

# Save image-------------------------------------------
plt.savefig("00 author and protagonist gender Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 author and protagonist gender Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 author and protagonist gender Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 29 - Variation in Answer
print("  Making Variation in Answers...")

#----------------------------------------------------------------------------------
# Creates a figure object with size 10x14 inches
figure_t3 = plt.figure(29, figsize = (10, 14))
gs = figure_t3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t3.add_subplot(gs[0])

# Heatmap-------------------------------------------
# Set var_flag to the measure of variation in answers that you want:
# 1 Difference two by two of the answers
# 2 Percent Agreement / Mode Consistency
# 3 Shannon Entropy (Diversity Index)
var_flag = 3

#-----------------------------
# 1 Difference two by two of the answers
if var_flag == 1:
    sns.heatmap(df_mean_difference, 
                annot=True, 
                cmap="coolwarm",
                fmt=".2f", 
                cbar_kws={'label': 'Mean Difference Score'},
                annot_kws={"size": 8})

#-----------------------------
# 2 Percent Agreement / Mode Consistency
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
    # Define tolerance for near-zero values
    TOLERANCE = 1e-6

    # Custom format function for your specific rule
    def custom_format(val):
        # Treat values within the tolerance as zero
        if abs(val) < TOLERANCE:
            return 0
        elif abs(val) >= 100:
            return f"{val:.0f}"  # No decimals for numbers >= 100
        elif abs(val) >= 10:
            return f"{val:.1f}"  # One decimal for numbers >= 10 and < 100
        else:
            return f"{val:.2f}"  # Two decimals for numbers < 10

    # Apply the function to each cell
    sns.heatmap(df_entropy, 
                annot=df_entropy.applymap(custom_format), 
                cmap="coolwarm",
                fmt="", 
                cbar_kws={'label': 'Shannon index'},
                annot_kws={"size": 8})

#-----------------------------
# Add thicker lines to separate the last row and column
"""
ax.get_xlim() returns a tuple with the x-axis limits, like (0.0, 4.0).
Using *ax.get_xlim() unpacks this tuple into two separate arguments. 
So, hlines interprets it as hlines(y, xmin=0.0, xmax=4.0).
"""
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
    plt.savefig("00 variation difference Gemini.png", bbox_inches = 'tight')
elif var_flag == 2:
    plt.savefig("00 variation mode Gemini.png", bbox_inches = 'tight')
elif var_flag == 3:
    plt.savefig("00 variation entropy Gemini.png", bbox_inches = 'tight')

#plt.savefig("00 variation Gemini.png", bbox_inches = 'tight')
#plt.savefig("00 variation Gemini.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("00 variation Gemini.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
print("All done.")

# Show figures-------------------------------------------------------------------------------------------
plt.show() # You must call plt.show() to make graphics appear.