"""
This script process basic data for the novels and makes figures.

Modules:
    - pandas
    - matplotlib.pyplot
    - seaborn
    - numpy
    - typing
    - pathlib
    - sys
"""

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import numpy as np
from typing import List

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED, ANSWERS, FIGURES

#---------------------------------------------------------------------------------------------------
print("Reading and processing tha data...")

# Read the data
df_filtered = pd.read_csv(FILTERED / "sci-fi_novels_FILTERED.csv", sep=";", encoding="utf-8-sig")
df_top = pd.read_csv(FILTERED / "sci-fi_novels_TOP_Wiki.csv", sep=";", encoding="utf-8-sig")
df_top_AI = pd.read_csv(ANSWERS / "sci-fi_novels_AI_ANSWERS.csv", sep=";", encoding="utf-8-sig")
df_top_AI_gender = pd.read_csv(ANSWERS / "sci-fi_novels_AI_ANSWERS_GENDER.csv", sep=";", encoding="utf-8-sig")

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
# For the boxplots

# Add a column to each DataFrame to label the dataset
df_filtered['dataset'] = 'Filtered sample'
df_top['dataset'] = 'Top sample'

# Concatenate dataframes
df_filtered_200 = pd.concat([df_filtered, df_top])

#---------------------------------------------------------------------------------------------------
# General information of the FILTERED sample of novels
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED novels.")

novel_per_decade = df_filtered['decade'].value_counts()
mean_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].std()

print(novel_per_decade.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_1 = {'quantity': novel_per_decade,
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

#---------------------------------------------------------------------------------------------------
# General information of the top 200 PER DECADE sample of novels
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE novels.")

novel_per_decade_200 = df_top['decade'].value_counts()
mean_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].std()

print(novel_per_decade_200.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_2 = {'quantity': novel_per_decade_200,
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
#---------------------------------------------------------------------------------------------------
# Make figures
print("Making the figures...")
#---------------------------------------------------------------------------------------------------
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)

#---------------------------------------------------------------------------------------------------
# Figure 1, Quantities
print("  Making novel counts...")

# Creates a figure object with size 12x6 inches
figure_1 = plt.figure(1, figsize = (12, 6))
gs = figure_1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_1.add_subplot(gs[0])

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
ax1.set_title("Number of Sci-fi novels per Decade In The Samples", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig(FIGURES / "00 Quantities.png", bbox_inches = 'tight')
#plt.savefig(FIGURES / "00 Quantities.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig(FIGURES / "00 Quantities.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_1)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Figure 2, Quantities
print("  Making rate and ratings...")

# Creates a figure object with size 14x8 inches
figure_2 = plt.figure(2, figsize = (14, 8))
gs = figure_2.add_gridspec(ncols = 2, nrows = 1)

#figure_2.subplots_adjust(hspace = 0.5)

#-----------------------------------------
# Create the main plot
ax1 = figure_2.add_subplot(gs[0])

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

#---------------------------------------------------------------------------------------------------
# Create the main plot
ax2 = figure_2.add_subplot(gs[1])

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
plt.savefig(FIGURES / "00 Rates and Ratings.png", bbox_inches = 'tight')
#plt.savefig(FIGURES / "00 Rates and Ratings.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig(FIGURES / "00 Rates and Ratings.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_2)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def figure_maker (number: int, 
                  column_name: str,
                  df_top: pd.DataFrame,
                  category_order: List[str], 
                  custom_colors: List[str], 
                  title: str, 
                  printing_name: str) -> None:
    
    """
    Function to make most of the figures.

    Args:
        number (int): Number of the figure.
        df_top (Dataframe): Dataframe with all data.
        column_name (str): Name of data column to be used.
        category_order (List[str]): Order of the categories.
        custom_colors (List[str]): Color of the categories.
        title (str): Figure title.
        printing_name (str): Name to be printed as the file name and label.

    Returns:
        No return.
    """

    #---------------------------------------------------------------------------------------------------
    # Creates a figure object with size 12x6 inches
    figure = plt.figure(number, figsize = (12, 6))
    gs = figure.add_gridspec(ncols = 1, nrows = 1)

    # Create the main plot
    ax1 = figure.add_subplot(gs[0])

    # Custom dark gray color
    custom_dark_gray = (0.2, 0.2, 0.2)

    # Custom label spacing
    custom_label_spacing = {
        1: 1.0,
        2: 12.0,
        3: 10.0,
        4: 8.0,
        5: 6.0,
        6: 4.5,
        7: 4.0,
        8: 3.2,
        9: 2.8,
        10: 2.4,
        11: 2.0,
        12: 1.6
    }

    #-------------------------------------------
     # Count the occurrences of each category per decade
    category_counts = pd.crosstab(df_top['decade'], df_top[column_name])
    # Normalize the counts to get percentages
    category_percent = category_counts.div(category_counts.sum(axis = 1), axis = 0) * 100

    #-------------------------------------------
    # Create a new DataFrame with all desired categories, filling with 0 if missing
    all_categories_df = pd.DataFrame(columns=category_order).astype(float)
    category_percent = pd.concat([all_categories_df, category_percent]).convert_dtypes().convert_dtypes().fillna(0.0)

    # Reorder the columns in the DataFrame according to the desired category order
    category_percent = category_percent[category_order]

    # Bar plot-------------------------------------------
    category_percent.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors,
                          width = 1.0,
                          alpha = 1.0,
                          label = printing_name)

    # Design-------------------------------------------
    ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
    #ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
    ax1.set_title(title, fontsize = 14, pad = 5, color = custom_dark_gray)
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
            labelspacing = custom_label_spacing[len(category_order)],
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
    plt.savefig(FIGURES / f"{printing_name}.png", bbox_inches = 'tight')
    #plt.savefig(fFIGURES / f"{printing_name}.eps", transparent = True, bbox_inches = 'tight')
    # Transparence will be lost in .eps, save in .svg for transparences
    #plt.savefig(fFIGURES / f"{printing_name}.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
    plt.close(figure)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Figure 3, Series (top)
print("  Making series (top)...")

# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (3, # number
              "series", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (top sample)", # title
              "00 series (top 200)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 4, Series (filtered)
print("    Making series (filtered)...")

# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (4, # number
              "series", # column_name
              df_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (filtered sample)", # title
              "00 series (filtered)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 5, Series (top 50)
print("    Making series (top 50)...")

# Create decade columns to use in the groupby below and keep the original decade
df_top_AI_2 = df_top_AI.copy(deep=True)
df_top_AI_2['decade_gb'] = df_top_AI_2['decade']

# Group by 'decade_gb', sort by 'ratings' in descending order, and select the top 50 per group
df_top_50 = (df_top_AI_2.groupby('decade_gb', group_keys=False)
                .apply((lambda x: x.sort_values('ratings', ascending=False).head(50)), 
                    include_groups=False))

#---------------------------------------------
# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (5, # number
              "series", # column_name
              df_top_50, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (top 50)", # title
              "00 series (top 50)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 6, author gender
print("  Making author gender...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

figure_maker (6, # number
              "author gender", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author?", # title
              "00 author gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 7, author gender accuracy
print("  Making author gender accuracy...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

# Define the condition to select rows
mask_1 = df_top_AI['1 accuracy'] == "High"
mask_2 = df_top_AI['1 accuracy'] == "Very high"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (7, # number
              "author gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author for (very) highly accurate sci-fi?", # title
              "00 author gender accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 8, author gender accuracy
print("  Making author gender discipline...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

# Define the condition to select rows
mask_1 = df_top_AI['2 discipline'] == "Leaning hard sciences"
mask_2 = df_top_AI['2 discipline'] == "Hard sciences"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (8, # number
              "author gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author for (leaning) hard sciences sci-fi?", # title
              "00 author gender discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
print("All done.")

# Show figures-------------------------------------------------------------------------------------------
# plt.show() # Too many figures (> 20), so better not to show them.