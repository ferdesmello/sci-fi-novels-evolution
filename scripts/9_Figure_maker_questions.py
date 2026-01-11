"""
This script process questions and answers data and makes figures.

Modules:
    - pandas
    - matplotlib.pyplot
    - typing
    - pathlib
    - sys
"""

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Make figures
print("Making the figures...")

#---------------------------------------------------------------------------------------------------
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
    Function to make most figures.

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
    #plt.savefig(FIGURES / f"{printing_name}.eps", transparent = True, bbox_inches = 'tight')
    # Transparence will be lost in .eps, save in .svg for transparences
    #plt.savefig(FIGURES / f"{printing_name}.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
    plt.close(figure)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# figures for the questions/answers
#---------------------------------------------------------------------------------------------------
# Figure 7 - 1 accuracy
print("1 accuracy...")

# Desired order of the categories
category_order = ['Very low',
                  'Low',
                  'Moderate', 
                  'High',
                  'Very high',
                  
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',
                 
                 '#FFD700']

figure_maker (7, # number
              "1 accuracy", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How important is scientific accuracy and plausibility in the story?", # title
              "01 accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 8 - 2 discipline
print("2 discipline...")

# Desired order of the categories
category_order = ['Soft sciences',
                  'Leaning soft sciences',
                  'Mixed', 
                  'Leaning hard sciences',
                  'Hard sciences',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (8, # number
              "2 discipline", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the main disciplinary focus of the story?", # title
              "02 discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 9 - 3 light heavy
print("3 light heavy...")

# Desired order of the categories
category_order = ['Very heavy',
                  'Heavy',
                  'Balanced', 
                  'Light',
                  'Very light',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (9, # number
              "3 light heavy", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the story more of a light or heavy reading experience?", # title
              "03 light heavy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 10 - 4 time
print("4 time...")

# Desired order of the categories
category_order = ['Distant past',
                  'Far past',
                  'Near past',
                  'Present',
                  'Near future',
                  'Far future',
                  'Distant future',

                  'Multiple timelines',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D', # Distant past
                 '#CF5D5F',
                 '#E3937B',
                 '#8B3FCF', # Present
                 '#6CACEB',
                 '#5580D0',
                 '#385AC2', # Distant future

                 '#008000', # Multiple timelines
                 '#FFD700'] # Uncertain

figure_maker (10, # number
              "4 time", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "When does most of the story take place in relation to the year the novel was published?", # title
              "04 time") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 11 - 5 mood
print("5 mood...")

# Desired order of the categories
category_order = ['Very pessimistic',
                  'Pessimistic',
                  'Balanced',
                  'Optimistic',
                  'Very optimistic',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (11, # number
              "5 mood", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the mood of the story?", # title
              "05 mood") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 12 - 6 ending
print("6 ending...")

# Desired order of the categories
category_order = ['Very negative',
                  'Negative',
                  'Ambivalent',
                  'Positive',
                  'Very positive',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (12, # number
              "6 ending", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the overall mood and outcome of the story's ending?", # title
              "06 ending") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 13 - 7 social political
print("7 social political...")

# Desired order of the categories
category_order = ['Dystopic',
                  'Leaning dystopic',
                  'Balanced',
                  'Leaning utopic',
                  'Utopic',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (13, # number
              "7 social political", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the social-political scenario depicted in the story?", # title
              "07 social political") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 14 - 8 politically unified
print("8 politically unified...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (14, # number
              "8 politically unified", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is a unified, planetary-level or multi-planet state or government depicted in the story?", # title
              "08 politically unified") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 15 - 9 on Earth
print("9 on Earth...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#8B3FCF',
                 '#AE305D',
                 '#FFD700']

figure_maker (15, # number
              "9 on Earth", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is most of the story set on planet Earth?", # title
              "09 on Earth") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 16 - 10 post apocalyptic
print("10 post apocalyptic...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (16, # number
              "10 post apocalyptic", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the setting of the story post-apocalyptic?", # title
              "10 post apocalyptic") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 17 - 11 conflict
print("11 conflict...")

# Desired order of the categories
category_order = ['Internal',
                  'Interpersonal',
                  'Societal',
                  'Synthetic',
                  'Technological',
                  'Extraterrestrial',
                  'Natural',
                  'Mixed',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D', # Internal
                 '#CF5D5F', # Interpersonal
                 '#E3937B', # Societal
                 '#385AC2', # Synthetic
                 '#5580D0', # Technological
                 '#6CACEB', # Extraterrestrial
                 '#008000', # Natural
                 '#8B3FCF', # Mixed

                 '#FFD700'] # Uncertain

figure_maker (17, # number
              "11 conflict", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the dominant type of conflict in the story?", # title
              "11 conflict") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 18 - 12 aliens
print("12 aliens...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (18, # number
              "12 aliens", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict extraterrestrial life forms or alien technology?", # title
              "12 aliens") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 19 - 13a aliens are
print("13a aliens are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (19, # number
              "13 aliens are", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts extraterrestrial life forms, how are they generally portrayed?", # title
              "13a aliens are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 20 - 13b aliens are
print("  13b aliens are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["13 aliens are"] != "Not applicable"]

figure_maker (20, # number
              "13 aliens are", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts extraterrestrial life forms, how are they generally portrayed?", # title
              "13b aliens are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 21 - 4 time
print("14 robots and AI...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (21, # number
              "14 robots and AI", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict robots or artificial intelligences?", # title
              "14 robots and AI") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 22 - 15a robots and AI are
print("15a robots and AI are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',
                
                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (22, # number
              "15 robots and AI are", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts robots or artificial intelligences, how are they generally portrayed?", # title
              "15a robots and AI are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 23 - 15b robots and AI are
print("  15b robots and AI are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',
                
                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["15 robots and AI are"] != "Not applicable"]

figure_maker (23, # number
              "15 robots and AI are", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts robots or artificial intelligences, how are they generally portrayed?", # title
              "15b robots and AI are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 24 - 16 protagonist
print("16 protagonist...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (24, # number
              "16 protagonist", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is there a single main character or protagonist in the story?", # title
              "16 protagonist") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 25 - 17a protagonist nature
print("17a protagonist nature...")

# Desired order of the categories
category_order = ['Human', 
                  'Non-human',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',

                 '#FFD700',
                 '#D3D3D3']

figure_maker (25, # number
              "17 protagonist nature", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their nature?", # title
              "17a protagonist nature") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 26 - 17b protagonist nature
print("  17b protagonist nature...")

# Desired order of the categories
category_order = ['Human', 
                  'Non-human',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["17 protagonist nature"] != "Not applicable"]

figure_maker (26, # number
              "17 protagonist nature", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their nature?", # title
              "17b protagonist nature") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 27 - 18a protagonist gender
print("  18a protagonist gender...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

figure_maker (27, # number
              "18 protagonist gender", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their gender?", # title
              "18a protagonist gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 28 - 18ba protagonist gender
print("  18b protagonist gender...")

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

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["18 protagonist gender"] != "Not applicable"]

figure_maker (28, # number
              "18 protagonist gender", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their gender?", # title
              "18b protagonist gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 29 - 18c protagonist gender accuracy
print("  18c protagonist gender accuracy...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

# Define the condition to select rows
mask_1 = df_top_AI['1 accuracy'] == "High"
mask_2 = df_top_AI['1 accuracy'] == "Very high"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (29, # number
              "18 protagonist gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the single protagonist or main character \nfor (very) highly accurate sci-fi?", # title
              "18c protagonist gender accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 30 - 18d protagonist gender discipline
print("  18d protagonist gender discipline...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

# Define the condition to select rows
mask_1 = df_top_AI['2 discipline'] == "Leaning hard sciences"
mask_2 = df_top_AI['2 discipline'] == "Hard sciences"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (30, # number
              "18 protagonist gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the single protagonist or main character \nfor (leaning) hard sciences sci-fi?", # title
              "18d protagonist gender discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 31 - 19a protagonist is
print("  19a protagonist is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (31, # number
              "19 protagonist is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, how are they generally portrayed?", # title
              "19a protagonist is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 32 - 19b protagonist is
print("  19b protagonist is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['19 protagonist is'] != "Not applicable"]

figure_maker (32, # number
              "19 protagonist is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, how are they generally portrayed?", # title
              "19b protagonist is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 33 - 20 virtual
print("20 virtual...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (33, # number
              "20 virtual", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict virtual or augmented reality?", # title
              "20 virtual") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 34 - 21a virtual is
print("21a virtual is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (34, # number
              "21 virtual is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts virtual or augmented reality, how are they generally depicted?", # title
              "21a virtual is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 35 - 21b virtual is
print("  21b virtual is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['21 virtual is'] != "Not applicable"]

figure_maker (35, # number
              "21 virtual is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts virtual or augmented reality, how are they generally depicted?", # title
              "21b virtual is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 36 - 22 biotech
print("22 biotech...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (36, # number
              "22 biotech", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict biotechnology, genetic engineering, or human biological alteration?", # title
              "22 biotech") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 37 - 23a biotech is
print("23a biotech is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (37, # number
              "23 biotech is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts biotechnology, genetic engineering, or human biological alteration, \nhow are they generally depicted?", # title
              "23a biotech is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 38 - 23b biotech is
print("  23b biotech is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['23 biotech is'] != "Not applicable"]

figure_maker (38, # number
              "23 biotech is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts biotechnology, genetic engineering, or human biological alteration, \nhow are they generally depicted?", # title
              "23b biotech is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 39 - 24 transhuman
print("24 transhuman...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (39, # number
              "24 transhuman", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict transhumanism or the transcendence of human limitations?", # title
              "24 transhuman") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 40 - 25a transhuman is
print("25a transhuman is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (40, # number
              "25 transhuman is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts transhumanism or the transcendence of human limitations, \nhow are they generally depicted?", # title
              "25a transhuman is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 41 - 25b transhuman is
print("  25b transhuman is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['25 transhuman is'] != "Not applicable"]

figure_maker (41, # number
              "25 transhuman is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts transhumanism or the transcendence of human limitations, \nhow are they generally depicted?", # title
              "25b transhuman is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 42 - 26 tech and science
print("26 tech and science...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

figure_maker (42, # number
              "26 tech and science", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How are science and technology depicted in the story?", # title
              "26 tech and science") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 43 - 27 social issues
print("27 social issues...")

# Desired order of the categories
category_order = ['Core', 
                  'Major',
                  'Minor',

                  'Absent',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#5580D0',
                 '#6CACEB',

                 '#AE305D',
                 '#FFD700']

figure_maker (43, # number
              "27 social issues", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How central is the critique of specific social issues in the story?", # title
              "27 social issues") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 44 - 28 enviromental
print("28 enviromental...")

# Desired order of the categories
category_order = ['Core', 
                  'Major',
                  'Minor',

                  'Absent',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#5580D0',
                 '#6CACEB',

                 '#AE305D',
                 '#FFD700']

figure_maker (44, # number
              "28 enviromental", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How central are ecological or environmental themes in the story?", # title
              "28 enviromental") # printing_name and label

#---------------------------------------------------------------------------------------------------
print("All done.")

# Show figures-------------------------------------------------------------------------------------------
# plt.show() # Too many figures (> 20), so better not to show them.