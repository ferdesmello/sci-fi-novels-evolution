"""
This script cleans the data if new data have been included tardily.

Modules:
    - pandas
    - winsound (only for Windows)
    - pathlib
    - sys
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
import pandas as pd
import winsound

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED, ANSWERS

#----------------------------------------------------------------------------------
def main():
    """
    Big main function with all the cleaning of the data if new data was added.
    """

    # Read the files
    print("Reading the data...")

    df_TOP = pd.read_csv(FILTERED / 'sci-fi_novels_TOP.csv', sep = ';', encoding="utf-8-sig")
    df_TOP_Wiki = pd.read_csv(FILTERED / 'sci-fi_novels_TOP_Wiki.csv', sep = ';', encoding="utf-8-sig")
    df_AI_ANSWERS = pd.read_csv(ANSWERS / 'sci-fi_novels_AI_ANSWERS.csv', sep = ';', encoding="utf-8-sig")
    df_AI_GENDER = pd.read_csv(ANSWERS / 'sci-fi_novels_AI_ANSWERS_GENDER.csv', sep = ';', encoding="utf-8-sig")

    #----------------------------------------------------------------------------------
    # Clean the files, so they have only sci-fi_novels_TOP.csv novels.

    df_cleaned_TOP_Wiki = pd.DataFrame(columns = df_TOP_Wiki.columns)
    df_cleaned_ANSWERS = pd.DataFrame(columns = df_AI_ANSWERS.columns)
    df_cleaned_GENDER = pd.DataFrame(columns = df_AI_GENDER.columns)
 
    keys = ["title", "author", "url goodreads"]

    #------------------------------------------
    # TOP_Wiki
    print("Cleaning TOP_Wiki...")

    # Ensure no duplicates in df_TOP_Wiki
    df_TOP_Wiki = df_TOP_Wiki.drop_duplicates(subset=keys, keep="first")

    # Left join: df_TOP defines what rows exist
    df_cleaned_TOP_Wiki = df_TOP.merge(df_TOP_Wiki, on=keys, how="left", suffixes=("", "_B"))

    # Add extra columns and delete _B columns
    for col in df_TOP_Wiki.columns:
        if col + "_B" in df_cleaned_TOP_Wiki.columns:
            df_cleaned_TOP_Wiki[col] = df_cleaned_TOP_Wiki[col].fillna(df_cleaned_TOP_Wiki[col + "_B"])
            df_cleaned_TOP_Wiki = df_cleaned_TOP_Wiki.drop(columns=[col + "_B"])

    print(df_cleaned_TOP_Wiki.info())
    
    # Get the sets of keys (turn each row of keys into a tuple so they're hashable)
    keys_top = set(map(tuple, df_TOP[keys].to_numpy()))
    keys_wiki = set(map(tuple, df_TOP_Wiki[keys].to_numpy()))

    # Find differences
    only_in_top = keys_top - keys_wiki # rows present in df_TOP but not in df_TOP_Wiki
    only_in_wiki = keys_wiki - keys_top # rows present in df_TOP_Wiki but not in df_TOP

    print(f"{len(only_in_top)} row(s) added (only in df_TOP).")
    print(f"{len(only_in_wiki)} row(s) deleted (only in df_TOP_Wiki).")

    #------------------------------------------
    # ANSWERS
    print("Cleaning ANSWERS...")

    # Ensure no duplicates in df_TOP_Wiki
    df_AI_ANSWERS = df_AI_ANSWERS.drop_duplicates(subset=keys, keep="first")

    # Left join: df_TOP defines what rows exist
    df_cleaned_ANSWERS = df_TOP.merge(df_AI_ANSWERS, on=keys, how="left", suffixes=("", "_B"))

    # Add extra columns and delete _B columns
    for col in df_cleaned_ANSWERS.columns:
        if col + "_B" in df_cleaned_ANSWERS.columns:
            df_cleaned_ANSWERS[col] = df_cleaned_ANSWERS[col].fillna(df_cleaned_ANSWERS[col + "_B"])
            df_cleaned_ANSWERS = df_cleaned_ANSWERS.drop(columns=[col + "_B"])

    print(df_cleaned_ANSWERS.info())

    # Get the sets of keys (turn each row of keys into a tuple so they're hashable)
    keys_top = set(map(tuple, df_TOP[keys].to_numpy()))
    keys_answers = set(map(tuple, df_AI_ANSWERS[keys].to_numpy()))

    # Find differences
    only_in_top = keys_top - keys_answers # rows present in df_TOP but not in df_AI_ANSWERS
    only_in_answers = keys_answers - keys_top # rows present in df_AI_ANSWERS but not in df_TOP

    print(f"{len(only_in_top)} row(s) added (only in df_TOP).")
    print(f"{len(only_in_answers)} row(s) deleted (only in df_AI_ANSWERS).")

    #------------------------------------------
    # GENDER
    print("Cleaning GENDER...")

    authors = set(df_TOP['author'])

    df_cleaned_GENDER = df_AI_GENDER[
        df_AI_GENDER['author'].isin(authors) 
    ]

    print(df_cleaned_GENDER.info())

    # Get the sets of keys (turn each row of keys into a tuple so they're hashable)
    keys_top = set(map(tuple, df_TOP["author"].to_numpy()))
    keys_gender = set(map(tuple, df_AI_GENDER["author"].to_numpy()))

    # Find differences
    only_in_top = keys_top - keys_gender # rows present in df_TOP but not in df_AI_GENDER
    only_in_gender = keys_gender - keys_top # rows present in df_AI_GENDER but not in df_TOP

    print(f"{len(only_in_top)} row(s) added (only in df_TOP).")
    print(f"{len(only_in_gender)} row(s) deleted (only in df_AI_GENDER).")

    #------------------------------------------
    column_order = [
         'title', 
         'author', 
         'year',
         'paragraph',
         '1 accuracy',
         'justifying accuracy',
         '2 discipline',
         'justifying discipline',
         '3 light heavy',
         'justifying light heavy',
         '4 time',
         'justifying time',
         '5 mood',
         'justifying mood',
         '6 social political',
         'justifying social political',
         '7 on Earth',
         'justifying on Earth',
         '8 post apocalyptic',
         'justifying post apocalyptic',
         '9 aliens',
         'justifying aliens',
         '10 aliens are',
         'justifying aliens are',
         '11 robots and AI',
         'justifying robots and AI',
         '12 robots and AI are',
         'justifying robots and AI are',
         '13 protagonist',
         'justifying protagonist',
         '14 protagonist nature',
         'justifying protagonist nature',
         '15 protagonist gender',
         'justifying protagonist',
         '16 virtual',
         'justifying virtual',
         '17 tech and science',
         'justifying tech and science',
         '18 social issues',
         'justifying social issues',
         '19 enviromental',
         'justifying enviromental',
         'complete answer',
         'decade', 
         'rate', 
         'ratings', 
         'series', 
         'genres', 
         'synopsis',
         'review',
         'url goodreads',
         'plot',
         'url wikipedia'
        ]

    # Reorder the columns
    df_cleaned_ANSWERS = df_cleaned_ANSWERS.reindex(columns=column_order)

    #----------------------------------------------------------------------------------
    # Save the cleaned files
    print("Saving the files...")

    outoput_file_TOP_Wiki = FILTERED / 'sci-fi_novels_TOP_Wiki.csv'
    outoput_file_ANSWERS = ANSWERS / 'sci-fi_novels_AI_ANSWERS.csv'
    output_file_gender = ANSWERS / 'sci-fi_novels_AI_ANSWERS_GENDER.csv'

    df_cleaned_TOP_Wiki.to_csv(outoput_file_TOP_Wiki, index=False, sep=';', encoding='utf-8-sig')
    df_cleaned_ANSWERS.to_csv(outoput_file_ANSWERS, index=False, sep=';', encoding='utf-8-sig')
    df_cleaned_GENDER.to_csv(output_file_gender, index=False, sep=';', encoding='utf-8-sig')

    print("All done!")
#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds