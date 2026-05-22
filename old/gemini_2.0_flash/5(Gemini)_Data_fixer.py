"""
This script cleans the data if new data have been included tardily.

Modules:
    - pandas
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
import pandas as pd

#----------------------------------------------------------------------------------
def main():
    """
    Big main function with all the cleaning of the data if new data was added.
    """

    df_top = pd.read_csv('../Data/Filtered/sci-fi_books_TOP_Wiki.csv', sep = ';', encoding="utf-8-sig")
    df_AI = pd.read_csv('sci-fi_books_AI_ANSWERS_Gemini.csv', sep = ';', encoding="utf-8-sig")
    df_AI_gender = pd.read_csv('sci-fi_books_AI_ANSWERS_GENDER_Gemini.csv', sep = ';', encoding="utf-8-sig")

    #----------------------------------------------------------------------------------
    # Cleaning the sci-fi_books_AI_ANSWERS.csv file, so it has only sci-fi_books_TOP.csv books.

    column_names = df_AI.columns
    df_cleaned = pd.DataFrame(columns = column_names)
    books = set(df_top['url goodreads'])
    counter = 0

    for _, row in df_AI.iterrows(): # index, row
        if row['url goodreads'] in books:
            df_cleaned = pd.concat([df_cleaned, row.to_frame().T], ignore_index=True)
        else:
            counter += 1
            continue

    print(f"{counter} book(s) have been excluded from sci-fi_books_AI_ANSWERS_Gemini.csv.")

    #----------------------------------------------------------------------------------
    # Cleaning the sci-fi_books_AI_ANSWERS_GENDER.csv file, so it has only sci-fi_books_TOP.csv authors.

    column_names_gender = df_AI_gender.columns
    df_cleaned_gender = pd.DataFrame(columns = column_names_gender)
    names = set(df_top['author'])
    counter_gender = 0

    for _, row_gender in df_AI_gender.iterrows(): # index_gender, row_gender
        if row_gender['author'] in names:
            df_cleaned_gender = pd.concat([df_cleaned_gender, row_gender.to_frame().T], ignore_index=True)
        else:
            counter_gender += 1
            continue

    print(f"{counter_gender} name(s) have been excluded from sci-fi_books_AI_ANSWERS_GENDER_Gemini.csv.")


    #----------------------------------------------------------------------------------
    df_cleaned.to_csv('sci-fi_books_AI_ANSWERS_Gemini.csv', index=False, sep=';', encoding='utf-8-sig')
    df_cleaned_gender.to_csv('sci-fi_books_AI_ANSWERS_GENDER_Gemini.csv', index=False, sep=';', encoding='utf-8-sig')

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()