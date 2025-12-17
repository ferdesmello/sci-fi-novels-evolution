"""
This script uses GPT-5, via the OPENAI API, to answer questions about the gender 
of the authors of the novels scraped before, parses the answers, and saves it.

Modules:
    - os
    - pandas
    - openai
    - dotenv
    - datetime
    - winsound (only for Windows)
    - pathlib
    - sys
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import winsound

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED, ANSWERS, KEYS

#----------------------------------------------------------------------------------
# Load environment variables from the .env file
load_dotenv(dotenv_path=KEYS / 'My_OPENAI_API_Key.env')

# Get the API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Configure the OpenAI client with the loaded API key
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI API key loaded successfully.")
else:
    print("Error: OPENAI_API_KEY not found in the environment variables. Make sure your .env file is correctly configured.")

#----------------------------------------------------------------------------------
def analyze_author(author: str) -> str:
    """
    Prompts GPT-5 to analyse the author gender of each novel.

    Args:
        author (str): novel author.

    Returns:
        answer (str): GPT-5 processed gender answer.
    """

    # Create the prompt with the gathered data
    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes novel plots based on your own knowledge and provided information.
    Consider the writer {author}.

    **Output Formatting Instructions**:
    Follow this exact format!
    Please, give only one-word answers without any punctuation marks and only from the set of four alternatives given.

    **Question**:
    What is the gender of author {author}?
        Male;
        Female;
        Other: Non-binary, genderfluid, ambiguous, or another gender identity that is not male nor female;
        Uncertain: Not enough information to say, or multiple authors of different genders.
    """
    
    # Call the OpenAI API with the crafted prompt
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=400,
            reasoning={"effort": "low"}, # can be "low", "medium", or "high"
            text={"verbosity":"low"} # can be "low", "medium", or "high"
        )

        # Extract and print the response
        answer = response.output_text

    except Exception as e:
        print("Error:", e)

    return answer

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.

    Returns:
        missing_authors (int): Number of authors not processed correctly.
    """
    
    # Name of the input file
    input_file = FILTERED / 'sci-fi_novels_TOP.csv'

    # Name of the output file
    output_file = ANSWERS / 'sci-fi_novels_AI_ANSWERS_GENDER.csv'

    #----------------------------------------------------------------------------------
    # Load novel data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")
    #print(df.info())

    authors_list = list(set(df['author'].str.strip()))
    print("Author's set length =", len(authors_list))

    #----------------------------------------------------------------------------------
    # Main operation

    # Determine if the file exists to write headers
    file_exists = os.path.exists(output_file)

    # Load existing progress if the file exists
    if file_exists:
        df_authors = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
    else:
        df_authors = pd.DataFrame(columns=['author', 'gender'])

    processed_authors = set(df_authors['author'].values)

    # Valid answers set
    answers_set = {
            "Male",
            "Female",
            "Other",
            "Uncertain"
            }
    
    #------------------------------------------
    number = 0

    # Iterate through the author names and query GPT-5
    for name in authors_list:
        # Skip already processed authors
        if name in processed_authors:
            continue
        
        gender = analyze_author(name)

        try:
            if gender.strip() in answers_set:
                print(f'  {name} is {gender}')
                new_row = pd.DataFrame([{'author': name, 'gender': gender.strip()}])
            
                # Append the new row to the CSV file
                new_row.to_csv(output_file, mode='a', header=not file_exists, index=False, sep=';', encoding='utf-8-sig')
                
                # The file now exists for subsequent writes
                file_exists = True
                number += 1

            else:
                print(f"Warning: Unexpected answer for {name}; {gender}")

        except Exception as e:
            print("Error:", e)

    print(f"\nAdded {number} author(s) and their gender(s) to the list.\n")

    #------------------------------------------
    # RE-READ and APPLY CORRECTIONS to the final DataFrame
    if os.path.exists(output_file):
        df_authors_final = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')

        gender_corrections = [
            # (author, correct_gender)
            ("Micaiah Johnson", "Female"),
            ("Kaliane Bradley", "Female"),
            ("Misba", "Female"),
            ("Murray Constantine", "Female"),
            ("N.E. Davenport", "Female"),
            ("T.A. White", "Female"),
            ("J.S. Dewes", "Female"),
            ("Joan He", "Female"),
            ("Lauren Thoman", "Female"),
            ("Sierra Greer", "Female"),
            ("Zoe Hana Mikuta", "Female"),

            ("A.R. Merrydew", "Male"),
            ("Blake Savage", "Male"),
            ("Brett Sterling", "Male"),
            ("Erik J. Brown", "Male"),
            ("J.M. Troska", "Male"),
            ("James Frey, Jobie Hughes", "Male"),
            ("Max Nowaz", "Male"),
            ("Skyler Ramirez", "Male"),
            ("Terry Miles", "Male"),
            ("Thomas R. Weaver", "Male"),
            ("Calvin Kasulke", "Male"),
            ("Edson McCann", "Male"),
            ("Frederik Pohl, Lester del Rey", "Male"),
            ("Owen Gregory", "Male"),
            ("Peter Brown", "Male"),
            ("K.M. Szpara", "Male"),

            ("Annalee Newitz", "Other"),
            ("Linden A. Lewis", "Other"),
            ("Nino Cipri", "Other"),
            ("Rivers Solomon", "Other"),
            ("Sarah Gailey", "Other"),
            ("Xiran Jay Zhao", "Other"),
            ("Hiron Ennes", "Other"),
            ("L.R. Lam", "Other"),
            ("Ness Brown", "Other"),
            ("Marisa Crane", "Other"),
            ("Kenneth Robeson", "Other"),

            ("Ilona Andrews", "Uncertain"),
            ("Ilona Gordon, Andrew Gordon", "Uncertain"),
            ("Trevor Alan Foris", "Uncertain"),
            ("Victor Appleton II", "Uncertain"),
        ]

        # Apply corrections to the final DataFrame
        for author, correct in gender_corrections:
            df_authors_final.loc[df_authors_final["author"] == author, "gender"] = correct

        # Clean, sort, and save the final DataFrame
        df_authors_final = df_authors_final.dropna(axis=0, subset=['gender'], how="any", ignore_index=True)
        df_authors_final = df_authors_final.sort_values(by=['gender', 'author'], ascending=True)
        df_authors_final = df_authors_final.reset_index(drop=True)
        df_authors_final.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
        
        length_in = len(authors_list)
        length_out = df_authors_final.shape[0]
        missing_authors = length_in - length_out

        return missing_authors, df_authors_final
    else:
        return len(authors_list), pd.DataFrame() # No file created, all authors are missing

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":

    # Record start time
    start = datetime.now()

    missing_authors = 1
    max_retries = 20
    attempt = 1

    while (missing_authors != 0) and (attempt <= max_retries):
        missing_authors, df_authors = main()
        print(f"Author(s) missing: {missing_authors}.\nAttempts made: {attempt}.")
        attempt += 1
        
    #------------------------------------------
    Count = df_authors.value_counts(subset='gender', normalize=False, sort=True, ascending=False, dropna=True)
    Fraction = df_authors.value_counts(subset='gender', normalize=True, sort=True, ascending=False, dropna=True).mul(100).round(1)

    print(df_authors.info())
    print("\n", Count)
    print("\n", Fraction)  

    # Record end time
    end = datetime.now()

    # How long did it take?
    print(f"Script started at {start}")
    print(f"Script finished at {end}")
    print(f"Total runtime: {end - start}")

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds