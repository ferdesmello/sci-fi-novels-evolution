"""
This script uses Gemini 2.0, via the Google API, to answer questions about the gender 
of the authors of the books scraped before, parses the answers and saves it.

Modules:
    - os
    - dotenv
    - google
    - pandas
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

#----------------------------------------------------------------------------------
# Load environment variables from the .env file
load_dotenv(dotenv_path='../../KEYs/My_GOOGLE_API_Key.env')

# Get the API key from the environment variable
api_key = os.environ.get("GOOGLE_API_KEY")

# Configure the genai library with the loaded API key
if api_key:
    genai.configure(api_key=api_key)
    print("Google Gemini API key loaded successfully.")
    # Now you can proceed to use the genai library
    model = genai.GenerativeModel('gemini-pro')
    # ... your code to interact with the model ...
else:
    print("Error: GOOGLE_API_KEY not found in the environment variables. Make sure your .env file is correctly configured.")

# Define your default generation configuration
default_generation_config = {
    'max_output_tokens': 10,  # Set your desired default max tokens
    'temperature': 0.2        # Set your desired default temperature
}

# Initialize the Gemini model with the default configuration
model = genai.GenerativeModel(model_name='gemini-2.0-flash', generation_config=default_generation_config)


#----------------------------------------------------------------------------------
def analyze_author(author: str) -> str:
    """
    Prompts Gemini 2.0 to analyse the author gender of each book.

    Args:
        author (str): Book author.

    Returns:
        answer (str): Gemini 2.0 processed gender answer.
    """

    # Create the prompt with the gathered data
    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes the gender of writers based on your own knowledge about them and their names.
    Consider the writer {author}.
    What is {author} gender?
        Male; 
        Female; 
        Other; 
        Uncertain.
    Please, give only one-word answers and only from the four alternatives given, following the example:
    Male
    """
    
    # Extract and print the response
    response = model.generate_content(prompt)
    # Check if the response and text attribute exist
    if response and hasattr(response, 'text'):
        answer = response.text
    else:
        answer = None

    print(f'{author} is {answer}')
    #print(prompt)
    #print(answer)

    return answer

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.
    """
    
    # Name of the input file
    input_file = '../Data/Filtered/sci-fi_books_TOP_Wiki.csv'
    #input_file = 'sci-fi_books_AI_ANSWERS_Gemini.csv'

    # Name of the output file
    output_file = 'sci-fi_books_AI_ANSWERS_GENDER_Gemini.csv'

    #----------------------------------------------------------------------------------
    # Load book data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")
    #print(df.info())

    authors_list = list(set(df['author'].str.strip()))
    print("length =",len(authors_list))

    #----------------------------------------------------------------------------------
    # Main operation

    # Create a list to store the results
    results = []
    number = 0

    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_authors = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
    else:
        df_authors = pd.DataFrame(columns=['author', 'gender'])

    processed_authors = set(df_authors['author'].values)

    # Iterate through the author names and query Gemini 2.0
    for name in authors_list:
        # Skip already processed authors
        if name in processed_authors:
            continue
        
        gender_exit = analyze_author(name)
        lines = gender_exit.split('\n')
        gender = lines[0]
        results.append((name, gender))
        number += 1

    print(f"\nAdded {number} author(s) and their gender to the list.\n")

    # Convert to a DataFrame and add it to the end of the present DataFrame
    df_added_authors = pd.DataFrame(results, columns=['author', 'gender'])
    df_authors = pd.concat([df_authors, df_added_authors], ignore_index=True)

    df_authors = df_authors.sort_values(by=['gender', 'author'], ascending=True)
    df_authors = df_authors.reset_index(drop=True)

    #------------------------------------------
    Count = df_authors.value_counts(subset='gender', normalize=False, sort=True, ascending=False, dropna=True)
    Fraction = df_authors.value_counts(subset='gender', normalize=True, sort=True, ascending=False, dropna=True).mul(100).round(1)

    print(df_authors.info())
    print("\n",Count)
    print("\n",Fraction)

    #----------------------------------------------------------------------------------
    df_authors.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nData saved to {output_file}")

    for row in df_authors['author']:
        if row not in authors_list:
            print(f"\nCheck this extra author name: {row}.")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()