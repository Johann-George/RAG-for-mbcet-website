import json
import re

def extract_title_and_content(json_file_path):
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1']  # List of encodings to try

    for encoding in encodings:
        try:
            # Try to open and read the JSON file with the current encoding
            with open(json_file_path, 'r', encoding=encoding) as file:
                data = json.load(file)
            print(f"File successfully read with encoding: {encoding}")
            break  # Exit the loop if successful
        except UnicodeDecodeError:
            print(f"Failed to read file with encoding: {encoding}")
        except FileNotFoundError:
            print(f"Error: The file {json_file_path} was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: The file {json_file_path} is not a valid JSON.")
            return None
    else:
        print("Error: Could not read the file with any of the specified encodings.")
        return None

    titles_and_contents = []

    # Function to clean text by removing special symbols
    def clean_text(text):
        return re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Iterate through the list and extract title and content from each element
    for item in data:
        if isinstance(item, dict):
            title = item.get('metadata', {}).get('title', 'No title found')
            content = item.get('content', 'No content found')

            # Clean title and content to remove special symbols
            clean_title = clean_text(title)
            clean_content = clean_text(content)

            titles_and_contents.append((clean_title, clean_content))

    return titles_and_contents

def save_to_text_file(titles_and_contents, text_file_path):
    try:
        with open(text_file_path, 'w', encoding='utf-8') as file:
            for title, content in titles_and_contents:
                file.write(f"{title}\n\n")
                file.write(f"\n{content}\n\n")
        print(f"Data successfully written to {text_file_path}")
    except Exception as e:
        print(f"Error: Could not write to file {text_file_path}. Exception: {str(e)}")

# Example usage
json_file_path = 'C:/Users/johan/PycharmProjects/VectorDB_Intro/results.json'
text_file_path = 'C:/Users/johan/PycharmProjects/VectorDB_Intro/mbcet_website_data.txt'
titles_and_contents = extract_title_and_content(json_file_path)

if titles_and_contents:
    save_to_text_file(titles_and_contents, text_file_path)
