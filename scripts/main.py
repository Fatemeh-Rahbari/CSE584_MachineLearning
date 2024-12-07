#title: runner.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-November-14 15:55:38
#------------------------------------*/





import os
import csv
import pandas as pd
import configparser
from pathlib import Path
import config_classes as CC
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint as pp
from gspread import WorksheetNotFound
import time




"""
    Constructs and returns a dictionary of paths based on the configuration.
"""
def construct_paths(config):
    paths_dict = {}
    # Construct paths
    folders = [CC.InputFiles.INPUTFOLDER, CC.OutputFiles.OUTPUTFOLDER]
    for folder in folders:
        paths_dict[f"{folder}_path"] = os.path.join(config[CC.InputFiles.LOCATION][folder])    
    return paths_dict

"""
    Validates that all paths in the paths_dict exist.
"""
def validate_paths(paths_dict):
        # Check if all paths exist
        try:
            for path in paths_dict.values():
                if not Path(path).exists():
                    raise FileNotFoundError(f"Path {path} does not exist.")
        except KeyError as e:
            print(f"Missing configuration for {e}")
            return
        except FileNotFoundError as e:
            print(e)
            return

"""
    Reads the configuration file and returns the configuration object.
"""
def read_configuration (config_file_name):
    config= None
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file_name)
    return config

"""
    Reads the input data from a CSV file and returns a DataFrame.
"""
def csv_reader(input_file_name, input_file_path):
    input_file_path = os.path.join(input_file_path, input_file_name)
    try:
        input_data = pd.read_csv(input_file_path, encoding='latin1')
    except UnicodeDecodeError:
        input_data = pd.read_csv(input_file_path, encoding='iso-8859-1')
    return input_data

"""
Reads from Google sheets and returns a specific column.
"""
def google_sheet_reader(json_path, input_Google_Sheet_Name, Google_Worksheet_Name):
    # Set up Google Sheets API credentials
    scope = ["https://spreadsheets.google.com/feeds", 
             "https://www.googleapis.com/auth/spreadsheets", 
             "https://www.googleapis.com/auth/drive.file", 
             "https://www.googleapis.com/auth/drive"]

    # Construct the full path to the JSON file using os.path.join
    #credentials_path = os.path.join(os.getcwd(), json_path)
    credentials_path = json_path
    
    # Update to use credentials_path
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open(input_Google_Sheet_Name).worksheet(Google_Worksheet_Name)

    # Extract the data: "Sub-Discipline", "Faulty question", and "Actual correct question" columns
    sub_discipline_col = sheet.find("Sub Discipline").col
    faulty_question_col = sheet.find("Faulty question").col

    # Read data from the respective columns
    sub_discipline_data = sheet.col_values(sub_discipline_col)[1:]  # Exclude headers
    faulty_question_data = sheet.col_values(faulty_question_col)[1:]  # Exclude headers
    return sub_discipline_data, faulty_question_data

"""
    Save the generated faulty questions to Google Sheets
"""
def save_Google_sheet(result_df, json_path, input_Google_Sheet_Name, result_Google_Sheet_Name, Google_Worksheet_Name):

    # Set up Google Sheets API credentials
    scope = ["https://spreadsheets.google.com/feeds", 
             "https://www.googleapis.com/auth/spreadsheets", 
             "https://www.googleapis.com/auth/drive.file", 
             "https://www.googleapis.com/auth/drive"]

    # Authorize the credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
    client = gspread.authorize(creds)

    try:
        # Open the Google Sheet
        sheet = client.open(input_Google_Sheet_Name).worksheet(Google_Worksheet_Name)
    except WorksheetNotFound:
        print(f"Error: Worksheet '{Google_Worksheet_Name}' not found in '{input_Google_Sheet_Name}'.")
        return

    # Get the headers from the Google Sheet
    headers = sheet.row_values(1)

    # Ensure the DataFrame columns match the Google Sheet headers
    if not all(col in headers for col in result_df.columns):
        print("Error: DataFrame columns do not match Google Sheet headers.")
        return

    # Find the column indices in the Google Sheet for the DataFrame columns
    col_indices = {col: headers.index(col) + 1 for col in result_df.columns}

    # Update the Google Sheet with the DataFrame data
    for i, row in result_df.iterrows():
        for col, value in row.items():
            cell = gspread.utils.rowcol_to_a1(i + 2, col_indices[col])  # +2 to account for header row
            sheet.update_acell(cell, str(value))
            time.sleep(1)  # Add a delay of 1 second between requests to avoid quota limit
    print(f"Results saved to Google Sheets: {result_Google_Sheet_Name} - {Google_Worksheet_Name}")
       
"""
    Save the generated faulty questions to a CSV file
"""
def save_outputs(result_df, output_file_name, output_file_path):
     output_file_path = os.path.join(output_file_path, output_file_name)
     with open(output_file_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_df.columns)
        for i, row in result_df.iterrows():
            writer.writerow(row)
        print(f"Results saved to {output_file_path}")
