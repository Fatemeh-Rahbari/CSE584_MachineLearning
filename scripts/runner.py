#title: runner.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-November-14 15:55:38
#------------------------------------*/





import main
import LLMs
import configparser
import time
import pandas as pd
import config_classes as CC
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from os.path import join
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import sys
import os





"""
Main function to run the LLM comparison pipeline.
"""
def run(config):
    
    ####### Step 1: Set up the API keys and LLM functions #######
    
    # Extract API keys from the configuration
    api_keys_list = {
        'gpt_api_key': f"{config[CC.Apikeys.APIKEY][CC.Apikeys.OPENAIKEY]}",
        'mistral_api_key': f"{config[CC.Apikeys.APIKEY][CC.Apikeys.MISTRALKEY]}",
        'ai21_api_key': f"{config[CC.Apikeys.APIKEY][CC.Apikeys.AI21KEY]}",
        'cohere_api_key': f"{config[CC.Apikeys.APIKEY][CC.Apikeys.COHEREKEY]}"
    }

    # Create a list of your LLM response functions
    llm_functions = [
        LLMs.gpt_prompt_func,
        LLMs.mistral_prompt_func,
        LLMs.ai21_labs_prompt_func,
        LLMs.cohere_prompt_func
    ]

    # Create a list of your LLM model names
    llm_model_names = {
        'gpt': f"{config[CC.LLMModels.MODEL][CC.LLMModels.OPENAI]}",
        'mistral': f"{config[CC.LLMModels.MODEL][CC.LLMModels.MISTRAL]}",
        'ai21': f"{config[CC.LLMModels.MODEL][CC.LLMModels.AI21]}",
        'cohere': f"{config[CC.LLMModels.MODEL][CC.LLMModels.COHERE]}"
    }

    # Construct paths
    paths_dict = main.construct_paths(config)

    # Validate all the paths exist 
    main.validate_paths(paths_dict)
    
    # Check if the API key is present in the configuration
    for key, value in api_keys_list.items():
        if not value:
            raise ValueError(f"API key for {key} is missing.")
    
     # Check if the LLM models is present in the configuration
    for key, value in llm_model_names.items():
        if not value:
            raise ValueError(f"LLM model for {key} is missing.")
        
    # Check if the LLM functions is present in the configuration
    for func in llm_functions:
        if not func:
            raise ValueError("One of the LLM functions is missing.")
    
    # Extract paths from the paths_dict
    input_folder_path = paths_dict['input_folder_path']
    output_folder_path = paths_dict['output_folder_path']

    ####### Step 2: Get the input data of disciplines from a CSV file or google sheet #######
    
    # Download your Google Cloud service account credentials as a JSON file 
    # and save it to your local machine, for example: 'credentials.json'
    # Replace 'credentials.json' with the actual path to your downloaded file.
    your_Google_credential_json_name = f"{config[CC.GoogleSheets.GOOGLESHEET][CC.GoogleSheets.CREDENTIALS]}"
    your_Google_credential_json_Path = join(input_folder_path, your_Google_credential_json_name)
    
    # Open the Google Sheet
    your_Google_sheet_name = f"{config[CC.GoogleSheets.GOOGLESHEET][CC.GoogleSheets.INPUTSHEETNAME]}"
    result_Google_sheet_name = f"{config[CC.GoogleSheets.GOOGLESHEET][CC.GoogleSheets.RESULTSHEETNAME]}"
    worksheet_name = f"{config[CC.GoogleSheets.GOOGLESHEET][CC.GoogleSheets.WORKSHEETNAME]}"
    input_Googlesheet_id = f"{config[CC.GoogleSheets.GOOGLESHEET][CC.GoogleSheets.GOOGLESHEETID]}"
    
    # Get data from CSV files
    input_csv_name = f"{config[CC.InputFiles.FILE][CC.InputFiles.INPUTCSVNAME]}"
    output_ques_csv_name = f"{config[CC.InputFiles.FILE][CC.OutputFiles.OUTPUTQUESTIONSCSVNAME]}"
    output_ans_csv_name = f"{config[CC.InputFiles.FILE][CC.OutputFiles.OUTPUTANSWERSCSVNAME]}"
    experiment_csv_name = f"{config[CC.InputFiles.FILE][CC.OutputFiles.EXPERIMENTCSVNAME]}"
    experiment_results_csv_name = f"{config[CC.InputFiles.FILE][CC.OutputFiles.EXPERIMENTRESULTSCSVNAME]}"

    ####### Step 3: Generate faulty completions using 4 LLMs #######
    
    # # Generate completions for each discipline using different LLMs
    for i, (api_key, llm_func, model) in enumerate(zip(api_keys_list, llm_functions, llm_model_names), start=1):
        
        output_data = []
        
        # CSV data reader
        # input_data = main.csv_reader(f"{model}_{input_csv_name}", input_folder_path)

        # Google Sheet data reader
        input_worksheet_name = worksheet_name + f"{i}"
        print(f"itteration: {i}")
        print(input_worksheet_name)
        input_data = main.google_sheet_reader(your_Google_credential_json_Path, your_Google_sheet_name, input_worksheet_name)
        apikey = api_keys_list[api_key]
        model_name = llm_model_names[model]
        print(f"{model} model is being used.")
        for idx, sub_discipline in enumerate(input_data[0]):
            print(f"Generating completions for x_[{idx+1}]: {sub_discipline} \n")
            faulty_qus_prompt = f" I want you to generate intentionally faulty question for the following sub-discipline: {sub_discipline}. This question is an example for you to understand the idea and conception behind what a faulty question can be: 'Lily received 3 cookies from her best friend yesterday and ate 5 for breakfast. Today, her friend gave her 3 more cookies. How many cookies does Lily have now?' This example is a faulty question in math because Lily can not eat 5 cookies when she only has 3. Please get the logic behind the example and generate creative faulty question. Please, pay attention, not include either the reason or other extra wordsm, just list one question."
            try:
                # Sleep for 60 seconds after every 20 iterations
                # if (idx + 1) % 10 == 0:
                #     print("Sleeping for 60 seconds...")
                #     time.sleep(60)
                # else:
                    faulty_question = llm_func(apikey, model_name, faulty_qus_prompt)
                    if faulty_question:
                        output_data.append({
                            "Sub Discipline": sub_discipline,
                            "Faulty question": faulty_question,
                            "The name of LLM model": model
                        })
                    else:
                        print(f"Waiting for LLM model response for x_[{idx+1}]: {sub_discipline}")
                        time.sleep(60)
            except Exception as e:
                print(f"Error generating completions for x_[{idx+1}]: {sub_discipline}")
                print(e)
                continue
        # Convert to DataFrame and display/save the output
        df = pd.DataFrame(output_data)

        # Save the outputs to a CSV file
        output_file_name = f"{model}_{output_ques_csv_name}"
        main.save_outputs(df, output_file_name, output_folder_path)
        print("The faulty questions generation results have been saved as CSV file successfully.")

        # # Save the outputs to Google Sheets
        output_worksheet_name = worksheet_name + f"{i}"
        main.save_Google_sheet(df, your_Google_credential_json_Path, your_Google_sheet_name, result_Google_sheet_name, output_worksheet_name)
        print("The faulty questions generation results have been saved in the Google Spreedsheet successfully.")

    ####### Step 4: Generate answer completions to distinct faulty questions using 4 LLMs #######
    
    # # Generate answer completions for each faulty question using the LLMs to see if they know the question is faulty or not
    for i, (api_key, llm_func, model) in enumerate(zip(api_keys_list, llm_functions, llm_model_names), start=1):
        output_data = []
        
        # CSV data reader
        input_data = main.csv_reader(f"{model}_{output_ques_csv_name}", output_folder_path)

        # Google Sheet data reader
        input_worksheet_name = worksheet_name + f"{i+1}"
        print(f"itteration: {i}")
        print(input_worksheet_name)
        input_data = main.google_sheet_reader(your_Google_credential_json_Path, your_Google_sheet_name, input_worksheet_name)

        apikey = api_keys_list[api_key]
        model_name = llm_model_names[model]
        print(f"{model} model is being used.")
        
        for idx, sub_discipline in enumerate(input_data[0]):
            print(f"Generating completions for x_[{idx+1}]: {sub_discipline} \n")
            print(f"index is: {idx}")
            print(f"The length of Faulty questions columns is {len(input_data[1])}")
            faulty_question = input_data[1][idx]
            faulty_qus_prompt = f" Answer the following question '{faulty_question}'in a very short way."
            try:
                answer_faulty_question = llm_func(apikey, model_name, faulty_qus_prompt)
                if answer_faulty_question:
                    output_data.append({
                            "Sub Discipline": sub_discipline,
                            "Faulty question": faulty_question,
                            "Response by a top LLM": answer_faulty_question,
                            "The name of LLM model": model,
                            "Which top LLM you tried": model
                    })
                else:
                    print(f"Waiting for LLM model response for x_[{idx+1}]: {sub_discipline}")
                    time.sleep(60)
            except Exception as e:
                print(f"Error generating answer completions for x_[{idx+1}]: {sub_discipline}")
                print(e)
                continue
        # Convert to DataFrame and display/save the output
        df = pd.DataFrame(output_data)

        # Save the outputs to a CSV file
        output_file_name = f"{model}_{output_ans_csv_name}"
        main.save_outputs(df, output_file_name, output_folder_path)
        print("The answer to faulty questions results have been saved as CSV file successfully.")

        # # Save the outputs to Google Sheets
        output_worksheet_name = worksheet_name + f"{i}"
        main.save_Google_sheet(df, your_Google_credential_json_Path, your_Google_sheet_name, result_Google_sheet_name, output_worksheet_name)
        print("The answer to faulty questions results have been saved in the Google Spreedsheet successfully.")

    ####### Step 5: Analyze the research question "What factors influence whether an LLM is fooled by a faulty question?" #######
    for i, (api_key, llm_func, model) in enumerate(zip(api_keys_list, llm_functions, llm_model_names), start=1):
        output_data = []
        
        # CSV data reader
        input_data = main.csv_reader(f"{model}_{input_csv_name}", output_folder_path)

        # Google Sheet data reader
        input_worksheet_name = worksheet_name + f"{i+1}"
        print(f"itteration: {i}")
        print(input_worksheet_name)
        input_data = main.google_sheet_reader(your_Google_credential_json_Path, your_Google_sheet_name, input_worksheet_name)

        apikey = api_keys_list[api_key]
        model_name = llm_model_names[model]
        print(f"{model} model is being used.")
        complexities = range(1, 4)
        ambiguities = range(1, 4)
        domain_knowledges = range(1, 4)
        logical_errors = range(1, 4)
        # for idx, sub_discipline in enumerate(input_data[0]):
        for idx, sub_discipline in enumerate(input_data['Sub Discipline']):
            print(f"Generating completions for x_[{idx+1}]: {sub_discipline} \n")
            print(f"index is: {idx}")
            for complexity in complexities:
                for ambiguity in ambiguities:
                    for domain_knowledge in domain_knowledges:
                        for logical_error in logical_errors:
                            faulty_qus_prompt = f"Generate a faulty science question in the field of {sub_discipline} with complexity level {complexity}/4, ambiguity level {ambiguity}/4, domain knowledge level {domain_knowledge}/4, and logical error level {logical_error}/4.'"
                            try:
                                faulty_question = llm_func(apikey, model_name, faulty_qus_prompt)
                                if faulty_question:
                                    output_data.append({
                                            "Sub Discipline": sub_discipline,
                                            "Faulty question": faulty_question,
                                            "Complexity": complexity,
                                            "Ambiguity": ambiguity,
                                            "Domain Knowledge": domain_knowledge,
                                            "Logical Errors": logical_error,
                                            "The name of LLM model": model

                                    })
                                else:
                                    print(f"Waiting for LLM model response for x_[{idx+1}]: {sub_discipline}")
                                    time.sleep(60)
                            except Exception as e:
                                print(f"Error generating answer completions for x_[{idx+1}]: {sub_discipline}")
                                print(e)
                                continue
        # Convert to DataFrame and display/save the output
        df = pd.DataFrame(output_data)

        # Save the outputs to a CSV file
        output_file_name = f"{model}_{experiment_csv_name}"
        main.save_outputs(df, output_file_name, output_folder_path)
        print("The answer to faulty questions results have been saved as CSV file successfully.")

        # # Save the outputs to Google Sheets
        output_worksheet_name = worksheet_name + f"{i}"
        main.save_Google_sheet(df, your_Google_credential_json_Path, your_Google_sheet_name, result_Google_sheet_name, output_worksheet_name)
        print("The answer to faulty questions results have been saved in the Google Spreedsheet successfully.")

####### Step 6: Test the research question "What factors influence whether an LLM is fooled by a faulty question?" #######
    for i, (api_key, llm_func, model) in enumerate(zip(api_keys_list, llm_functions, llm_model_names), start=1):
        output_data = []
        
        # CSV data reader
        input_data = main.csv_reader(f"{model}_{experiment_csv_name}", output_folder_path)

        # Google Sheet data reader
        input_worksheet_name = worksheet_name + f"{i+1}"
        print(f"itteration: {i}")
        print(input_worksheet_name)
        input_data = main.google_sheet_reader(your_Google_credential_json_Path, your_Google_sheet_name, input_worksheet_name)

        apikey = api_keys_list[api_key]
        model_name = llm_model_names[model]
        print(f"{model} model is being used.")
        
        # for idx, sub_discipline in enumerate(input_data[0]):
        for idx, sub_discipline in enumerate(input_data['Sub Discipline']):
            print(f"Generating completions for x_[{idx+1}]: {sub_discipline} \n")
            print(f"index is: {idx}")
            faulty_question = input_data['Faulty question'][idx]
            faulty_qus_prompt = f"Is the following question '{faulty_question}' faulty? Answer with 'Yes' or 'No'"
            try:
                answer_faulty_question = llm_func(apikey, model_name, faulty_qus_prompt)
                if answer_faulty_question:
                    answer_faulty_question =='1' if answer_faulty_question == 'No' else '0'  # 1 if fooled, 0 if not fooled
                    output_data.append({
                        "Sub Discipline": sub_discipline,
                        "Faulty question": faulty_question,
                        "The name of LLM model": model,
                        "Reason you think it is faulty": answer_faulty_question
                    })
                else:
                    print(f"Waiting for LLM model response for x_[{idx+1}]: {sub_discipline}")
                    time.sleep(60)
            except Exception as e:
                print(f"Error generating answer completions for x_[{idx+1}]: {sub_discipline}")
                print(e)
                continue
        # Convert to DataFrame and display/save the output
        df = pd.DataFrame(output_data)

        # Save the outputs to a CSV file
        output_file_name = f"{model}_{experiment_results_csv_name}"
        main.save_outputs(df, output_file_name, output_folder_path)
        print("The answer to faulty questions results have been saved as CSV file successfully.")

        # # Save the outputs to Google Sheets
        output_worksheet_name = worksheet_name + f"{i}"
        main.save_Google_sheet(df, your_Google_credential_json_Path, your_Google_sheet_name, result_Google_sheet_name, output_worksheet_name)
        print("The answer to faulty questions results have been saved in the Google Spreedsheet successfully.")

    ####### Step 7: Analyze the results #######
    for i, (api_key, llm_func, model) in enumerate(zip(api_keys_list, llm_functions, llm_model_names), start=1):  
        # CSV data reader
        input_data = main.csv_reader(f"{model}_{experiment_results_csv_name}", output_folder_path)
        
        # Prepare features
        X = input_data[['Complexity', 'Ambiguity', 'Domain Knowledge', 'Logical Errors']]
        y = input_data['Reason you think it is faulty']

        # Convert categorical variables to numeric
        X = pd.get_dummies(X, columns=['Complexity', 'Ambiguity', 'Domain Knowledge', 'Logical Errors'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Analyze feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)

        print("Top 10 most influential features:")
        print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(join(output_folder_path, f"{model}_feature_importance.png"))
    plt.show()

    # Save classification report to a text file
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(join(output_folder_path, f"{model}_classification_report.csv"), index=True)
    
"""
    Main function to run the microlearning generator.
"""
if __name__ == "__main__":
    config = None
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} configfile")
    elif not os.path.exists(sys.argv[1]):
        print(f"Can't find configuration file {sys.argv[1]}")
    else:
        #Read the configuration file
        config = main.read_configuration(sys.argv[1]) 
    if not config:
        print("Error, conf was 'None'.")
    else:
        run(config)
    run(config)
