#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/





import os
import csv
import logging
import configparser
from pathlib import Path
import config_classes as CC
import LLM_functuions as LF






"""
    Constructs and returns a dictionary of paths based on the configuration.
"""
def construct_paths(config):
    paths_dict = {}
    # Construct paths
    folders = [CC.InputFiles.INPUTFOLDER, CC.OutputFiles.OUTPUTFOLDER]
    for folder in folders:
        paths_dict[f"{folder}_path"] = os.path.join(config[CC.InputFiles.LOCATIONS][folder])    
    return paths_dict





"""
    Ensures that the path exists as a directory. If the path exists but is a file,
    logs a warning and does not attempt to create the directory.
"""
def ensure_directory(path):
    # Check if the path exists
    if os.path.exists(path):
        if os.path.isfile(path):
            logging.warning(f"Expected a directory but found a file: {path}")
        # If it"s a directory, there"s nothing more to do
    else:
        # Path does not exist, create the directory
        os.makedirs(path, exist_ok=True)





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
    Generate random truncated texts (x_i) using gpt-3.5-turbo model from OpenAI API 
"""
def generate_input_texts(num_samples, considered_txt_generator, considered_model,api_key):
    x_i_list = []
        # Prompt for the considered text generator
    x_i_prompt = f"Generate {num_samples} sets of random truncated texts x_i such as “Yesterday I went”"
    for i in range(num_samples):
        x_i = considered_txt_generator(x_i_prompt, considered_model, api_key)
        x_i_list.append(x_i)
    return x_i_list





"""
    Generate completions for each x_i using different LLMs
"""
def generate_llm_completions(x_i_list, llm_functions, llm_model_names, api_keys_list, num_completions):   
    outcomes = {}
    for idx, x_i in enumerate(x_i_list):
        print(f"Generating completions for x_i[{idx+1}]: {x_i}")
        outcomes[x_i] = {}
        for llm_model_name, llm_func, api_key in zip(llm_model_names, llm_functions, api_keys_list):
            outcomes[x_i][llm_model_name] = []
            for _ in range(num_completions):
                try:
                    # Prompt to complete truncated x_i text
                    x_j_prompt = f"Given the truncated text '{x_i}', complete it by appending x_j to get a complete text."
                    # Use each LLM to generate x_j for the given x_i
                    x_j = llm_func(x_j_prompt, llm_model_name, api_key)
                    complete_text = f"{x_i} {x_j}"
                    outcomes[x_i][llm_model_name].append(complete_text)
                except Exception as e:
                    print(f"Error in model {llm_model_name}: {str(e)}")
                    outcomes[x_i][llm_model_name].append("Error generating completion")
    return outcomes





"""
    Save the generated x_i and x_j pairs to a file
"""
def save_outcomes(outcomes, output_file_path):
    with open(output_file_path, "w", newline='') as csvfile:
        fieldnames = ['input_text', 'llm_model', 'completion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for input_text, completions in outcomes.items():
            for llm_model, completion in completions.items():
                writer.writerow({'input_text': input_text, 'llm_model': llm_model, 'completion': completion})
    
    print(f"Results saved to {output_file_path}")

