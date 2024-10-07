#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/






import configparser
import main
import config_classes as CC
import LLM_functuions as LF
import warnings





"""
Main function to run the LLM comparison pipeline.
"""
def run(config):
    
    ##################################################
    # Extract API keys from the configuration
    api_keys_list = {
        'gemini_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.GOOGLEKEY]}",
        'palm_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.GOOGLEKEY]}",
        'bert_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.GOOGLEKEY]}",
        'gpt_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.OPENAIKEY]}",
        'meta_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.METAKEY]}",
        'microsoft_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.MICROSOFTKEY]}",
        'amazon_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.AMAZONKEY]}",
        'apple_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.APPLEKEY]}",
        'bloom_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.BLOOMKEY]}",
        'falcon_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.FALCONKEY]}",
        'cohere_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.COHEREKEY]}",
        'mistral_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.MISTRAKEY]}",
        'ai21_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.AI21KEY]}",
        'cmu_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.CMUKEY]}",
        'alibaba_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.ALIBABAKEY]}",
        'shanghai_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.SHANGHAIKEY]}",
        'databricks_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.DATABRICKSKEY]}",
        'bigcode_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.BIGCODEKEY]}",
        'stability_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.STABILITYKEY]}",
        'zhipuai_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.ZHIPUAIKEY]}",
        'whylabs_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.WHYLABKEY]}"
    }
    ##################################################

    ##################################################
    # Create a list of your LLM response functions
    llm_functions = [
        LF.gemini_prompt_func,
        LF.palm_prompt_func,
        LF.bert_prompt_func,
        LF.gpt_prompt_func,
        LF.meta_prompt_func,
        LF.micro_prompt_func,
        LF.amazon_prompt_func,
        LF.apple_prompt_func,
        LF.bloom_prompt_func,
        LF.falcon_prompt_func,
        LF.cohere_prompt_func,
        LF.mistral_prompt_func,
        LF.ai21_labs_prompt_func,
        LF.cmu_princeton_prompt_func,
        LF.alibaba_prompt_func,
        LF.shanghai_ai_prompt_func,
        LF.databricks_prompt_func,
        LF.bigcode_prompt_func,
        LF.stablity_prompt_func,
        LF.zhipuai_prompt_func,
        LF.whylabs_prompt_func
    ]
    ##################################################

    ##################################################
    # Create a list of your LLM model names
    llm_model_names = {
        'gemini': f"{config[CC.LLMModels.MODEL][CC.LLMModels.GEMINI]}",
        'palm': f"{config[CC.LLMModels.MODEL][CC.LLMModels.PALM]}",
        'bert': f"{config[CC.LLMModels.MODEL][CC.LLMModels.BERT]}",
        'gpt': f"{config[CC.LLMModels.MODEL][CC.LLMModels.OPENAI]}",
        'meta': f"{config[CC.LLMModels.MODEL][CC.LLMModels.META]}",
        'microsoft': f"{config[CC.LLMModels.MODEL][CC.LLMModels.MICROSOFT]}",
        'amazon': f"{config[CC.LLMModels.MODEL][CC.LLMModels.AMAZON]}",
        'apple': f"{config[CC.LLMModels.MODEL][CC.LLMModels.APPLE]}",
        'bloom': f"{config[CC.LLMModels.MODEL][CC.LLMModels.BLOOM]}",
        'falcon': f"{config[CC.LLMModels.MODEL][CC.LLMModels.FALCON]}",
        'cohere': f"{config[CC.LLMModels.MODEL][CC.LLMModels.COHERE]}",
        'mistral': f"{config[CC.LLMModels.MODEL][CC.LLMModels.MISTRA]}",
        'ai21': f"{config[CC.LLMModels.MODEL][CC.LLMModels.AI21]}",
        'cmu': f"{config[CC.LLMModels.MODEL][CC.LLMModels.CMU]}",
        'alibaba': f"{config[CC.LLMModels.MODEL][CC.LLMModels.ALIBABA]}",
        'shanghai': f"{config[CC.LLMModels.MODEL][CC.LLMModels.SHANGHAI]}",
        'databricks': f"{config[CC.LLMModels.MODEL][CC.LLMModels.DATABRICKS]}",
        'bigcode': f"{config[CC.LLMModels.MODEL][CC.LLMModels.BIGCODE]}",
        'stability': f"{config[CC.LLMModels.MODEL][CC.LLMModels.STABILITY]}",
        'zhipu': f"{config[CC.LLMModels.MODEL][CC.LLMModels.ZHIPU]}",
        'whylab': f"{config[CC.LLMModels.MODEL][CC.LLMModels.WHYLAB]}"
    }
    ##################################################

    ##################################################
    # Combine all data into a single dictionary
    combined_data = {
        'api_keys': api_keys_list,
        'llm_functions': llm_functions,
        'llm_model_names': llm_model_names
    }
    ##################################################

    ##################################################
    # Construct paths
    paths_dict = main.construct_paths(config)

    # Validate the input folder exists 
    main.validate_paths(paths_dict)
    
    # Ensure the output directory exists
    for key, path in paths_dict.items():
        if 'path' in key:  # Assuming keys containing 'path' are directories
            main.ensure_directory(path)
    ##################################################

    ##################################################
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
    ##################################################

    ##################################################
    # Extract paths from the paths_dict
    # input_folder_path = paths_dict['input_folder_path']
    output_folder_path = paths_dict['output_folder_path']
    
    # Step 1: Generate 100 x_i
    # Number of x_i prompts to generate
    num_samples = 100
    # Consider gpt-3.5-turbo model from O penAI API as considered text generator function
    considered_txt_generator = LF.gpt_prompt_func
    # Considered model for the considered text generator
    considered_model = llm_model_names['gpt']
    considered_api_key = api_keys_list['gpt_api_key']
    x_i_list = main.generate_input_texts(num_samples, considered_txt_generator, considered_model, considered_api_key)

    # Step 2: Generate completions using 20 LLMs
    # Number of completions to generate for eac of the x_i and LLM pairs
    num_completions = 100
    outcomes = main.generate_llm_completions(x_i_list, llm_functions, llm_model_names, api_keys_list, num_completions)

    # Step 3: Save the results to a file
    main.save_outcomes(outcomes, output_folder_path)
    
    # Handling warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")




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

    