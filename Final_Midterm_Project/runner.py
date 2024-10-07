#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/






import os
import configparser
import main
import config_classes as CC
import LLM_functuions as LF
import classifier as C
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from classifier import BERTClassifier  # Assuming BERTClassifier is defined in classifier.py
import matplotlib.pyplot as plt
import seaborn as sns





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
        'zhipuai_api_key': f"{config[CC.Apikeys.APIKEYS][CC.Apikeys.ZHIPUAIKEY]}"
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
        LF.zhipuai_prompt_func
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
    num_samples = 2
    # Consider gpt-3.5-turbo model from O penAI API as considered text generator function
    considered_txt_generator = LF.gpt_prompt_func
    # Considered model for the considered text generator
    considered_model = llm_model_names['gpt']
    considered_api_key = api_keys_list['gpt_api_key']
    x_i_list = main.generate_input_texts(num_samples, considered_txt_generator, considered_model, considered_api_key)

    # Step 2: Generate completions using 20 LLMs
    # Number of completions to generate for eac of the x_i and LLM pairs
    num_completions = 2
    outcomes = main.generate_llm_completions(x_i_list, llm_functions, llm_model_names, api_keys_list, num_completions)

    # Step 3: Save the results to a file
    main.save_outcomes(outcomes, output_folder_path)
    ##################################################

    #################################################
    """Feature Extraction"""
    # Semantic Similarity
    # Save the similarity results to a csv file
    similarity_results_path = os.path.join(output_folder_path, 'similarity_results.csv')
    with open(similarity_results_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                similarity = main.cosine_sim(x_i, x_j)
                f.write(f"Cosine Similarity between {x_i} and {x_j}: {similarity}\n")
    
    # Word Embedding
    # Get embeddings for xi and xj
    # Save the embeddings to a csv file
    embeddings_results_path = os.path.join(output_folder_path, 'embeddings_results.csv')
    with open(embeddings_results_path, 'w') as f:
        for x_i in x_i_list:
            xi_embedding = main.get_embeddings(x_i)
            for x_j in outcomes[x_i]:
                xj_embedding = main.get_embeddings(x_j)
                similarity = torch.nn.functional.cosine_similarity(xi_embedding, xj_embedding)
                f.write(f"Cosine Similarity between {x_i} and {x_j}: {similarity.item()}\n")
    
    # Style Features
    # Save the sentence length to a csv file
    Length_results_path = os.path.join(output_folder_path, 'Length_results.csv')
    with open(Length_results_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                f.write(f"Sentence Length of {x_i}: {main.sentence_leng(x_i)}\n")
                f.write(f"Sentence Length of {x_j}: {main.sentence_leng(x_j)}\n")

    # Save the average word length to a csv file
    Average_result_path = os.path.join(output_folder_path, 'average_word_length_results.csv')
    with open(Average_result_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                f.write(f"Average Word Length of {x_i}: {main.avg_word_length(x_i)}\n")
                f.write(f"Average Word Length of {x_j}: {main.avg_word_length(x_j)}\n")

    # Save the pose tags to a csv file
    Tags_result_path = os.path.join(output_folder_path, 'tags_results.csv')
    with open(Tags_result_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                f.write(f"POS Tags of {x_i}: {main.pos_tags(x_i)}\n")
                f.write(f"POS Tags of {x_j}: {main.pos_tags(x_j)}\n")

    # Save the count the frequency of n-grams across different LLM-generated completions to a csv file
    Ngram_result_path = os.path.join(output_folder_path, 'ngram_results.csv')
    with open(Ngram_result_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                f.write(f"Frequency of n-grams in {x_i}: {main.ngram_freq(x_i)}\n")
                f.write(f"Frequency of n-grams in {x_j}: {main.ngram_freq(x_j)}\n")  
    #################################################

    ##################################################
    # Use Bert deep learning classifier to figure out, for each input (xi, xj), which LLM was used for this pair.
    # Initialize model
    num_classes = len(llm_model_names)  # Define num_classes based on the number of LLM models
    model = C.BERTClassifier(num_classes)  # Use the imported classifier

    #Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Define Loss 
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    num_epochs = 10
    train_loader = main.get_train_loader(x_i_list, outcomes, llm_model_names)
    C.train_model(model, train_loader, optimizer, criterion, num_epochs)
    ##################################################

    ##################################################
    # Model Evaluation
    # Evaluate the model
    test_loader = main.get_test_loader(x_i_list, outcomes, llm_model_names)
    accuracy = C.evaluate_model(model, test_loader)
    print(f"Model accuracy: {accuracy}")
    precision = C.evaluate_model(model, test_loader, metric='precision')
    print(f"Model precision: {precision}")
    recall = C.evaluate_model(model, test_loader, metric='recall')
    print(f"Model recall: {recall}")
    f1_score = C.evaluate_model(model, test_loader, metric='f1')
    print(f"Model F1 score: {f1_score}")
    conf_matrix = C.confusion_metrix()
    print(f"Confusion Matrix: {conf_matrix}")

    # Plotting confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted LLM')
    plt.ylabel('True LLM')
    plt.show()

    # Analyze LLM Behavior
    # Save the model predictions to a csv file
    model_predictions_path = os.path.join(output_folder_path, 'model_predictions.csv')
    with open(model_predictions_path, 'w') as f:
        for x_i in x_i_list:
            for x_j in outcomes[x_i]:
                prediction = C.get_model_prediction(model, x_i, x_j)
                f.write(f"Model prediction for ({x_i}, {x_j}): {prediction}\n")
    ##################################################

    ##################################################
    # Handling warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    ##################################################



"""
    Main function to run the microlearning generator.
"""
if __name__ == "__main__":
#     config = None
#     if len(sys.argv) != 2:
#         print(f"Usage: python3 {sys.argv[0]} configfile")
#     elif not os.path.exists(sys.argv[1]):
#         print(f"Can't find configuration file {sys.argv[1]}")
#     else:
#         #Read the configuration file
#         config = main.read_configuration(sys.argv[1]) 
#     if not config:
#         print("Error, conf was 'None'.")
#     else:
#         run(config)
#     run(config)

    config = configparser.ConfigParser()
    config.read('/Volumes/MyDrive/MSC_CSE_PSU_Research/CSE584/config.ini')
    run(config)
