class InputFiles:
    LOCATION = "LOCATION"
    INPUTFOLDER = "input_folder"
    FILE= "FILE"
    INPUTCSVNAME = "input_csv_name"

class OutputFiles:
    OUTPUTFOLDER = "output_folder" 
    OUTPUTQUESTIONSCSVNAME= "output_questions_csv_name"
    OUTPUTANSWERSCSVNAME= "output_answers_csv_name"
    EXPERIMENTCSVNAME= "experiment_csv_name"
    EXPERIMENTRESULTSCSVNAME= "experiment_results_csv_name"


class GoogleSheets:
    GOOGLESHEET = "GOOGLESHEET"
    CREDENTIALS = "Google_credential_json_name"
    INPUTSHEETNAME = "your_Google_sheet_name"
    RESULTSHEETNAME = "output_Google_sheet_name"
    GOOGLESHEETID = "input_Google_sheet_id"
    WORKSHEETNAME = "worksheet_name"
    
   
class LLMModels:
    MODEL = "MODEL"
    # 1, OpenAI
    OPENAI= "openai_considered_model"
    # 2. Mistra
    MISTRAL= "mistral_considered_model"
    # 3. AI21
    AI21= "aI21_considered_model"
    # 4. Cohere
    COHERE= "cohere_considered_model" 

class Apikeys:
    APIKEY= "APIKEY"
    OPENAIKEY= "OpenAI"
    MISTRALKEY= "MistralAI"
    AI21KEY= "AI21Studio"
    COHEREKEY= "Cohere"
    