#title: runner.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-November-14 15:55:38
#------------------------------------*/




import torch
import transformers
from openai import OpenAI
import cohere
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from mistralai import Mistral  # Add this import statement
from transformers import AutoTokenizer







# Define response function
def gpt_prompt_func(api_key, considered_model, prompt):
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model = considered_model,
        temperature = 0.5,
        messages = [
            # {"role": "system", "content": "You are a creative assistant that generates intentionally faulty questions."},
            {"role": "system", "content": "You are an intelligent assistant that divides faulty questions into categories based on their characteristics: \
            1. Ambiguity: Questions with unclear language. \
            2. Complexity: Questions requiring multi-step reasoning. \
            3. Domain Knowledge: Questions requiring specialized knowledge. \
            4. Logical Errors: Questions with embedded logical inconsistencies."},
            {"role": "user", "content": prompt},
        ] 
    )
    messages = response.choices[0].message.content.strip().split('\n')
    # Access the content attribute directly
    # print(response.choices[0].message.content)
    return messages[0]

# Define MistralAI (mistral-large-latest) response function
#No streaming
def mistral_prompt_func(api_key, considered_model, prompt):
    mistral_client = Mistral(api_key=api_key)
    response = mistral_client.chat.complete(
        model = considered_model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    messages =response.choices[0].message.content
    return messages


# Define AI21 Labs response function
def ai21_labs_prompt_func(api_key, considered_model, prompt):
    ai21_client = AI21Client(api_key=api_key)
    response = ai21_client.chat.completions.create(
        model= considered_model,  # Latest model
        messages=[ChatMessage(   # Single message with a single prompt
            role="user",
            content = prompt
    )],
        temperature = 0.7,
        max_tokens = 200, # You can also mention a max length in the prompt "limit responses to twenty words"
        top_p = 0.5
    )
    messeages = response.choices[0].message.content
    return messeages


#Define Falcon(falcon-7b-instruct) response function
def falcon_prompt_func(api_key, considered_model, prompt):
    # Tokenize the input and completion texts
    tokenizer = AutoTokenizer.from_pretrained(considered_model) # use "tiiuae/falcon-7b-instruct" for instruction model
    pipeline = transformers.pipeline(
        "text-generation",
        model=considered_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # To automatically distribute the model layers across various cards
    )
    sequences = pipeline(prompt,  # try to be creative here
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        # print(seq["generated_text"])
        return seq["generated_text"]
    

# Define Cohere (command-r-plus-08-2024) response function
def cohere_prompt_func(api_key, considered_model, prompt):
    cohere_client = cohere.ClientV2(api_key = api_key) # Get your free API key: https://dashboard.cohere.com/api-keys
    response = cohere_client.chat(model = considered_model,
                                  messages = [
                                      {"role": "system", 
                                       "content": "You are a creative assistant that generates intentionally faulty questions."},
                                      {
                                            "role": "user",
                                            "content": prompt,
                                      } 
                                    ],
                                  temperature=0.6)  
    messages = response.message.content[0].text
    return messages