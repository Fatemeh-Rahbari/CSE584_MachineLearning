#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/





import os
import json
import logging
import requests
import random
import torch
# 1. Google packages
# Gemini
import google.generativeai as genai
# PaLM
import pprint
import google.generativeai as palm
# BERT
from transformers import BertTokenizer, BertForMaskedLM
# 2. Openai packages
from openai import OpenAI
# 3. Meta packages
from langchain_community.chat_models import ChatOllama
# 4. Microsoft packages
# 5. Amazon packages
import boto3
from botocore.exceptions import ClientError
# 6. Apple packages
from transformers import AutoTokenizer, AutoModelForCausalLM
# 7. Bloom packages
import transformers
from transformers import set_seed
# 8. Falcon packages
# 9. Cohere packages
import cohere
# 10. Mistral AI packages
from mistralai import Mistral
# 11. AI21 Python SDK
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
# 12. CMU&Princeton (Mamba) packages
from transformers import AutoTokenizer
# 13. Alibaba packages
import dashscope
from http import HTTPStatus
# 14. Shanghai AI Laboratory(InternLM2) packages
# import lmdeploy
# 15. DataBricks packages
import transformers
from transformers import AutoTokenizer
from transformers import pipeline
# 16. BigCode (StarCoder) packages
# 17. Stability AI (StableLM) packages
# 18. Zhipu AI packages
from zhipuai import ZhipuAI
# 19. WhyLab packages
import whylabs_client
from pprint import pprint
from whylabs_client.api import api_key_api











""" 
Generates pair for given pieces of the text (input) using diffeent large language language (LLMs).
1. Define the response function for each LLM model.
2. Define the generator function that generates pairs for each model.
3. Save the pair to a file.
"""

# 1. Define Google (gemini-1.5-flash) response function
# Running with the pipeline API
def gemini_prompt_func(prompt, considered_model,api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(considered_model)
    response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        max_output_tokens=200,
        temperature=0.8,
        ),
    )
    print(response.text)
    return response.text    





# Define Google (PaLM) response function
def palm_prompt_func(prompt, considered_model,api_key):
    palm.configure(api_key=api_key)
    completion = palm.generate_text(
        model=considered_model,
        prompt=prompt,
        temperature=0.8,
        # The maximum length of the response
        max_output_tokens=800,
    )
    print(completion.result)
    return completion.result

# define Google (BERT) response function
def bert_prompt_func(prompt, considered_model):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(considered_model)
    # Encode text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Find the position of the [MASK] token
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    # Load pre-trained model
    model = BertForMaskedLM.from_pretrained(considered_model)
    model.eval()
    # Predict all tokens
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]
    # Extract the predicted word
    predicted_index = torch.argmax(predictions[0, mask_token_index]).item()
    predicted_words = tokenizer.decode([predicted_index])
    print(predicted_words)
    return predicted_words





# 2. Define OpenAI (gpt-3.5-turbo) response function
def gpt_prompt_func(prompt, considered_model, api_key):
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the correct model ID
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are an expert in generating random text and completing any truncated text."},
            {"role": "user", "content": prompt}
        ],
    )
    # Access the content attribute directly
    print(response.choices[0].message.content)
    return response.choices[0].message.content





# 3. Define Meta (llama3) response function
def meta_prompt_func(prompt, considered_model, api_key):
    api_key =api_key
    url = "http://localhost:11434/api/chat"
    data = {
        "model": considered_model,
        "messages": [
            {
                "role": "user",
                "content": prompt

            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    print (response.json()["message"]["content"])
    return response.json()["message"]["content"]
# #Use local Llama 3.1 via LangChain   
# def chat_ollama(prompt):
#     llm = ChatOllama(model="llama3.1", temperature=0.8)
#     response = llm.invoke(prompt)
#     print(response.content)





# 4. Define Microsoft (Phi-3) response function
def micro_prompt_func(prompt, considered_model, api_key):
    api_key = api_key
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(considered_model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    return text





# 5. Define Amazon (Titan) response function
def amazon_prompt_func(prompt, considered_model):
    # Configure logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
       
    logger.info("Generating text with Amazon Titan Text Premier model %s", considered_model)
    try:
        # Initialize Bedrock client
        bedrock = boto3.client(service_name="bedrock-runtime")
        accept = "application/json"
        content_type = "application/json"
        
        # Prepare request body
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "stopSequences": [],
                "temperature": 0.8,
                "topP": 0.9
            }
        }
        body = json.dumps(request_body)
        
        # Invoke model
        bedrock_client = boto3.client(service_name="bedrock")
        response = bedrock.invoke_model(
            body=body, modelId=considered_model, accept=accept, contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
        return response_body
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        return None





# 6. Define Apple (OpenELM) response function
def apple_prompt_func(prompt, considered_model, api_key):
    # Set device on which you"ll run the model 
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    hf_access_token = 500
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_access_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(considered_model, token=hf_access_token, trust_remote_code=True)
    model.to(device).eval()

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Decode and print output
    outputs = model.generate(inputs.input_ids, max_length=512)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text





# 7. Define Bloom ("bigscience/bloom-1b7")response function
def bloom_prompt_func(prompt, considered_model, api_key):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = AutoModelForCausalLM.from_pretrained(considered_model, use_cache=True) 
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    set_seed(2024)
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, 
                            max_length=200, top_k=1, 
                            temperature=0.3, repetition_penalty=2.0)
    generated_text = tokenizer.decode(sample[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text





# 8. Define Falcon(falcon-7b-instruct) response function
def falcon_prompt_func(prompt, considered_model):
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
        print(seq["generated_text"])
        return seq["generated_text"]





# 9. Define Cohere (command-r-plus-08-2024) response function
def cohere_prompt_func(prompt, considered_model, api_key):
    cohere_client = cohere.ClientV2(api_key=api_key) # Get your free API key: https://dashboard.cohere.com/api-keys
    # Generate the response
    response = cohere_client.chat(model=considered_model, messages=[{"role": "user", "content": prompt}], temperature=0.8)  # Or messages=[cohere.UserMessage(content=message)])
    print(response.message.content[0].text)
    return response.message.content[0].text

    # Streaming responses
    # Add the user message
    # message = "I"m joining a new startup called Co1t today. Could you help me write a one-sentence introduction message to my teammates."
    # # Generate the response by streaming it
    # response = cohere_client.chat_stream(model="command-r-plus-08-2024",
    #                           messages=[{"role": "user", "content": message}])
    # for event in response:
    #     if event:
    #         if event.type == "content-delta":
    #             print(event.delta.message.content.text, end="")





# 10. Define MistralAI (mistral-large-latest) response function
#No streaming
def mistral_prompt_func(prompt, considered_model, api_key):
    mistral_client = Mistral(api_key=api_key)
    chat_prompt = mistral_client.chat.complete(
        model = considered_model,
        temperatur = 0.8,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_prompt.choices[0].message.content

# With async
# def mistral_async_prompt_func(prompt, considered_model, api_key):
    # api_key = os.environ["mistral_API_KEY"]
    # model = considered_model
    # prompt = prompt
    # mistral_client = Mistral(api_key=api_key)
    # async_prompt = await mistral_client.chat.stream_async(
    #     model = model,
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         },
    #     ]
    # )
    # async for chunk in async_prompt: 
    #     return chunk.data.choices[0].delta.content





# 11. Define AI21 Labs response function
def ai21_labs_prompt_func(prompt, considered_model, api_key):
    ai21_client = AI21Client(api_key=api_key)
    response = ai21_client.chat.completions.create(
        model= considered_model,  # Latest model
        messages=[ChatMessage(   # Single message with a single prompt
            role="user",
            content = prompt
    )],
        temperature=0.8,
        max_tokens=200 # You can also mention a max length in the prompt "limit responses to twenty words"
    )
    return response.choices[0].message.content





# 12. Define CMU&Princeton (state-spaces/mamba-2.8b)response function
def cmu_princeton_prompt_func(prompt, considered_model):
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    model = AutoModelForCausalLM.from_pretrained(considered_model).to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1] + 80, temperature=0.7, top_k=50, top_p=0.95)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)





# 13. Define Alibaba ("qwen1.5-110b-chat") response function
def alibaba_prompt_func(prompt, considered_model,api_key):
    api_key = api_key
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    messages = [
        {"role": "user", "content": prompt}]
    responses = dashscope.Generation.call(
        considered_model,
        messages=messages, 
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format="message",  # set the result to be "message"  format.
        stream=True,
        output_in_full=True  # get streaming output incrementally
    )
    full_content = ""
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            full_content += response.output.choices[0]["message"]["content"]
            print(response)
        else:
            print("Request id: %s, Status code: %s, error code: %s, error message: %s" % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    print("Full content: \n" + full_content)
    return full_content





# 14. Define Shanghai AI Laboratory(internlm/internlm2-chat-7b) response function
def shanghai_ai_prompt_func(prompt, considered_model,api_key):
    api_key= api_key
    tokenizer = AutoTokenizer.from_pretrained(considered_model, trust_remote_code=True)
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(considered_model, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model = model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)
    return output
    # api_key =  api_key
    # pipe = lmdeploy.pipeline(considered_model)
    # response = pipe(prompt)
    # print(response)
    # return response




# 15. Define DataBricks (mosaicml/mpt-7b) response function
def databricks_prompt_func(prompt, considered_model):
    model = transformers.AutoModelForCausalLM.from_pretrained(
    considered_model,
    trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        print(
            pipe(prompt,
                max_new_tokens=100,
                do_sample=True,
                use_cache=True))





# 16. Define BigCode (bigcode/starcoder) response function
def bigcode_prompt_func(prompt, considered_model, api_key):
    api_key = api_key
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    model = AutoModelForCausalLM.from_pretrained(considered_model).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

# def bigcode_prompt_func(prompt, considered_model, api_key):
#     API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
#     headers = {"Authorization": f"{api_key}"}
#     response = requests.post(API_URL, headers=headers, json=prompt)
#     print (response.json())
#     return response.json()





# 17. Define Stability AI (stabilityai/stablelm-2-12b-chat) response function
def stablity_prompt_func(prompt, considered_model, api_key):
    api_key = api_key
    tokenizer = AutoTokenizer.from_pretrained(considered_model)
    model = AutoModelForCausalLM.from_pretrained(
        considered_model,
        device_map="auto",
    )
    prompt = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )
    output = tokenizer.decode(tokens[:, inputs.shape[-1]:][0], skip_special_tokens=False)
    print(output)
    return output





# 18. Define Zhipu AI (glm-4) response function
def zhipuai_prompt_func(prompt, considered_model, api_key):
    zhipu_client = ZhipuAI(api_key=api_key)  # Please fill in your own APIKey
    response = zhipu_client.chat.completions.create(
        model=considered_model,  # Please fill in the model name you want to call
        messages=[
            {"role": "user", "content": prompt}],
    )
    result_text = response.choices[0].message.content
    print(result_text)
    return result_text





# 19. Define WhyLabs (Turing-1.3B) response function
def whylabs_prompt_func(prompt, considered_model, api_key):
    # Defining the host is optional and defaults to https://api.whylabsapp.com
    # See configuration.py for a list of all supported configuration parameters.
    configuration = whylabs_client.Configuration(
        host = "https://api.whylabsapp.com"
    )
    configuration.api_key["ApiKeyAuth"] = api_key

    # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
    # configuration.api_key_prefix["ApiKeyAuth"] = "Bearer"
    # Enter a context with an instance of the API client
    with whylabs_client.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = api_key_api.ApiKeyApi(api_client)
        org_id = "org-123" # str | Your company"s unique organization ID
    user_id = "user-123" # str | The unique user ID in an organization.
    expiration_time = 1577836800000 # int, none_type | Expiration time in epoch milliseconds (optional)
    scopes = [
            ":user",
        ] # [str], none_type | Scopes of the token (optional)
    alias = "MLApplicationName" # str, none_type | A human-friendly name for the API Key (optional)

    try:
            # Generate an API key for a user.
            api_response = api_instance.create_api_key(org_id, user_id, expiration_time=expiration_time, scopes=scopes, alias=alias)
            print(api_response)
            return(api_response)
    except whylabs_client.ApiException as e:
            print("Exception when calling ApiKeyApi->create_api_key: %s\n" % e)