import os
from openai import AzureOpenAI
import time


#client = AzureOpenAI(
#		    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai", 
#			api_key=os.getenv("AZURE_OPENAI_KEY"),
#		    api_version=os.getenv("AZURE_API_VERSION")
#		)
 
client = AzureOpenAI(
           azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
           api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
           api_version="2024-02-01"
        )

def generate_prompt(sample):
    return [
        {
            "role": "user",
            "content": sample
            ,
        }
    ]


def make_answer_gpt(message, model_type, sleep):
    requests = 1
    while requests <= 10:
        try:
            #print("model_type", model_typ)
            print("message", message[0]['content'])
            response = client.chat.completions.create(
                model=model_type, messages=message, max_tokens=600, temperature = 0
            )
            
            if requests > 1:
                print("Request Retry Success...")
            
            return response.choices[0].message.content
        
        except Exception as e: 
            print(e)
            print(f"Retrying request of trial # {requests}...")
            requests += 1
            time.sleep(sleep) 

    print("Request Exceeded Trial Limit...")
    
    return ""
