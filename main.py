from fastapi import FastAPI
#Initialize FastAPI
import uvicorn
from pydantic import BaseModel
from ast import literal_eval

import google.generativeai as genai
import textwrap
import pandas as pd
import numpy as np
app = FastAPI()


GOOGLE_API_KEY = 'AIzaSyCyEirvh5YNKK1dekGvfw8AZf7PZN1uoko'

genai.configure(api_key=GOOGLE_API_KEY)
# df= pd.read_json('output.json')
df_embeddings = pd.read_csv('embeddings.csv')
df_embeddings.loc[:,'Embeddings'] = df_embeddings.loc[:,'Embeddings'].apply(lambda x: literal_eval(x))

model = 'models/embedding-001' 

model_ai = genai.GenerativeModel('gemini-pro')






class queryType(BaseModel):
    query:str

@app.post("/bookmechatbot")
def chatbot(request: queryType ):
    print(request)
    answer =   mainFlow(request.query)
    print("answer",answer)
    return {"message" : answer}



def mainFlow(query):

    #print(df_embeddings)
    print('SUCCESS')

    
    # df['Embeddings'] = df.apply(lambda row: embed_fn(row['input_text'], row['output_text']), axis=1)
    
    result =  find_best_passage(query, df_embeddings,model)

    prompt = make_prompt(query, result)

  
    query_output = model_ai.generate_content(prompt) 
    answer = query_output.text


    return answer





################ FUNCTIONS  ######################
    
def find_best_passage(query, df_embeddings,model):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(df_embeddings['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return df_embeddings.iloc[idx]['output_text'] # Return text from index with max value



# Get the embeddings of each text and add to an embeddings column in the dataframe
model = 'models/embedding-001'
def embed_fn(title, text):
  return genai.embed_content(model=model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]



def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""Your name is Bookme Assitant.You are a helpful and informative bot that answers the questions
  using text fro refernce information provided below.\ Be sure to respond in a complete sentence,
  being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \ If the result is not found or irrelevant to the answer, you will
  respond with  "Sorry, I am unable to answer. Please email your problem to our customer support.".\ If the question has
  hate speech or contains questions unrelated to bookme, you may ignore it. \ You will not answer any question about any other
  ticketing or courier service other than  Bookme.


  QUESTION: '{query}'
  Result: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
