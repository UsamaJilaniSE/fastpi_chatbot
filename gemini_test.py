# -*- coding: utf-8 -*-
"""gemini_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IWq1UNjwzsiY3Te2FfWo3wQrX_iYZcOD
"""

#!pip install -q   --upgrade google-generativeai langchain-google-genai python-dotenv pandas

import google.generativeai as genai

import textwrap
import pandas as pd
import numpy as np
#import google.generativeai as genai
#import google.ai.generativelanguage as glm


# Or use `os.getenv('API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY='AIzaSyCyEirvh5YNKK1dekGvfw8AZf7PZN1uoko'

genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)

# #df = pd.DataFrame(documents)
# #df.columns = ['Title', 'Text']
# #df

fp = 'output.json'
df = pd.read_json(fp)
# df.columns = ['input_text', 'output_text']
# df

# # Get the embeddings of each text and add to an embeddings column in the dataframe
model = 'models/embedding-001'
def embed_fn(title, text):
  return genai.embed_content(model=model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['input_text'], row['output_text']), axis=1)
# df

# check = pd.DataFrame(df)

# # Save the DataFrame to a CSV file
# check.to_csv('embeddings.csv', index=False)

# df = pd.read_csv("embeddings.csv")
print(df["Embeddings"][0])
query = "What is location of daewoo express lahore terminal?"
model = 'models/embedding-001'

request = genai.embed_content(model=model,
                              content=query,
                              task_type="retrieval_query")

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  print(query_embedding["embedding"])
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['output_text'] # Return text from index with max value

em_path = '/content/embeddings.csv'
result = find_best_passage(query, df)
result

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

prompt = make_prompt(query, result)
print(prompt)

model = genai.GenerativeModel('gemini-pro')
answer = model.generate_content(prompt)
print(answer)
# Markdown(answer.text)
