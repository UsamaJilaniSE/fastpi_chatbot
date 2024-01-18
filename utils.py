import google.generativeai as genai
import textwrap
import pandas as pd
import numpy as np
#import google.generativeai as genai
#import google.ai.generativelanguage as glm

# Used to securely store your API key
#dotenv package to laod api key


GOOGLE_API_KEY = 'AIzaSyCyEirvh5YNKK1dekGvfw8AZf7PZN1uoko'

genai.configure(api_key=GOOGLE_API_KEY)

# dataset = 'output.json'
# df = pd.read_json(dataset)
# df.columns = ['input_text', 'output_text']


# model = 'models/embedding-001'
# def embed_fn(title, text):
#   print("function applied")
#   return genai.embed_content(model=model,
#                              content=text,
#                              task_type="retrieval_document",
#                              title=title)["embedding"]

# df['Embeddings'] = df.apply(lambda row: embed_fn(row['input_text'], row['output_text']), axis=1)
# df = pd.read_csv("embeddings.csv")

# print(df)

def mainFlow(query):

  df = pd.DataFrame("embedding.csv")
  model = 'models/embedding-001'
  request = genai.embed_content(model=model,
                              content=query,
                              task_type="retrieval_query")
  result = find_best_passage(query, df)
  prompt = make_prompt(query, result)
  model = genai.GenerativeModel('gemini-pro')
  answer = model.generate_content(prompt)
  return answer


def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['output_text'] # Return text from index with max value

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





####################  temp place  ##################

def find_best_passage(query, dataframe,model):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding =  genai.embed_content(model=model,content=query,task_type="retrieval_query")
  print(query_embedding)
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"]) #this is not
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['output_text'] # Return text from index with max value

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