from langchain import PromptTemplate
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import nltk
from langchain.chains import SequentialChain
from langchain.agents import load_tools, initialize_agent

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
llm = OpenAI(temperature=0)
tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
query = "What is langchain?"
# Sentiment Analysis
nltk.download("vader_lexicon")

# Import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the class
sid = SentimentIntensityAnalyzer()

# Get sentiment scores for a text
scores = sid.polarity_scores(query)

if scores["compound"] >= 0.05:
    factual = 1
else:
    factual = 0
if factual == 0:
    template = "{text}"
    prompt_template = PromptTemplate(input_variables=["text"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = answer_chain.run(query)
    print(answer)
else:
    print(agent.run(query))
