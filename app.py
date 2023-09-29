import os
import nltk
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.agents import load_tools, initialize_agent
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set environment variables
# os.environ["OPENAI_API_KEY"] = ''
# os.environ["SERPAPI_API_KEY"] = ''

# Initialize OpenAI
llm = OpenAI(temperature=0)

# Initialize tools and agent
tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Initialize NLTK for sentiment analysis
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        query = data.get('query', '')

        # Get sentiment scores for the query
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
            # response = {"answer": answer}
            # return jsonify(response), 204  # Status code 204 for success
        else:
            answer = agent.run(query)
        response = {"answer": answer}
        print("------------------")
        return jsonify(response), 204  # Status code 204 for success
    except Exception as e:
        error_message = str(e)
        response = {"error": error_message}
        return jsonify(response), 500  # Status code 500 for server-side error

if __name__ == '__main__':
    app.run(debug=True)

