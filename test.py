import requests

# Define the API URL
api_url = "http://127.0.0.1:5000/query"  # Replace with your API URL

# Define the JSON data
data = {"query": "What is langchain?"}

# Send the POST request
response = requests.post(api_url, json=data)
print(response)
# Print the response content and status code
print("Response Content:", response.text)
print("Status Code:", response.status_code)

