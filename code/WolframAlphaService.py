import requests


def wolfram_alpha_query(query, app_id):
    # Wolfram Alpha API endpoint
    # url = "http://api.wolframalpha.com/v1/result"
    url = "http://api.wolframalpha.com/v1/spoken"

    # Parameters for the query
    params = {
        "appid": "QG759U-K96T398GRW",
        "i": query
    }

    # Make the API call
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print("Error making API call:", response.text)
        return None


# Example query
query = "how do i cook pasta?"

# Replace 'YOUR_APP_ID' with your actual Wolfram Alpha app ID
app_id = "QG759U-K96T398GRW"

# Make the API call and get the response content
response = wolfram_alpha_query(query, app_id)


print(response)
# # Save or display the response content as needed
# if response_content:
#     with open("wolfram_alpha_response.png", "wb") as f:
#         f.write(response_content)
#     print("Response saved as 'wolfram_alpha_response.png'.")
# else:
#     print("No response received.")
