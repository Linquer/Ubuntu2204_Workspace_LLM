"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

# 代理设置：
# export https_proxy=http://172.18.176.1:7890

import os
import time

import google.generativeai as genai

api_key = "AIzaSyCeS9ynYWGY9xOOpHrgaVjh8kqCTCyYrhs"

genai.configure(api_key=api_key)
print(1)

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)
print(2)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "You are Tom, my assistant of academic",
      ],
    }
  ]
)
start_time = time.time()
response = chat_session.send_message("Find some books about RL")
end_time = time.time()
print(end_time - start_time)
print(3)
print(response.text)