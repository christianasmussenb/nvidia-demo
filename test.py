"""
from code_documentation_generator import LoadingAnimation
from crewai import LLM

loader = LoadingAnimation()
loader.start("Calling NVIDIA NIM")

llm = LLM(model="nvidia_nim/meta/llama-3.3-70b-instruct")
response = llm.call(
    messages=[
        {
            "role": "user",
            "content": "What's a good name for a dog?",
        }
    ]
)
loader.stop("Complete")
"""

from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = ""
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"content":"{\"fullName\":\"Albus Percival Wulfric Brian Dumbledore\",\"nickname\":\"Dumbledore\",\"hogwartsHouse\":\"Gryffindor\",\"interpretedBy\":\"Richard Harris\",\"children\":[],\"birthdate\":\"Aug 29, 1881\",\"index\":14}","role":"tool","tool_call_id":"chatcmpl-tool-96f08eed57f849e4acf1952180c1ce39","name":"describe_harry_potter_character"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


print('\n')
#print(response)

