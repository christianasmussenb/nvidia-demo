#!/usr/bin/env python
import sys
import warnings

from new_project.crew import NewProject

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import textwrap
from rich.markdown import Markdown
from rich.console import Console

# Must precede any llm module imports
import os
from langtrace_python_sdk import langtrace
langtrace.init(api_key = os.getenv('LANGTRACE_API_KEY'))

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs'
    }
    result = NewProject().crew().kickoff(inputs=inputs)

    # Verificar la salida del blog
    #print("Resultado del crew:", result.pydantic.model_dump())  # Agregar esta línea para depuración

    # Display blog post
    title = result.pydantic.model_dump().get('title', 'Sin título')
    content = result.pydantic.model_dump().get('content', 'Sin contenido')
    
    print(f"Título: {title}")
    wrapped_content = textwrap.fill(content, width=50)
    print(wrapped_content)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        NewProject().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        NewProject().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        NewProject().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
