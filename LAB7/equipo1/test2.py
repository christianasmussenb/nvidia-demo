from crewai import LLM, Agent, Crew, Process, Task, Knowledge

from crewai_tools import SerperDevTool

import os
from dotenv import load_dotenv
load_dotenv()

import time  # Importar el módulo time

start_time = time.time()  # Iniciar el temporizador
print(f"Tiempo para Start time: {start_time} ")  # Imprimir el tiempo
# Configure the LLM to use Cerebras
cerebras_llm = LLM(
    model="cerebras/llama3.3-70b",  # Replace with your chosen Cerebras model name i.e. "cerebras/llama3.1-8b"
    api_key=os.environ.get("CEREBRAS_API_KEY"), # Your Cerebras API key
    temperature=0.1,
)
print(f"Tiempo para crear cerebras_llm: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
# Create a PDF knowledge source
pdf_source = PDFKnowledgeSource(
    file_paths=["uno.pdf", "dos.pdf"]
)
print(f"Tiempo para crear PDFKnowledgeSource: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo
# Create knowledge with PDF source
knowledge = Knowledge(
    collection_name="pdf_knowledge",
    sources=[pdf_source]
)
print(f"Tiempo para crear PDFKnowledge: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo

from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
# Create a CSV knowledge source
csv_source = CSVKnowledgeSource(
    file_paths=["uno.csv", "dos.csv"]
)
print(f"Tiempo para crear CSVKnowledgeSource: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo
# Create knowledge with CSV source
knowledge = Knowledge(
    collection_name="csv_knowledge",
    sources=[csv_source]
)
print(f"Tiempo para crear CSVKnowledge: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gpt-4o-mini", temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About papers",
    goal="You know everything about the papers.",
    backstory="""You are a master at understanding papers and their content.""",
    verbose=True,
    allow_delegation=False,
    llm=cerebras_llm,
    knowledge_sources=[pdf_source, csv_source]
)
task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[
        pdf_source,
        csv_source,
    ],  # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
    planning=True,
    planning_llm=llm,
)

# Ejecutar el proceso
result = crew.kickoff(
    inputs={
        "question": "Cual es el valor para el 8 de julio? Be sure to provide sources."
    }
)
print(f"Tiempo para ejecutar crew.kickoff: {time.time() - start_time:.2f} segundos")  # Imprimir el tiempo

#print(result)
