#!/usr/bin/env python3

import getpass
import os
import sys
import time
import threading
import yaml
import subprocess
import re
from pathlib import Path
from pydantic import BaseModel
from typing import List
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start
from crewai_tools import DirectoryReadTool, FileReadTool, WebsiteSearchTool
import dotenv
from dotenv import dotenv_values
from langtrace_python_sdk import langtrace
import litellm
import warnings

# litellm.set_verbose = True

# Variables globales para los crews
planning_crew = None
documentation_crew = None

api_key = os.getenv('LANGTRACE_API_KEY')
langtrace.init(api_key=api_key)

# Clase para animación de carga
class LoadingAnimation:
    def __init__(self):
        self.stop_event = threading.Event()
        self.animation_thread = None

    def _animate(self, message="Loading"):
        chars = "/—\\|"
        while not self.stop_event.is_set():
            for char in chars:
                sys.stdout.write('\r' + message + '... ' + char)
                sys.stdout.flush()
                time.sleep(0.1)
                if self.stop_event.is_set():
                    sys.stdout.write("\n")
                    break

    def start(self, message="Loading"):
        self.stop_event.clear()
        self.animation_thread = threading.Thread(target=self._animate, args=(message,))
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop(self, completion_message="Complete"):
        self.stop_event.set()
        if self.animation_thread:
            self.animation_thread.join()
        print(f"\r{completion_message} ✓")

# Modelos de datos
class DocItem(BaseModel):
    title: str
    description: str
    prerequisites: str
    examples: list[str]
    goal: str

class DocPlan(BaseModel):
    overview: str
    docs: list[DocItem]

class DocumentationState(BaseModel):
    project_url: str = "https://github.com/crewAIInc/nvidia-demo"  # Valor por defecto
    repo_path: Path = Path("workdir/")
    docs: List[str] = []

# Función para verificar sintaxis mermaid
def check_mermaid_syntax(task_output):
    text = task_output.raw
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', text, re.DOTALL)
    
    for block in mermaid_blocks:
        diagram_text = block.strip()
        lines = diagram_text.split('\n')
        corrected_lines = []
        
        for line in lines:
            corrected_line = re.sub(r'\|.*?\|>', lambda match: match.group(0).replace('|>', '|'), line)
            corrected_lines.append(corrected_line)
            
        text = text.replace(block, "\n".join(corrected_lines))
    
    task_output.raw = text
    return (True, task_output)

# Clase principal del flujo de documentación
class CreateDocumentationFlow(Flow[DocumentationState]):
    @start()
    def clone_repo(self):
        print(f"# Clonando repositorio: {self.state.project_url}\n")
        repo_name = self.state.project_url.split("/")[-1]
        self.state.repo_path = Path(f"{self.state.repo_path}/{repo_name}")

        if self.state.repo_path.exists():
            print(f"# El directorio del repositorio ya existe en {self.state.repo_path}\n")
            subprocess.run(["rm", "-rf", str(self.state.repo_path)])
            print("# Directorio existente eliminado\n")

        subprocess.run(["git", "clone", self.state.project_url, str(self.state.repo_path)])
        return self.state

    @listen(clone_repo)
    def plan_docs(self):
        print(f"# Planificando documentación para: {self.state.repo_path}\n")
        result = planning_crew.kickoff(inputs={'repo_path': str(self.state.repo_path)})
        print(f"# Documentación planificada para {self.state.repo_path}:")
        for doc in result.pydantic.docs:
            print(f"    - {doc.title}")
        return result

    @listen(plan_docs)
    def save_plan(self, plan):
        Path("docs").mkdir(exist_ok=True)
        with open("docs/plan.json", "w") as f:
            f.write(plan.raw)

    @listen(plan_docs)
    def create_docs(self, plan):
        for doc in plan.pydantic.docs:
            print(f"\n# Creando documentación para: {doc.title}")
            result = documentation_crew.kickoff(inputs={
                'repo_path': str(self.state.repo_path),
                'title': doc.title,
                'overview': plan.pydantic.overview,
                'description': doc.description,
                'prerequisites': doc.prerequisites,
                'examples': '\n'.join(doc.examples),
                'goal': doc.goal
            })

            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            title = doc.title.lower().replace(" ", "_") + ".mdx"
            self.state.docs.append(str(docs_dir / title))
            with open(docs_dir / title, "w") as f:
                f.write(result.raw)
        print(f"\n# Documentación creada para: {self.state.repo_path}")

def main():
    # Verificar API key
    if not os.environ.get("NVIDIA_NIM_API_KEY", "").startswith("nvapi-"):
        nvapi_key = getpass.getpass("Ingrese su NVIDIA API key: ")
        assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... no es una clave válida"
        os.environ["NVIDIA_NIM_API_KEY"] = nvapi_key
        os.environ["NVIDIA_API_KEY"] = nvapi_key
        os.environ["OPENAI_API_KEY"] = nvapi_key

    # Cargar configuraciones
    with open('config/planner_agents.yaml', 'r') as f:
        planner_agents_config = yaml.safe_load(f)

    with open('config/documentation_agents.yaml', 'r') as f:
        documentation_agents_config = yaml.safe_load(f)

    with open('config/planner_tasks.yaml', 'r') as f:
        planner_tasks_config = yaml.safe_load(f)

    with open('config/documentation_tasks.yaml', 'r') as f:
        documentation_tasks_config = yaml.safe_load(f)

    # Configurar agentes y tareas
    global planning_crew, documentation_crew

    # Templates para Llama 3.3
    system_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>"""
    prompt_template = """<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>"""
    response_template = """<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>"""

    # Configurar el LLM base para todos los agentes
    llm = LLM(
        model="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )

    # Crear los agentes para planning_crew
    code_explorer = Agent(
        config=planner_agents_config['code_explorer'],
        system_template=system_template,
        prompt_template=prompt_template,
        response_template=response_template,
        tools=[DirectoryReadTool(), FileReadTool()],
        llm=llm  # Usar la instancia de LLM configurada
    )
    
    documentation_planner = Agent(
        config=planner_agents_config['documentation_planner'],
        system_template=system_template,
        prompt_template=prompt_template,
        response_template=response_template,
        tools=[DirectoryReadTool(), FileReadTool()],
        llm=llm  # Usar la misma instancia de LLM
    )

    # Crear las tareas para planning_crew
    analyze_codebase = Task(
        config=planner_tasks_config['analyze_codebase'],
        agent=code_explorer
    )
    
    create_documentation_plan = Task(
        config=planner_tasks_config['create_documentation_plan'],
        agent=documentation_planner,
        output_pydantic=DocPlan
    )

    # Inicializar planning_crew
    planning_crew = Crew(
        agents=[code_explorer, documentation_planner],
        tasks=[analyze_codebase, create_documentation_plan],
        verbose=False
    )

    # Crear los agentes para documentation_crew
    overview_writer = Agent(
        config=documentation_agents_config['overview_writer'],
        llm=llm,  # Añadir el LLM aquí también
        tools=[
            DirectoryReadTool(),
            FileReadTool(),
            WebsiteSearchTool(
                website="https://mermaid.js.org/intro/",
                config=dict(
                    embedder=dict(
                        provider="openai",
                        config=dict(
                            model="text-embedding-3-small",
                            api_key=os.environ.get("OPENAI_API_KEY")
                        )
                    )
                )
            )
        ]
    )

    documentation_reviewer = Agent(
        config=documentation_agents_config['documentation_reviewer'],
        llm=llm,  # Añadir el LLM aquí también
        tools=[
            DirectoryReadTool(directory="docs/", name="Check existing documentation folder"),
            FileReadTool()
        ]
    )

    # Crear las tareas para documentation_crew
    draft_documentation = Task(
        config=documentation_tasks_config['draft_documentation'],
        agent=overview_writer
    )

    qa_review_documentation = Task(
        config=documentation_tasks_config['qa_review_documentation'],
        agent=documentation_reviewer,
        guardrail=check_mermaid_syntax,
        max_retries=5
    )

    # Inicializar documentation_crew
    documentation_crew = Crew(
        agents=[overview_writer, documentation_reviewer],
        tasks=[draft_documentation, qa_review_documentation],
        verbose=False
    )

    # Configurar y ejecutar el flujo
    project_url = "https://github.com/crewAIInc/nvidia-demo"  # URL por defecto
    if len(sys.argv) > 1:
        project_url = sys.argv[1]

    flow = CreateDocumentationFlow()
    if project_url != flow.state.project_url:
        flow.state.project_url = project_url
    
    flow.kickoff()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main() 