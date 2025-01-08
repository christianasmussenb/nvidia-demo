from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent
import yaml
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_agent_with_tools():
    # Configurar el agente con las herramientas
    agent = Agent(
        role="Test Agent",
        goal="Test search functionality",
        backstory="Testing agent for tools",
        tools=[SerperDevTool()],
        verbose=True
    )
    
    try:
        # Crear una tarea simple de búsqueda
        result = agent.execute_task(
            "Search for information about US inflation data 2024"
        )
        print("✅ Agent search test passed")
        print("Result:", str(result)[:100], "...")
        return True
    except Exception as e:
        print(f"❌ Agent search test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_agent_with_tools() 