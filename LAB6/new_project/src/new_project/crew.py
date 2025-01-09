from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

groq_llm = "groq/llama3-8b-8192"
llm = "gpt-4o-mini"

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import litellm
litellm.set_verbose=False

from openai import OpenAIError

from pydantic import BaseModel, Field, ValidationError
from typing import List

class SocialMediaPost(BaseModel):
    platform: str = Field(..., description="The social media platform where the post will be published (e.g., Twitter, LinkedIn).")
    content: str = Field(..., description="The content of the social media post, including any hashtags or mentions.")

class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")
    social_media_posts: List[SocialMediaPost] = Field(..., description="A list of social media posts related to the article.")

class Blog(BaseModel):
    title: str = Field(..., description="The title of the blog post, formatted in markdown.")
    content: str = Field(..., description="The content of the blog post, formatted in markdown.")

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class NewProject():
	"""NewProject crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=False,
			llm=groq_llm
			#tools=[SerperDevTool(n_results=5)]
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=False,
			llm=groq_llm
		)

	@agent
	def blog_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['blog_writer'],
			verbose=True,
			llm=groq_llm
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			context=[self.research_task()],
			output_file='report.md',
			output_pydantic=ContentOutput
		)

	@task
	def blogging_task(self) -> Task:
		try:
			return Task(
				config=self.tasks_config['blogging_task'],
				context=[self.research_task()],
				output_file='report_blog.md',
				output_pydantic=Blog
			)
		except OpenAIError as e:
			print(f"Error al generar el contenido del blog: {e}")
			# Puedes decidir cómo manejar el error, por ejemplo, devolver un valor por defecto o un mensaje de error.
			return None  # O cualquier otro valor que tenga sentido en tu contexto

	@crew
	def crew(self) -> Crew:
		"""Creates the NewProject crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
