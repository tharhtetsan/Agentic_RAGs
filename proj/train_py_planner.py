from crewai import Agent,Crew
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

# load SERPER_API_KEY
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
llm = Ollama(
    model="llama3.2:latest",
    temperature=0,
    # other params...
)


location = "Bangkok"
cities = "Berlin or Japan"
date_range = "New year holidays"
interests = "AI , Latest Technology and nature "


from trip_tasks import TripTasks
tasks = TripTasks()


city_selector_agent = Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True,
        llm =llm)


local_expert_agent = Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True,
        llm =llm)

travel_concierge_agent = Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        verbose=True,
        llm =llm)


identify_task = tasks.identify_task(
      city_selector_agent,
      location,
      cities,
      interests,
      date_range)

gather_task = tasks.gather_task(
  local_expert_agent,
  location,
  interests,
  date_range
)

plan_task = tasks.plan_task(
   travel_concierge_agent, 
   location,
   interests,
   date_range
   
   )


crew = Crew(
    agents=[
      city_selector_agent, local_expert_agent, travel_concierge_agent
    ],
    tasks=[identify_task, gather_task, plan_task],
    verbose=True
  )



result = crew.kickoff()

print("###########################")
print(result)