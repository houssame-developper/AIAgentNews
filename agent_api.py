# -*- coding: utf-8 -*-
# This script demonstrates using LangChain, OpenRouter, and weather/news tools in Python.
# Purpose: Fetch news, search weather, and interact with LLM agents using LangChain.

# Import necessary libraries for LangChain, OpenAI, RSS parsing, HTML cleaning, embeddings, and HTTP requests
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI # Import ChatOpenAI from langchain_openai
import feedparser # For parsing RSS feeds
from bs4 import BeautifulSoup # For cleaning HTML
from langchain.schema import Document # For document schema
from langchain_community.vectorstores import FAISS # For vector storage
from requests import get # For HTTP requests
from langchain.agents import AgentType # For agent type
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("./settings.env") 

os.environ['LANGSMITH_TRACING_V2'] = os.getenv('LANGSMITH_TRACING_V2')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
API_KEY_OPENAI =os.getenv('OPENROUTER_API')
API_KEY_COHERE = os.getenv('COHERE_API')
API_KEY_WEATHER = os.getenv('WEATHER_API')

# Initialize the ChatOpenAI model with OpenRouter API key and endpoint
llm = ChatOpenAI(model_name='gpt-4o-mini',api_key=API_KEY_OPENAI,base_url="https://openrouter.ai/api/v1")



# Function to fetch news from an RSS feed URL
def rss_getNews(feed_url):
    feed = feedparser.parse(feed_url)
    return [entry['title'] + "\n" + entry['summary'] for entry in feed.entries]


# Function to clean HTML content and extract text
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n")


# List of RSS feed URLs to fetch news from
urls = [
    'https://www.hespress.com/feed',
    'https://alyaoum24.com/feed',
    'https://al3omk.com/feed'
]
# Parse feeds and create Document objects for each news entry
documents = []
for url in urls:
    feed = feedparser.parse(url)
    for entry in feed.entries[:30]:  # Limit to 10 entries per feed
        tags = [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else []
        content = f"{entry.title}\n{entry.published}\n{entry.summary}"
        metadata = {"tags": tags}
        documents.append(Document(page_content=clean_html(content), metadata=metadata))



# Initialize Cohere embeddings
embedding = CohereEmbeddings(cohere_api_key=API_KEY_COHERE,model="embed-multilingual-v3.0",user_agent="langchain")
# Create FAISS vector store from documents and save locally
db = FAISS.from_documents(documents, embedding)
db.save_local('data_news')



# Create a retriever from the FAISS vector store
retriever = db.as_retriever()
# Define a tool for filtered news search using the retriever
def filtered_retriever_tool():
    def search_func(query):
        results = retriever.get_relevant_documents(query)
        return results
    return Tool(
        name='News Search',
        func=search_func,
        description="Search news articles by topic and tag"
    )
# Initialize the filtered news search tool
filtered_tool = filtered_retriever_tool()





# Format weather data with emojis and print nicely
def format_weather_with_emoji(weather_data, city_name="Casablanca"):
    weather_desc = weather_data["weather"][0]["description"]
    temp = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]

    # Map weather descriptions to emojis
    emoji_map = {
        "clear sky": "☀️",
        "few clouds": "🌤️",
        "scattered clouds": "⛅",
        "broken clouds": "☁️",
        "shower rain": "🌧️",
        "rain": "🌦️",
        "thunderstorm": "⛈️",
        "snow": "❄️",
        "mist": "🌫️"
    }
    emoji = emoji_map.get(weather_desc.lower(), "🌈")  # Default emoji
    result = (
        f"🌆 Weather in {city_name}:\n"
        f" - Condition: {weather_desc} {emoji}\n"
        f" - Temperature: {temp}°C (feels like {feels_like}°C)\n"
        f" - Humidity: {humidity}%\n"
        f" - Wind speed: {wind_speed} m/s\n"
        f"Have a great day! 😊"
    )

   
    return result



# Function to get coordinates (latitude, longitude) for a city using OpenWeatherMap
def get_coordinates(city: str):
    """Automatically fetch coordinates from city name"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&APPID={API_KEY_WEATHER}"
    response = get(url)
    if response.status_code == 200 and response.json():
        data = response.json()['coord']
        return data['lat'],data['lon']
    else:
        return response.json()
# Function to search weather for a city and format the result
def weather_search_tool(city, part: str = ""):
    lat,lon =get_coordinates(city)
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&APPID={API_KEY_WEATHER}"
    response = get(url)
    if response.status_code == 200:
        return format_weather_with_emoji(response.json(),city)
    else:
        return {"error": response.status_code, "message": response.text}


# Create a Tool for weather search with emoji formatting
weather_search = Tool(
    name='weather search',
    func=weather_search_tool,
    description="🔍 Search weather conditions by location 🌍. The tool fetches current weather data using latitude and longitude coordinates 📍 and explains results with fun emojis ☀️🌧️🌬️. The output is formatted neatly in multiple lines for better readability."
)


# Initialize conversation memory for the agent
memory = ConversationBufferWindowMemory(memory_key='chat_history',return_messages=True,k=8)
# Initialize the conversational agent with tools and memory
agent =  initialize_agent(tools=[filtered_tool,weather_search],llm=llm,memory=memory,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION)

def sendPromptToAgent(prompt:str):
# Invoke the agent with a sample query and print the output
  out=  agent.invoke(prompt)
  return str(out) 
