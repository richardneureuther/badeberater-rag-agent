"""
LangChain tools for my Bodensee bathing spot agent.

Each tool is a function wrapped with the @tool decorator, which exposes it to the agent along with its docstring

"""

from functools import lru_cache

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bodensee_spots"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


#esnure that the embedder is only run once
@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """
    Load the persisted Chroma index once and cache it.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

@tool
def search_bathing_spots(query: str, city: str = "") -> str:
    """
    Search for swimming spots around Lake Constance (Bodensee).

    Use this tool whenever the user asks about where to swim, bathing
    spots, Strandbäder, Freibäder, or similar. The query should describe
    what the user is looking for in their own words — e.g. "family-friendly
    with a slide", "quiet", "FKK area", "Near a city or place".

    Args:
        query: Natural-language description of what the user wants.
            Can be in German or English. Required.
        city: Optional city name to filter by (e.g. "Konstanz",
            "Überlingen","Bregenz"). Use this when the user explicitly mentions
            a city. Leave empty to search all spots around the lake.

    Returns:
        A formatted string with up to 5 matching spots, including title,
        location, URL, and a short description snippet for each.
    """
    vectorstore = _get_vectorstore()

    ##fetch all results if we have a city match so we dont loose any info due to matching issues
    k = 38 if city.strip() else 5
    results = vectorstore.similarity_search(query, k=k)
    #
    if city.strip():
        city_lower = city.strip().lower()
        results = [
            doc for doc in results
            if city_lower in doc.metadata.get("plz_ort", "").lower()
        ]
        results = results[:5]  # keep only top 5 after filterin

    #tell the agent if there were no results
    if not results:
        if city:
            #tell the model that the query filter didnt give any information and they could try without the filter
            return (
                f"No swimming spots found matching '{query}' in {city}. "
                f"You may want to retry without the city filter."
            )
        return f"No swimming spots found matching '{query}'."

    ##create output readable for the llm
    formatted = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("title", "(no title)")
        plz_ort = doc.metadata.get("plz_ort", "")
        url = doc.metadata.get("url", "")
        #trim full text to a reasonable lenght for LLM
        snippet = doc.page_content[:500].replace("\n", " ")
        formatted.append(
            f"[{i}] {title} ({plz_ort})\n"
            f"    URL: {url}\n"
            f"    {snippet}..."
        )

    return "\n\n".join(formatted)


#tool to fetch the current weather at a bathing spot
@tool
def get_weather(location:str):
    """
    Get the curren weather and today foreacast for locations around lace Constance. 

    Use this tool when the user asks about the weather, when they want to know if it is a good day to go swimming. 
    (e.g "Wo kann ich heute schwimmen gehen" or "Ist heute ein guter Tag um schwimmen zu gehen")
      
       
     Args:
        location: A city name around Lake Constance, e.g. "Konstanz",
            "Überlingen", "Friedrichshafen", "Bregenz". Defaults to
            Konstanz if the location is unclear.

    Returns:
        A short natural-language summary of current temperature,
        conditions, and today's max temperature.
        IMPORTANT: If the response contains a note like "(location 'X' not recognized,
        showing Konstanz instead)", you MUST tell the user in your
        final answer that you couldn't find weather for their requested
        location and are showing Konstanz weather as the nearest
        available reference.
    """

    #hardcode most common bodensee locations for simplicity
    KNOWN_LOCATIONS = {
        "konstanz":        (47.6603, 9.1758),
        "überlingen":      (47.7697, 9.1686),
        "meersburg":       (47.6947, 9.2711),
        "friedrichshafen": (47.6542, 9.4781),
        "lindau":          (47.5598, 9.6807),
        "bregenz":         (47.5031, 9.7471),
        "radolfzell":      (47.7422, 8.9694),
        "singen":          (47.7594, 8.8392),
    }

    key = location.strip().lower()
    if key not in KNOWN_LOCATIONS:
        #default to konstanz and tell the model
        lat, lon = KNOWN_LOCATIONS["konstanz"]
        fallback_note = f" (location '{location}' not recognized, showing Konstanz instead)"
    else:
        lat, lon = KNOWN_LOCATIONS[key]
        fallback_note = ""

    #Open Meteo API 
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weather_code,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,weather_code"
        f"&timezone=Europe/Berlin"
        f"&forecast_days=1"
    )

    #try to get a response
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    #tell the agent if the call didnt succeed
    except requests.RequestException as e:
        return f"Could not fetch weather data: {e}"
    
    #get current temperature and forecast
    current = data.get("current", {})
    daily = data.get("daily", {})

    #store results 
    temp_now = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")
    temp_max = daily.get("temperature_2m_max", [None])[0]
    temp_min = daily.get("temperature_2m_min", [None])[0]

    condition = _weather_code_to_text(weather_code)

    return (
        f"Weather in {location.title()}{fallback_note}:\n"
        f"  Currently: {temp_now}°C, {condition}, wind {wind} km/h\n"
        f"  Today: {temp_min}°C to {temp_max}°C"
    )

from bs4 import BeautifulSoup


@tool
def get_water_temperature(location: str = "Konstanz") -> str:
    """
    Get the current water temperature of Lake Constance (Bodensee).

    Use this tool when the user asks about water temperature, whether
    the water is warm enough for swimming, wether they can go swimming today or when water temperature
    would improve your recommendation.

    Args:
        location: Not used since the water temperature for the bodesee is generally pretty 
        much the same regardless if you are in Bregenz or Konstanz.

    Returns:
        A short summary of the current water temperature with a
        swimming comfort assessment.
    """
    url = "https://www.wassertemperatur.org/bodensee/"

    #try to acess the webpage
    try:
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Badeberater-RAG-Agent)"
        })
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch water temperature: {e}"
    #store response
    soup = BeautifulSoup(response.text, "html.parser")

    #get the headline temperature 
    temp_span = soup.find("span", style=lambda s: s and "color:blue" in s)

    if not temp_span:
        return "Could not find water temperature on the page."

    temp_text = temp_span.get_text(strip=True)
    try:
        temp = float(temp_text.replace("°C", "").replace("°", "").strip())
    except ValueError:
        return f"Could not parse water temperature: '{temp_text}'"

    #assess the swimming comfort 
    if temp >= 20:
        comfort = "Warm and comfortable for swimming."
    elif temp >= 16:
        comfort = "Pleasant, though a bit refreshing."
    elif temp >= 12:
        comfort = "Cool — only for hardy swimmers or quick dips."
    else:
        comfort = "Too cold for most swimmers."
    #return answer for the agent
    return (
        f"Bodensee water temperature: {temp}°C "
        f"(source: wassertemperatur.org)\n"
        f"Assessment: {comfort}"
    )

def _weather_code_to_text(code: int | None) -> str:
    """
    Small helper function that translates Open-Meteo's WMO weather codes into short plain-English
    descriptions. 
    """
    if code is None:
        return "unknown conditions"
    if code == 0:
        return "clear sky"
    if code in (1, 2, 3):
        return "partly cloudy"
    if code in (45, 48):
        return "foggy"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "thunderstorm"
    return f"weather code {code}"