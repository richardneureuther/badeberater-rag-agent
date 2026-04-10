"""
Bodensee Badeberater: An agentic RAG assistant that recommends
swimming spots around Lake Constance using retrieved data and
live weather.

Run this script to interact with the badeberater in the console. 
Important: First run build_index to create the RAG for the agent!
"""
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import search_bathing_spots, get_weather

#load API key from .env
load_dotenv()

#initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)

#create system promt
SYSTEM_PROMPT = """
You are the Badeberater, a friendly, knowledgeable assistant that
helps people find great swimming spots around Lake Constance (Bodensee).

You have access to two tools:
1. search_bathing_spots: searches a database of 38 real swimming
   spots scraped from bodensee.de. Use this for any question about where
   to swim, what facilities a spot has, or to find spots matching specific
   criteria.
2. get_weather: fetches current weather and today's forecast for towns
   around the lake. Use this when the user asks about weather or when
   knowing the weather would improve your recommendation.

RULES:
- ALWAYS use the search_bathing_spots tool before recommending spots.
  Never recommend a spot purely from your own knowledge! Your training
  data may be outdated or wrong. The tool returns real, scraped data.
- When the user mentions a specific city, pass it as the city parameter
  to search_bathing_spots so results are filtered to that area.
- If a spot's opening hours or prices are missing from the tool results,
  tell the user to check the spot's official website (include the URL if available).
- You can answer in German or English: match the language the user
  writes in. German is default.
- Keep answers concise but warm. You're a local who loves the lake,
  not a corporate chatbot.
- Always cite the URL for each spot you recommend so the user can
  look up details.
- If the weather tool returns a fallback note about an unrecognized
  location, tell the user honestly that you're showing the weather of Konstanz instead.
"""

#combine tools
tools = [search_bathing_spots, get_weather]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


#run
def main():
    print("=" * 60)
    print("  Badeberater — Bodensee Swimming Spot Assistant")
    print("  Ready to give you all the swimming related information you need!")
    print("  You can interact with me in english or german.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Tschüss! Enjoy the lake.")
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                {"recursion_limit": 20},
            )
            #get final message, handle the output if the output is multiple conent blocks
            raw = result["messages"][-1].content
            if isinstance(raw, list):
            # Filter for text blocks and join them
                final_message = "\n".join(
                    block["text"] for block in raw
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                    final_message = raw
            print(f"\nBadeberater: {final_message}")
        except Exception as e:
            print(f"\nBadeberater: Sorry, something went wrong: {e}")


if __name__ == "__main__":
    main()