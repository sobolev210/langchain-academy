import os
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with searching cheap flights for given direction for user and booking them. Before booking, ask user if he agrees to the flight you chose. Do the best effort to find flight without asking extra info from user. Do not make more than 2 searches"
)

booking_sys_message = SystemMessage(
    content="You are a helpful assistant tasked with searching cheap flights for given direction for user and booking them. You are invoked after user approves booking. Imitate you're doing some booking with some emojis. Keep message short, don't ask for any extra data, assume everything is provided"
)


def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    travel_class: str = "economy",
) -> str:
    """
    Search for flights using SerpAPI Google Flights API.

    Args:
        departure_airport: Departure airport code (e.g., 'LAX', 'JFK')
        arrival_airport: Arrival airport code (e.g., 'AUS', 'SFO')
        outbound_date: Departure date in YYYY-MM-DD format (e.g., '2025-10-14')
        return_date: Optional return date in YYYY-MM-DD format for round trips
        adults: Number of adult passengers (default: 1)
        travel_class: Travel class - 'economy', 'premium_economy', 'business', or 'first' (default: 'economy')

    Returns:
        A formatted string with flight information including prices, airlines, and durations
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY environment variable not set"

    # Map travel class strings to SerpAPI numeric values
    travel_class_map = {"economy": 1, "premium_economy": 2, "business": 3, "first": 4}
    travel_class_value = travel_class_map.get(travel_class.lower(), 1)

    # Build the API request parameters
    params = {
        "engine": "google_flights",
        "departure_id": departure_airport.upper(),
        "arrival_id": arrival_airport.upper(),
        "outbound_date": outbound_date,
        "adults": adults,
        "travel_class": travel_class_value,
        "currency": "USD",
        "hl": "en",
        "api_key": api_key,
    }

    # Add return date if provided (makes it a round trip)
    if return_date:
        params["return_date"] = return_date
        params["type"] = "1"  # Round trip
    else:
        params["type"] = "2"  # One way

    try:
        # Make the API request
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for errors in the response
        if "error" in data:
            return f"Error from SerpAPI: {data['error']}"

        # Parse and format the results
        results = []

        # Check for best flights first
        flights_list = data.get("best_flights", [])
        if not flights_list:
            flights_list = data.get("other_flights", [])

        if not flights_list:
            return f"No flights found from {departure_airport} to {arrival_airport} on {outbound_date}"

        # Format the top 5 flights
        for idx, flight in enumerate(flights_list[:5], 1):
            price = flight.get("price", "N/A")
            airline = flight["flights"][0].get("airline", "Unknown")
            total_duration = flight.get("total_duration", 0)
            hours = total_duration // 60
            minutes = total_duration % 60

            # Get departure and arrival times
            first_flight = flight["flights"][0]
            dep_time = first_flight["departure_airport"]["time"]

            last_flight = flight["flights"][-1]
            arr_time = last_flight["arrival_airport"]["time"]

            # Check for layovers
            num_stops = len(flight["flights"]) - 1
            stops_text = "Direct" if num_stops == 0 else f"{num_stops} stop(s)"

            # Carbon emissions info
            carbon = flight.get("carbon_emissions", {})
            carbon_text = ""
            if carbon:
                this_flight_kg = carbon.get("this_flight", 0) / 1000
                carbon_text = f", Carbon: {this_flight_kg:.0f} kg"

            results.append(
                f"{idx}. ${price} - {airline}\n"
                f"   {dep_time} → {arr_time} ({hours}h {minutes}m, {stops_text}){carbon_text}"
            )

        # Add price insights if available
        price_insights = data.get("price_insights", {})
        if price_insights:
            lowest = price_insights.get("lowest_price")
            price_level = price_insights.get("price_level", "")
            if lowest:
                results.append(f"\nLowest price: ${lowest} ({price_level})")

        trip_type = "round trip" if return_date else "one-way"
        header = f"Found {len(flights_list)} flights ({trip_type}) from {departure_airport} to {arrival_airport}:\n\n"

        return header + "\n".join(results)

    # except requests.exceptions.RequestException as e:
    #     return f"Error making API request: {str(e)}"
    except Exception as e:
        raise e
        return f"Error processing flight data: {str(e)}"


def assistant(state: MessagesState):
    return {"messages": model_with_tools.invoke([sys_msg] + state["messages"])}


def booking_assistant(state: MessagesState):
    return {
        "messages": model_with_tools.invoke([booking_sys_message] + state["messages"])
    }


def get_human_approval(state: MessagesState):
    return {"messages": HumanMessage("Approved")}


tools = [search_flights]


model = ChatOpenAI(model="gpt-5-mini")
model_with_tools = model.bind_tools(tools)


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_approval", get_human_approval)
builder.add_node("booking_assistant", booking_assistant)


builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    path_map={"tools": "tools", "__end__": "human_approval"},
)
builder.add_edge("tools", "assistant")
builder.add_edge("human_approval", "booking_assistant")
builder.add_edge("booking_assistant", END)

memory = MemorySaver()
graph = builder.compile(
    # checkpointer=memory
)  # fmt: skip


# if __name__ == "__main__":
#     res = search_flights("BEG", "GRZ", "2026-05-15")
#     print(res)
