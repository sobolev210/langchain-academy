"""
conferences
hackathons
intensives


"""

import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResultsQualityAssesment(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the results are sufficient or need to continue searching."
    )
    required_improvements: str = Field(
        description="If the results are not sufficient, what are the required improvements (i.e. what's not fully correct or might be missing or might be improved) comparing to current documents."
    )


query_generation_instruction = SystemMessage(
    content="""You will be given a type of opportunity in ML field that AI assistant is searching for.

    Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

    Pay particular attention to type of opportunity and additional context from previous messages

    Create well structured search query.
    Keep it concise: **max 200 characters.**
    """
)

search_instructions = SystemMessage(
    """
    You're an assistant asked to help with searching different opportunities in Machine learning field. You will be given a type of opportunity you need to search for, e.g. hackathons. Additionally, you will be given a time constraint - within which period to seach for such opportunities, e..g within next 3 months, and other information such as in which location to search if applicable.
    Your end goal is to help the user to find the best opportunities to develop and grow in Machine Learning field (gen AI, traditional machine learning).
    Now, given the type of opportunity to search, make the search query.
    Opportunity type: {opportunity_type}
    Additional context: {additional_context}
    Do not ask for any additional information, just make the search query.
    Ensure you're looking for future events only, not past ones.
    Additionally, here're proposed refinements from previous search results, incorporate them into your search query: {required_improvements}. If empty, just ignore it
    """
)

generate_report_instructions = SystemMessage(
    """
You're given search results for specific type of opportunities in ML field.
Your task is to compose structured report based on found documents.
Provide key information, such as dates, location, duration, topic, cost, or any other information you find useful. The set of attributes is flexible.

Provide final result in text format, eg.
--------
Event 1
Topic: ...
Short summary: ...
Date: ...
...
Link: ...
--------
Event 2
Topic: ...
Short summary: ...
Date: ...
...
Link: ...

Now, here's the list of documents found:
{found_documents}
"""
)


class ResearcherInstanceInternalState(MessagesState):
    opportunity_type: str
    additional_context: str
    last_assessment_results: ResultsQualityAssesment
    found_documents: Annotated[list, operator.add]
    researcher_reports: list[str]


def search_web(state: ResearcherInstanceInternalState):
    """Retrieve docs from web search"""

    # Search
    tavily_search = TavilySearch(max_results=3, include_raw_content=True)

    last_assessment_results = state["last_assessment_results"]

    # Search queryx
    structured_llm = llm.with_structured_output(SearchQuery)
    search_instruction = SystemMessage(
        search_instructions.content.format(
            opportunity_type=state["opportunity_type"],
            additional_context=state["additional_context"],
            required_improvements=last_assessment_results.required_improvements
            if last_assessment_results
            else "",
        )
    )
    search_query: SearchQuery = structured_llm.invoke(
        [search_instruction] + state["messages"]
    )
    raw_query = (search_query.search_query or "").strip()
    if len(raw_query) > 380:
        raw_query = raw_query[:380]

    # Search
    data = tavily_search.invoke({"query": raw_query})
    if not isinstance(data, dict):
        return {"found_documents": [""]}
    search_docs = data.get("results", data)
    if not isinstance(search_docs, list):
        search_docs = [search_docs]

    # Format
    formatted_documents = []
    for doc in search_docs:
        if isinstance(doc, str):
            formatted_documents.append(f"<Document>\n{doc}\n</Document>")
            continue
        if isinstance(doc, dict):
            title = doc.get("title") or ""
            url = doc.get("url") or doc.get("href") or ""
            content = (
                doc.get("raw_content") or doc.get("content") or doc.get("snippet") or ""
            )
            header = f'<Document href="{url}"/>' if url else "<Document>"
            title_block = f"Title: {title}\n" if title else ""
            formatted_documents.append(f"{header}\n{title_block}{content}\n</Document>")
            continue
        formatted_documents.append(f"<Document>\n{doc}\n</Document>")

    formatted_search_docs = "\n\n---\n\n".join(formatted_documents)

    return {"found_documents": [formatted_search_docs]}


def generate_report(state: ResearcherInstanceInternalState):
    previous_messages = state["messages"]
    found_documents = state["found_documents"]
    formatted_instructions = SystemMessage(
        content=generate_report_instructions.content.format(
            found_documents=found_documents
        )
    )
    result = llm.invoke([formatted_instructions] + previous_messages)
    return {"researcher_reports": [result.content]}


# todo
def assess_results_quality(state: ResearcherInstanceInternalState):
    pass


researcher_builder = StateGraph(ResearcherInstanceInternalState)
researcher_builder.add_node("search_web", search_web)
researcher_builder.add_node("generate_report", generate_report)
# researcher_builder.add_node("assess_results_quality", assess_results_quality)

researcher_builder.add_edge(START, "search_web")
researcher_builder.add_edge("search_web", "generate_report")
researcher_builder.add_edge("generate_report", END)
# TODO add edge


identifying_research_tasks_instructions = SystemMessage(
    """
    You're an assistant asked to help with searching different opportunities in Gen AI and Machine learning field.
    Your first task is to identify the types of opportunities user is searching for, and also some additional context,
    like within which time period, location, offline/online, etc.

    Until other types or other context is provided, use these as your defaults:

    Types:
    - conferences
    - hackathons
    - intensives
    - events
    - summer / winter programs by big tech companies

    Time frame:
    Within next 6 months

    Location:
    Belgrade, Serbia. For intensives, winter/summer programs - it can be online
    
    """
)

summarize_research_results_instructions = HumanMessage(
    """
    You'll be given list of search results for different types of opportunities user was searching for. Summarize search results for different types of opportunities into a structured report.
    Divide it into sections, highlight some most important events.
    Do not omit any of the opportunities returned by web researchers
    Here's the list of search results:
    {search_results}
    """
)


class ResearchTask(BaseModel):
    opportunity_type: str
    additional_context: str


class ResearchTasksList(BaseModel):
    tasks: list[ResearchTask]


class MlOpportunitiesResearcherState(MessagesState):
    research_tasks: list[ResearchTask]
    human_feedback_on_tasks: str
    researcher_reports: Annotated[list, operator.add]
    opportunities_report: str


def identify_research_tasks(state: MlOpportunitiesResearcherState):
    previous_messages = state["messages"]
    messages = (
        [identifying_research_tasks_instructions] + previous_messages
        if len(previous_messages) <= 1
        else previous_messages
    )
    structured_llm = llm.with_structured_output(ResearchTasksList)
    result = structured_llm.invoke(messages)
    return {
        "research_tasks": result.tasks,
        "messages": messages,
    }


def human_feedback(state: MlOpportunitiesResearcherState):
    pass


def initiate_researches_or_iterate_to_tasks(state: MlOpportunitiesResearcherState):
    human_feedback = state["human_feedback_on_tasks"]
    if human_feedback is None or human_feedback.lower() == "approve":
        research_tasks = state["research_tasks"]
        return [
            Send(
                "conduct_research",
                arg={
                    "opportunity_type": r_task.opportunity_type,
                    "additional_context": r_task.additional_context,
                },
            )
            for r_task in research_tasks
        ]
    else:
        return "identify_research_tasks"


def summarize_final_result(state: MlOpportunitiesResearcherState):
    previous_messages = state["messages"]
    research_results = state["researcher_reports"]
    formatted_instructions = HumanMessage(
        content=summarize_research_results_instructions.content.format(
            search_results=research_results
        )
    )
    result = llm.invoke(previous_messages + [formatted_instructions])
    return {"opportunities_report": result.content}


ml_opportunities_researcher_builder = StateGraph(MlOpportunitiesResearcherState)
ml_opportunities_researcher_builder.add_node(
    "identify_research_tasks", identify_research_tasks
)
ml_opportunities_researcher_builder.add_node("human_feedback", human_feedback)
ml_opportunities_researcher_builder.add_node(
    "conduct_research", researcher_builder.compile()
)
ml_opportunities_researcher_builder.add_node(
    "summarize_final_result", summarize_final_result
)


ml_opportunities_researcher_builder.add_edge(START, "identify_research_tasks")
ml_opportunities_researcher_builder.add_edge(
    "identify_research_tasks", "human_feedback"
)
# https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges
ml_opportunities_researcher_builder.add_conditional_edges(
    "human_feedback",
    initiate_researches_or_iterate_to_tasks,
    {
        "identify_research_tasks": "identify_research_tasks",
        "conduct_research": "conduct_research",
    },
)
ml_opportunities_researcher_builder.add_edge(
    "conduct_research", "summarize_final_result"
)
ml_opportunities_researcher_builder.add_edge("summarize_final_result", END)

# TODO include kaggle
memory = MemorySaver()
graph = ml_opportunities_researcher_builder.compile(
    interrupt_before=["human_feedback"], 
    # checkpointer=memory
)  # fmt: skip

# config = {"configurable": {"thread_id": "15"}}
# graph.invoke(
#     {"messages": HumanMessage("Hey what are opportunities in nearest 3 months?")},
#     config=config,
# )
# graph.update_state(config, {"human_feedback_on_tasks": None}, as_node="human_feedback")
# graph.invoke(None, config)
# graph.get_state(config).next

# tavily_search = TavilySearch(max_results=3, include_raw_content=True)

# query = {
#     "query": "Please specify: 1) type of opportunity (hackathon, internship, fellowship, competition, conference/workshop, bootcamp, course), 2) timeframe (e.g., within next 3 months), 3) location (online or city/country), 4) your experience level (student/early-career/experienced), and 5) focus area if any (generative AI, NLP, CV, RL, etc.)."
# }
# res = tavily_search.invoke({"query": query["query"]})
