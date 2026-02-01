## Custom Agents

### Slang Agent
**Location:** `module-1/studio/slang-agent.py`

Searches Urban Dictionary for slang definitions and rewrites them as short rap quatrains with usage examples.

### Dictionary Agent
**Location:** `module-2/studio/dictionary_agent.py`

General-purpose assistant that maintains a persistent dictionary of definitions the user has asked about, storing them in SQLite for future reference.

### Cheap Trips Agent
**Location:** `module-3/studio/cheap-trips-agent.py`

Flight search and booking agent that uses SerpAPI to find cheap flights, presents options to the user, and simulates booking after approval.

### ML Opportunities Researcher with Memory
**Location:** `module-5/studio/ml_opportunities_researcher_with_memory.py`

Research agent that searches for ML-related opportunities (conferences, hackathons, intensives) using Tavily with iterative query refinement, and uses PostgresStore to remember past searches and user preferences across sessions.
