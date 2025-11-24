# The agent library is a collection of agents modules

## What is an agent module?
Agent modules are langchain runnables, which have an invoke functionallity `agent.invoke()`. They will take different inputs, process them and 
return a result. The agent modules are specialised task executres. They focus on specific set of skills. These agents need to have a specific format to be used by agentic flows.

## Agentic flows

An agentic flow is a ccollection of agent modules which are executed in a specific order to fullfill a more complex request. 
Each agent modules take up speciffic tasks. Agent modules are executed in nodes and linked via edges to each other.
Each edge can be either a fixed edge between A -> B or an conditional edge between multiple nodes. 
In a conditional case we need a router function. The router function maps certain conditions to certain edges. This can be either 
deterministic, with a preset of conditions, or an LLM can function as a router. Some more can be found on [LangGraph](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)

## Folder structure

This module contains two main folders:

    ├── agentic_flow
    ├── base
    │   ├── agent_library
    │   │   ├── agent_modules
    │   │   ├── Agent_factory.py
    │   │   ├── base.py
    │   ├── tool_library
    │   ├── memory
    └── README.md

in agentic_flow is a collection of high level agents that automise complex workflows while
base holds all the tools to build these complex workflows.

In the respective folders are more README files to explain certain patterns, best practices and default setups.

1. [Definition of submodules in base](agents/base/README.md)
2. [Agent modules](agents/base/agent_library/README.md)
3. [Agent modules blueprint](agents/base/agent_library/agent_modules/0_blueprint/README.md)
3. [Agentic Flow blueprint](agents/agentic_flows/0_blueprint/README.md)


## Init files, modules

`MOD_INIT_PATH=/home/simon/Documents/Pure_Inference/Malvius/agents`
Set the following path for your directory 

The best practices for the __init__.py are thefollowing:

1. Only put an init file into a folder/module you want to expose
2. The 0_blueprint folders in agentic_flow and agent_modules have a default init file which wil import all exposed classes from the agent.py and utils.py and ignore other files
3. If you want to expose other files you can add then to `__submodules__ = ["agent", "utils"]` they will be automatically included
The 
```
# <AUTOGEN_INIT>

# </AUTOGEN_INIT>
``` 
Indicate where mkinit will put the auto generate parts
4. Within each file expose the class, function or Variable you want to expose with `__all__ = ['Yourclass']`
5. Run `mkinit "$MOD_INIT_PATH" --recursive --noattrs --relative -w` This will create all the imports, update the files