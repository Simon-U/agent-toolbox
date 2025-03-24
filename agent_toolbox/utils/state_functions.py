import uuid
from typing import List, Annotated, Dict
from typing_extensions import TypedDict


def update_plan(existing_plan: dict, update_dict: dict) -> dict:
    """Update existing plan with changes from update dict."""
    # Create new plan with deep copy of existing nodes
    if existing_plan == {}:
        return update_dict
    new_plan = {
        "nodes": {
            node_id: dict(node) for node_id, node in existing_plan["nodes"].items()
        },
        "edges": existing_plan["edges"],
    }

    # Update nodes with any changed values
    for node_id, update_node in update_dict["nodes"].items():
        if node_id in new_plan["nodes"]:
            new_plan["nodes"][node_id].update(update_node)

    return new_plan


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    merged_dict = dict1.copy()  # Start with dict1's keys and values
    merged_dict.update(
        dict2
    )  # Modify merged_dict with dict2's keys and values & return merged_dict
    return merged_dict


def reduce_list(left: list | None, right: list | None) -> list:
    """Append the right-hand list, replacing any elements with the same id in the left-hand list."""
    if not left:
        left = []
    if not right:
        right = []

    # Check if the lists are identical
    if left == right:
        return left

    left_, right_ = [], []
    for orig, new in [(left, left_), (right, right_)]:
        for val in orig:
            if not isinstance(val, dict):
                val = {"val": val}
            if "id" not in val:
                val["id"] = str(uuid.uuid4())
            new.append(val)
    # Merge the two lists
    left_idx_by_id = {val["id"]: i for i, val in enumerate(left_)}
    merged = left_.copy()
    for val in right_:
        if (existing_idx := left_idx_by_id.get(val["id"])) is not None:
            merged[existing_idx] = val
        else:
            merged.append(val)

    result = [val.get("val", val) for val in merged]
    return result

    """
    
    [{'step_description': {'tool_name': 'ToCalendarAssistant', 
    'request': 'Please check my calendar for today and summarize any events or meetings I have scheduled.'}, 
    'dependencies': []}, 
    
    {'step_description': {'tool_name': 'ToNotionAssistant', 
    'request': 'Please list my open todos from the Notion workspace.'}, 
    'dependencies': []}]
    """


def pop(left, right):
    # Filter out elements from 'left' that are present in 'right'
    if left == []:
        return right
    if right != []:
        test = [item for item in left if item not in right]
        return test
    return left
