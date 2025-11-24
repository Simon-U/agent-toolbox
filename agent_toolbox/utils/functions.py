import json
import os
import pandas as pd
from datetime import datetime


def get_parameter_value(key, json_path=None):
    # Get the directory of the current file
    if not json_path:
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
        # Define the relative path to the JSON file
        json_path = os.path.join(
            parent_dir, "agentic_flows/causal_explanation/config/parameters.json"
        )
    # Load the JSON data from the file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Return the value for the specified key
    return data.get(key, None)


def convert_to_utc(date_string):
    """Handle utc time stamps"""
    # Step 1: Parse the date string into a datetime object
    local_dt = datetime.fromisoformat(date_string)

    # Step 2: Convert to UTC
    # utc_dt = local_dt.astimezone(pytz.UTC)

    # Step 3: Format the datetime object as a string without timezone information
    utc_date_string = local_dt.strftime("%Y-%m-%d %H:%M:%S")

    return utc_date_string


def transform_timestamp(timestamp: str) -> str:
    """Transform timestamps to appropiate date format"""
    # Convert the timestamp to a datetime object
    datetime_obj = pd.to_datetime(timestamp)

    # Format the datetime object as a string with the desired format
    formatted_date = datetime_obj.strftime("%Y-%m-%d")

    return formatted_date


def is_before(publish_date, benchmark_date):
    """This function checks if the publishdate is before the benchmark date

    Args:
        publish_date (_type_): _description_
        benchmark_date (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Convert publish_date to datetime if it's a pandas.Timestamp
    if isinstance(publish_date, pd.Timestamp):
        publish_dt = publish_date.to_pydatetime()
    else:
        publish_dt = datetime.fromisoformat(publish_date)

    # Convert benchmark_date to datetime
    benchmark_dt = datetime.fromisoformat(benchmark_date)

    # Compare the datetime objects
    return publish_dt < benchmark_dt


def get_publish_date(item):
    publish_date = (
        item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("article:published_time", "")
        or item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("article:modified_time", "")
        or item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("article:updated_time", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("article:created", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("og:updated_time", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("og:published_time", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("og:release_date", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("date", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("datePublished", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dateModified", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("last-modified", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("pubdate", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("publishdate", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("timestamp", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dc.date", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dcterms.created", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dcterms.modified", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dcterms.date", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("dcterms.issued", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("created", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("revised", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("news_publish_date", "")
        or item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("originalpublicationdate", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("datepublished", "")
        or item.get("pagemap", {}).get("metatags", [{}])[0].get("datemodified", "")
        or item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("article_date_original", "")
        or item.get("pagemap", {})
        .get("metatags", [{}])[0]
        .get("article_date_published", "")
    )
    return publish_date
