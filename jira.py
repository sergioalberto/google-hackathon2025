import requests
import json
import os

JIRA_URL = "https://sergioalbertogq.atlassian.net"
EMAIL = "sergioalbertogq@gmail.com"
API_TOKEN = os.environ.get("JIRA_API_TOKEN", "") # Generate from Jira profile settings

# Example JQL query to get issues from a specific project
jql_query = "project = 'My Scrum Project'"
# Fields to include (customize as needed)
fields = "summary,description,status,issuetype,priority,reporter,assignee,labels,comments,attachment"

all_issues = []
start_at = 0
max_results = 50 # Adjust based on Jira's API limits and performance

while True:
    response = requests.get(
        f"{JIRA_URL}/rest/api/3/search",
        headers={"Accept": "application/json"},
        auth=(EMAIL, API_TOKEN),
        params={
            "jql": jql_query,
            "fields": fields,
            "startAt": start_at,
            "maxResults": max_results
        }
    )
    response.raise_for_status() # Raise an exception for bad status codes
    data = response.json()

    issues = data.get("issues", [])
    if not issues:
        break

    all_issues.extend(issues)
    start_at += len(issues)

    if start_at >= data.get("total", 0):
        break

with open("jira_issues.json", "w", encoding="utf-8") as f:
    json.dump(all_issues, f, ensure_ascii=False, indent=4)

print(f"Exported {len(all_issues)} issues to jira_issues.json")
