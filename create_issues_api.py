#!/usr/bin/env python3
"""
Script to create GitHub issues from issues_to_create.json using GitHub API
This script uses the PyGithub library or direct REST API calls

Prerequisites:
    pip install PyGithub

Usage:
    python create_issues_api.py <github_token> [repo_name] [issues_file]
    
    github_token: Your GitHub personal access token (or set GITHUB_TOKEN env var)
    repo_name: Repository in format 'owner/repo' (default: Kaladn/CortexOS_NLP)
    issues_file: Path to JSON file (default: issues_to_create.json)

Examples:
    python create_issues_api.py my_token
    python create_issues_api.py my_token owner/repo
    python create_issues_api.py my_token owner/repo custom_issues.json
    
Or set GITHUB_TOKEN environment variable:
    export GITHUB_TOKEN=your_token_here
    python create_issues_api.py
"""

import json
import os
import sys
import time

try:
    from github import Github
    USE_PYGITHUB = True
except ImportError:
    print("PyGithub not installed. Will use REST API with requests.")
    USE_PYGITHUB = False
    try:
        import requests
    except ImportError:
        print("ERROR: Neither PyGithub nor requests is installed.")
        print("Please install one: pip install PyGithub  OR  pip install requests")
        sys.exit(1)

# Default values (can be overridden by command line arguments)
DEFAULT_REPO_NAME = "Kaladn/CortexOS_NLP"
DEFAULT_ISSUES_FILE = "issues_to_create.json"

def load_issues(filepath):
    """Load issues from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_issues_with_pygithub(token, repo_name, issues):
    """Create issues using PyGithub library."""
    print(f"Connecting to GitHub as authenticated user...")
    g = Github(token)
    
    try:
        repo = g.get_repo(repo_name)
        print(f"Repository: {repo.full_name}")
        print(f"Creating {len(issues)} issues...\n")
        
        created_count = 0
        for idx, issue_data in enumerate(issues, 1):
            title = issue_data['title']
            body = issue_data['description']
            labels = issue_data['labels']
            
            print(f"[{idx}/{len(issues)}] Creating: {title}")
            
            try:
                issue = repo.create_issue(
                    title=title,
                    body=body,
                    labels=labels
                )
                print(f"    ✓ Created: {issue.html_url}")
                created_count += 1
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
        
        print(f"\n{'='*80}")
        print(f"Successfully created {created_count}/{len(issues)} issues")
        print(f"{'='*80}")
        print(f"\nView all issues at: https://github.com/{repo_name}/issues")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

def create_issues_with_requests(token, repo_name, issues):
    """Create issues using requests library and GitHub REST API."""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    api_url = f"https://api.github.com/repos/{repo_name}/issues"
    
    print(f"Creating {len(issues)} issues using GitHub REST API...\n")
    
    created_count = 0
    for idx, issue_data in enumerate(issues, 1):
        title = issue_data['title']
        body = issue_data['description']
        labels = issue_data['labels']
        
        print(f"[{idx}/{len(issues)}] Creating: {title}")
        
        payload = {
            'title': title,
            'body': body,
            'labels': labels
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            issue_url = response.json()['html_url']
            print(f"    ✓ Created: {issue_url}")
            created_count += 1
            time.sleep(1)  # Rate limiting
        except requests.exceptions.HTTPError as e:
            print(f"    ✗ Failed: HTTP {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Successfully created {created_count}/{len(issues)} issues")
    print(f"{'='*80}")
    print(f"\nView all issues at: https://github.com/{repo_name}/issues")

def main():
    # Parse command line arguments
    token = None
    repo_name = DEFAULT_REPO_NAME
    issues_file = DEFAULT_ISSUES_FILE
    
    # Get GitHub token
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.environ.get('GITHUB_TOKEN')
    
    # Get repository name (optional)
    if len(sys.argv) > 2:
        repo_name = sys.argv[2]
    
    # Get issues file path (optional)
    if len(sys.argv) > 3:
        issues_file = sys.argv[3]
    
    if not token:
        print("ERROR: GitHub token not provided")
        print("\nUsage:")
        print("  python create_issues_api.py <github_token> [repo_name] [issues_file]")
        print("\nExamples:")
        print("  python create_issues_api.py my_token")
        print("  python create_issues_api.py my_token owner/repo")
        print("  python create_issues_api.py my_token owner/repo custom.json")
        print("\nOr set environment variable:")
        print("  export GITHUB_TOKEN=your_token_here")
        print("  python create_issues_api.py")
        print("\nGet a token at: https://github.com/settings/tokens")
        print("Required scopes: 'repo' or 'public_repo'")
        sys.exit(1)
    
    # Load issues
    try:
        issues = load_issues(issues_file)
    except FileNotFoundError:
        print(f"ERROR: {issues_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {issues_file}: {e}")
        sys.exit(1)
    
    print("="*80)
    print(f"GitHub Issues Creator for {repo_name}")
    print("="*80)
    print(f"Issues file: {issues_file}")
    print(f"Issues to create: {len(issues)}")
    print()
    
    # Create issues using available library
    if USE_PYGITHUB:
        create_issues_with_pygithub(token, repo_name, issues)
    else:
        create_issues_with_requests(token, repo_name, issues)

if __name__ == "__main__":
    main()
