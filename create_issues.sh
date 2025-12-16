#!/bin/bash
# Script to create GitHub issues from issues_to_create.json
# This script uses GitHub CLI (gh) to create issues
# 
# Prerequisites:
# 1. Install GitHub CLI: https://cli.github.com/
# 2. Authenticate: gh auth login
# 3. Run this script: ./create_issues.sh [REPO] [ISSUES_FILE]
#
# Usage:
#   ./create_issues.sh                           # Uses defaults
#   ./create_issues.sh owner/repo                # Custom repo
#   ./create_issues.sh owner/repo issues.json    # Custom repo and file

REPO="${1:-Kaladn/CortexOS_NLP}"
ISSUES_FILE="${2:-issues_to_create.json}"

echo "=========================================="
echo "Creating GitHub Issues for $REPO"
echo "=========================================="
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    exit 1
fi

# Check if issues file exists
if [ ! -f "$ISSUES_FILE" ]; then
    echo "Error: $ISSUES_FILE not found"
    exit 1
fi

# Parse JSON and create issues using jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to parse JSON."
    echo "Installation: https://stedolan.github.io/jq/download/"
    exit 1
fi

# Read the JSON file and create each issue
issue_count=$(jq 'length' "$ISSUES_FILE")
echo "Found $issue_count issues to create"
echo ""

for i in $(seq 0 $((issue_count - 1))); do
    title=$(jq -r ".[$i].title" "$ISSUES_FILE")
    description=$(jq -r ".[$i].description" "$ISSUES_FILE")
    labels=$(jq -r ".[$i].labels | join(\",\")" "$ISSUES_FILE")
    priority=$(jq -r ".[$i].priority" "$ISSUES_FILE")
    
    echo "Creating issue $((i + 1))/$issue_count: $title"
    
    # Create the issue
    gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body "$description" \
        --label "$labels" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Created successfully"
    else
        echo "✗ Failed to create"
    fi
    echo ""
    
    # Add a small delay to avoid rate limiting
    sleep 1
done

echo "=========================================="
echo "Issue creation complete!"
echo "=========================================="
echo ""
echo "View all issues at: https://github.com/$REPO/issues"
