#!/bin/bash

# Load configuration
ENV_FILE=".env"

# Load environment variables
load_env() {
  if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
  fi

  if [ -z "$API_URL" ]; then
    echo "Error: API_URL is not set in $ENV_FILE."
    exit 1
  fi
}

# Login and save token
login() {
  echo "=== Login ==="
  read -p "Username: " USERNAME
  read -s -p "Password: " PASSWORD
  echo

  TOKEN=$(curl -s -X POST "$API_URL/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

  if [ -n "$TOKEN" ]; then
    echo "Saving new token..."
    echo "API_URL=$API_URL" > "$ENV_FILE"
    echo "TOKEN=$TOKEN" >> "$ENV_FILE"
    echo "Login successful!"
  else
    echo "Login failed. Check your username and password."
    exit 1
  fi
}

# Check if token exists
load_token() {
  if [ -z "$TOKEN" ]; then
    login
  fi
}

# Display help
show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

Options:
  --help                   Show this help message
  --list <folder>           List files in a folder
  --upload <file> <path>    Upload a file
  --delete <path>           Delete a file
  --rename <old> <new>      Rename a file
  --get <path>              Download a file
  --search <query>          Search for files

Examples:
  $0 --list somefolder
  $0 --upload localfile.txt somefolder/newfile.txt
  $0 --delete somefolder/file.txt
  $0 --rename somefolder/file.txt somefolder/renamedfile.txt
  $0 --get somefolder/file.txt
  $0 --search image
EOF
}

# Check for unauthorized access and re-login
check_auth() {
  if echo "$1" | grep -q "Unauthorized"; then
    echo "Token expired or invalid. Re-authenticating..."
    login
    return 1
  fi
  return 0
}

# --- Main ---

load_env
load_token

case "$1" in
  --help)
    show_help
    ;;

  --list)
    FOLDER="$2"
    RESPONSE=$(curl -s -X GET "$API_URL/resources/$FOLDER" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  --upload)
    FILE="$2"
    DEST="$3"
    if [ ! -f "$FILE" ]; then
      echo "Error: File $FILE does not exist."
      exit 1
    fi

    echo "Uploading $FILE to $DEST..."
    RESPONSE=$(curl --progress-bar -X POST "$API_URL/resources/$DEST" \
      -H "X-Auth: $TOKEN" \
      -H "Content-Type: application/octet-stream" \
      -H "User-Agent: bash-script" \
      -T "$FILE")

    echo
    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "Upload response: $RESPONSE"
    ;;

  --delete)
    PATH_TO_DELETE="$2"
    RESPONSE=$(curl -s -X DELETE "$API_URL/resources/$PATH_TO_DELETE" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  --rename)
    OLD_PATH="$2"
    NEW_PATH="$3"
    RESPONSE=$(curl -s -X PATCH "$API_URL/resources/$OLD_PATH?action=rename&destination=%2F$NEW_PATH&override=false&rename=false" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  --get)
    PATH_TO_GET="$2"
    FILENAME=$(basename "$PATH_TO_GET")

    echo "Downloading $PATH_TO_GET to ./$FILENAME ..."
    curl --progress-bar -X GET "$API_URL/raw/$PATH_TO_GET" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script" \
      --output "$FILENAME"

    if [ $? -eq 0 ]; then
      echo
      echo "File downloaded successfully: $FILENAME"
    else
      echo
      echo "Failed to download file."
    fi
    ;;

  --search)
    QUERY="$2"
    RESPONSE=$(curl -s -X GET "$API_URL/search/?query=$QUERY" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  *)
    echo "Unknown option: $1"
    echo "Use --help to see available commands."
    exit 1
    ;;
esac
