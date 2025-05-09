#!/bin/bash

# Load configuration
ENV_FILE=".env"

# Load environment variables
load_env() {
  if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
  fi

  if [ -z "$HOSTING_URL" ]; then
    echo "Error: HOSTING_URL is not set in $ENV_FILE."
    exit 1
  fi
}

# Login and save token
login() {
  echo "=== Login ==="
  read -p "Username: " USERNAME
  read -s -p "Password: " PASSWORD
  echo

  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$HOSTING_URL/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

  BODY=$(echo "$RESPONSE" | sed '$d')
  STATUS=$(echo "$RESPONSE" | tail -n1)

  if [ "$STATUS" -eq 200 ]; then
    TOKEN="$BODY"
    echo "Saving new token..."

    grep -q '^TOKEN=' "$ENV_FILE" && \
      sed -i "s|^TOKEN=.*|TOKEN=$TOKEN|" "$ENV_FILE" || \
      echo -e "\nTOKEN=$TOKEN" >> "$ENV_FILE"

    echo "Login successful!"
  else
    echo "Login failed. HTTP status: $STATUS"
    echo "Response: $BODY"
    exit 1
  fi
}

# Remove token from .env
logout() {
  if grep -q '^TOKEN=' "$ENV_FILE"; then
    sed -i '/^TOKEN=/d' "$ENV_FILE"
    sed -i '/^\s*$/d' "$ENV_FILE"
    echo "Logged out successfully. Token removed."
  else
    echo "You are not logged in."
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
  --help                    Show this help message
  --list <folder>           List files in a folder
  --upload <file> <path>    Upload a file
  --delete <path>           Delete a file
  --rename <old> <new>      Rename a file
  --get <path>              Download a file
  --search <query>          Search for files
  --logout                  Logout and remove saved token

Examples:
  $0 --list somefolder
  $0 --upload localfile.txt somefolder/newfile.txt
  $0 --delete somefolder/file.txt
  $0 --rename somefolder/file.txt somefolder/renamedfile.txt
  $0 --get somefolder/file.txt
  $0 --search image
  $0 --logout
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

case "$1" in
  --help)
    show_help
    exit 0
    ;;

  --logout)
    logout
    exit 0
    ;;
esac

load_env
load_token

case "$1" in
  --list)
    FOLDER="$2"
    RESPONSE=$(curl -s -X GET "$HOSTING_URL/resources/$FOLDER" \
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
    RESPONSE=$(curl --progress-bar -X POST "$HOSTING_URL/resources/$DEST" \
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
    RESPONSE=$(curl -s -X DELETE "$HOSTING_URL/resources/$PATH_TO_DELETE" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  --rename)
    OLD_PATH="$2"
    NEW_PATH="$3"
    RESPONSE=$(curl -s -X PATCH "$HOSTING_URL/resources/$OLD_PATH?action=rename&destination=%2F$NEW_PATH&override=false&rename=false" \
      -H "X-Auth: $TOKEN" -H "User-Agent: bash-script")

    check_auth "$RESPONSE" || exec "$0" "$@"
    echo "$RESPONSE"
    ;;

  --get)
    PATH_TO_GET="$2"
    FILENAME=$(basename "$PATH_TO_GET")

    echo "Downloading $PATH_TO_GET to ./$FILENAME ..."
    curl --progress-bar -X GET "$HOSTING_URL/raw/$PATH_TO_GET" \
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
    RESPONSE=$(curl -s -X GET "$HOSTING_URL/search/?query=$QUERY" \
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
