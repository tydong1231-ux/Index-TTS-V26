#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/app"
exec python -m audiobook_server.main
