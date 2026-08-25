#!/bin/sh
set -e

APP_DIR="/opt/reinforcement-studio"
SANDBOX="$APP_DIR/chrome-sandbox"

if [ -f "$SANDBOX" ]; then
  chown root:root "$SANDBOX"
  chmod 4755 "$SANDBOX"
fi

exit 0
