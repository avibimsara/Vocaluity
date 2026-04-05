#!/bin/sh
set -e

# Inject NEXT_PUBLIC_API_URL at runtime if set
if [ -n "$NEXT_PUBLIC_API_URL" ]; then
  echo "[startup] Injecting NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}"
  # Replace any build-time placeholder with the runtime value in all JS files
  find /app/.next -name "*.js" -exec sed -i "s|NEXT_PUBLIC_API_URL_PLACEHOLDER|${NEXT_PUBLIC_API_URL}|g" {} + 2>/dev/null || true
fi

echo "[startup] Starting Next.js server on port ${PORT:-3000}..."
exec node server.js
