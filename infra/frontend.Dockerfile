# Stage 1: Install dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY client/package.json client/package-lock.json* ./
RUN npm ci

# Stage 2: Build the application
FROM node:20-alpine AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY --from=deps /app/node_modules ./node_modules
COPY client/ ./
RUN npm run build

# Stage 3: Production runner
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy standalone build output
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Copy startup script (strip Windows CRLF line endings)
COPY infra/start_frontend.sh /app/start_frontend.sh
RUN sed -i 's/\r$//' /app/start_frontend.sh && chmod +x /app/start_frontend.sh

USER nextjs

ENV PORT=3000
EXPOSE 3000

CMD ["/app/start_frontend.sh"]
