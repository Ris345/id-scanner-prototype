FROM node:20-slim

WORKDIR /app

# Install with the lockfile first so this layer caches across server.js edits.
COPY package.json package-lock.json ./
RUN npm ci

COPY server.js .

EXPOSE 3001

CMD ["node", "server.js"]
