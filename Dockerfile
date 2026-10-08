FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl ca-certificates && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

COPY ml/requirements-serve.txt ./requirements-serve.txt
RUN python -m pip install --no-cache-dir -r requirements-serve.txt

COPY package*.json ./
RUN npm ci --omit=dev

COPY src ./src
COPY public ./public
COPY ml ./ml

ENV NODE_ENV=production
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHON_PATH=python

EXPOSE 8080

CMD ["npx", "--yes", "tsx", "src/index.ts"]
