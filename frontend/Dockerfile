FROM node:22.21.1-bookworm-slim

WORKDIR /app
ENV npm_config_registry=https://registry.npmjs.org/ \
    npm_config_audit=false \
    npm_config_fund=false \
    npm_config_update_notifier=false
COPY frontend/app/package.json ./
RUN npm install --no-audit --no-fund --loglevel=warn
COPY frontend/app ./
COPY data ./public/data
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
