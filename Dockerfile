FROM node:22-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates curl \
  && corepack enable \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nemoclawd

COPY package.json ./package.json
COPY tsconfig.json ./tsconfig.json
COPY README.md ./README.md
COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY scripts ./scripts
COPY python_src ./python_src
COPY workflows ./workflows
COPY apps ./apps

RUN pnpm install
RUN pnpm run build:all

ENV NEMOCLAWD_ROOT_DIR=/opt/nemoclawd
ENV NEMOCLAWD_NAT_PYTHON=/opt/nemoclawd/apps/NeMo-Agent-Toolkit-develop/.venv/bin/python
ENV NEMOCLAWD_NAT_WORKDIR=/opt/nemoclawd/apps/NeMo-Agent-Toolkit-develop

CMD ["node", "apps/clawdbot-main/dist/index.js"]
