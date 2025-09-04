FROM python:3-slim

ARG HOST=0.0.0.0
ARG PORT=8003
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV PATH="/usr/local/bin:${PATH}"
# Update the base packages
RUN pip install --upgrade repository-manager

# set the entrypoint to the start.sh script
ENTRYPOINT exec repository-manager-mcp --transport=http --host=${HOST} --port=${PORT}
