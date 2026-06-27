FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements-fastapi.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_session_receiver.py ./
COPY ferbai_session_outputs.py ./
COPY ferbai_agent_swarm_sessions.json ./
COPY ferbai_session_outputs.json ./

EXPOSE 8080

CMD ["sh", "-c", "uvicorn fastapi_session_receiver:app --host 0.0.0.0 --port ${PORT}"]
