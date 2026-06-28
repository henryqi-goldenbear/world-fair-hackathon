FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements-fastapi.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_session_receiver.py ./
COPY atlas_integration.py ./
COPY feature_extraction.py ./
COPY llm_evaluators.py ./
COPY swarm_sessions.py ./
COPY ferbai_session_outputs.py ./
COPY ferbai_session_outputs.json ./
COPY synthetic_student_swarm.json ./

EXPOSE 8080

CMD ["sh", "-c", "uvicorn fastapi_session_receiver:app --host 0.0.0.0 --port ${PORT}"]
