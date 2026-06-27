# FerbAI Session Outputs Proof

This workspace now demonstrates the architecture node:

> FerbAI Session Outputs - raw multi-modal data from tutoring sessions, external POSTs to FastAPI.

## Files

- `ferbai_session_outputs.py` converts FerbAI agent-session recordings into the external payload schema.
- `fastapi_session_receiver.py` receives and validates `POST /sessions`.
- `ferbai_session_outputs.json` contains exported payloads.
- `received_ferbai_sessions.jsonl` contains payloads accepted by FastAPI.

## Payload Schema

```json
{
  "session_id": "ferbai_agent_session_54psmk85",
  "timestamp": "2026-06-27T21:39:02.924697Z",
  "transcript": [{ "t": 0, "speaker": "Tutor", "text": "..." }],
  "events": [{ "type": "click", "t": 1200, "payload": {} }],
  "drawings": ["<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>"],
  "metadata": {
    "duration_ms": 36000,
    "subject": "photosynthesis",
    "student_id": "agent_student_001_byemdl0x",
    "student_kind": "agent_student",
    "agent_generated": true,
    "human_student": false
  }
}
```

## Verified Result

The local FastAPI receiver accepted three FerbAI agent-student sessions:

```text
ferbai_agent_session_54psmk85: transcript=8, events=9, drawings=1
ferbai_agent_session_6gmvmixc: transcript=8, events=9, drawings=1
ferbai_agent_session_cqmgoj6f: transcript=8, events=9, drawings=1
```

All received payloads have `student_kind: "agent_student"` and `human_student: false`.

## Run It

```powershell
$env:PYTHONPATH = "$PWD\.deps"
python -m uvicorn fastapi_session_receiver:app --host 127.0.0.1 --port 8899
```

In another shell:

```powershell
python ferbai_session_outputs.py --input ferbai_agent_swarm_sessions.json --out ferbai_session_outputs.json --limit 3 --post-url http://127.0.0.1:8899/sessions
```
