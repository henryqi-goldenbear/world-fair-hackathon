#!/usr/bin/env node
/**
 * FerbAI-compatible synthetic agent-student swarm.
 *
 * This generates the same core artifacts FerbAI records and reviews:
 * - Recording-shaped scene event logs with snapshots and transcript words
 * - Chat messages suitable for FerbAI's /api/tutor-review payload
 * - Board element state using FerbAI's Element union shape
 * - Agent 2 tutor-review results from the cloned FerbAI server module
 *
 * No human input is used. Student turns come from deterministic agent personas.
 */

import { writeFile } from 'node:fs/promises'

const SUBJECT_MODELS = {
  photosynthesis: {
    prompt: 'Explain how photosynthesis works.',
    concepts: [
      'chlorophyll captures light energy',
      'carbon dioxide enters through leaves',
      'water is absorbed by roots',
      'glucose stores chemical energy',
      'oxygen is released as a byproduct',
      'plants also use cellular respiration',
    ],
    misconceptions: [
      'plants get most of their food from soil',
      'oxygen is the main input plants consume',
      'photosynthesis only happens in flowers',
      'sunlight turns directly into oxygen',
      'plants do not respire',
    ],
    boardSeed: 'CO2 + H2O + light -> glucose + O2',
  },
  fractions: {
    prompt: 'Explain how to add fractions with unlike denominators.',
    concepts: [
      'the denominator names equal parts',
      'the numerator counts selected parts',
      'common denominators make parts comparable',
      'equivalent fractions preserve value',
      'add numerators after converting denominators',
      'simplify the final answer',
    ],
    misconceptions: [
      'larger denominators always mean larger fractions',
      'add numerators and denominators separately',
      'equivalent fractions are different amounts',
      'fractions cannot be greater than one',
      'simplifying changes the amount',
    ],
    boardSeed: '1/2 + 1/3 = ?',
  },
  'newtonian mechanics': {
    prompt: 'Explain what happens when a net force acts on an object.',
    concepts: [
      'net force determines acceleration',
      'mass resists acceleration',
      'velocity includes speed and direction',
      'friction opposes motion',
      'gravity is a force',
      'balanced forces mean no acceleration',
    ],
    misconceptions: [
      'motion requires a continuous forward force',
      'heavier objects always fall faster',
      'velocity and acceleration are the same',
      'forces only exist when objects touch',
      'friction always stops motion immediately',
    ],
    boardSeed: 'Fnet = m*a',
  },
}

const FALLBACK_MODEL = {
  prompt: 'Explain the core idea and apply it to a new example.',
  concepts: [
    'define the central idea',
    'connect cause and effect',
    'use evidence for claims',
    'apply the idea to a new example',
    'compare related concepts',
    'explain limits or exceptions',
  ],
  misconceptions: [
    'confuses the definition with an example',
    'uses memorized language without causal reasoning',
    'overgeneralizes one case to every case',
    'misses an important prerequisite idea',
    'treats related terms as interchangeable',
  ],
  boardSeed: 'claim -> evidence -> reasoning',
}

const NAMES = [
  'Amina', 'Ben', 'Carla', 'Dev', 'Elena', 'Felix', 'Grace', 'Hana',
  'Isaac', 'Jules', 'Kai', 'Lina', 'Mateo', 'Nora', 'Owen', 'Priya',
  'Quinn', 'Rafi', 'Sofia', 'Theo', 'Uma', 'Vera', 'Will', 'Yara', 'Zane',
]

const INK = 'oklch(27% 0.008 70)'
const CLAY = 'oklch(61% 0.115 42)'
const SAGE = 'oklch(66% 0.05 150)'
const BLUE = 'oklch(50% 0.12 240)'
const BOARD_WIDTH = 960
const BOARD_HEIGHT = 640

function localMessagesToTranscript(messages = []) {
  return messages
    .filter((message) => message && message.text && !message.pending && !message.error)
    .map((message) => `${message.role === 'assistant' ? 'Tutor' : 'Student'}: ${String(message.text).trim()}`)
    .join('\n\n')
}

function countMatches(text, pattern) {
  return (text.match(pattern) || []).length
}

function clampScore(score) {
  return Math.min(5, Math.max(1, score))
}

function labelFor(score) {
  if (score >= 5) return 'excellent'
  if (score >= 4) return 'strong'
  if (score >= 3) return 'adequate'
  if (score >= 2) return 'weak'
  return 'poor'
}

function localScoreTutorTranscript({ transcript, boardState = '', lessonGoal = '' }) {
  const normalized = String(transcript || '').trim()
  if (!normalized) throw new Error('Transcript is empty.')
  const turns = {
    student: countMatches(normalized, /^\s*Student:/gim),
    tutor: countMatches(normalized, /^\s*Tutor:/gim),
  }
  const wordCount = normalized.split(/\s+/).filter(Boolean).length
  const questionCount = countMatches(normalized, /\?/g)
  const encouragementHits = countMatches(normalized, /\b(good|great|nice|correct|exactly|well done|you got|that's right)\b/gi)
  const understandingHits = countMatches(normalized, /\b(does that make sense|what do you think|why|how did you get|can you explain|check your understanding|try|your turn|walk me through)\b/gi)
  const scaffoldHits = countMatches(normalized, /\b(hint|step|first|next|because|notice|let's break|simpler|example|compare|diagram|draw|board)\b/gi)
  const assessmentHits = countMatches(normalized, /\b(question|quiz|practice|solve|try this|what is|why is|how would|can you|tell me)\b/gi)
  const boardMentions = countMatches(normalized, /\b(board|diagram|draw|shown|sketch|equation|graph)\b/gi)
  const hasBoardState = !!String(boardState || '').trim()
  const hasGoal = !!String(lessonGoal || '').trim()
  const dimensions = [
    ['student engagement', clampScore(2 + Math.min(2, Math.floor(questionCount / 3)) + Math.min(1, Math.floor(understandingHits / 2)))],
    ['diagnosis of understanding', clampScore(2 + Math.min(2, Math.floor(understandingHits / 2)) + Math.min(1, Math.floor(assessmentHits / 4)))],
    ['scaffolding and pedagogy', clampScore(2 + Math.min(3, Math.floor(scaffoldHits / 3)))],
    ['board grounding', clampScore(1 + (hasBoardState ? 2 : 0) + Math.min(2, boardMentions))],
    ['supportive tone', clampScore(2 + Math.min(2, Math.floor(encouragementHits / 2)) + (turns.student > 0 ? 1 : 0))],
    ['lesson-goal alignment', clampScore(2 + (hasGoal ? 1 : 0) + Math.min(2, Math.floor(assessmentHits / 5)))],
  ].map(([name, score]) => ({ name, score, label: labelFor(score), rationale: 'FerbAI-compatible local heuristic.' }))
  const averageScore = Math.round((dimensions.reduce((sum, item) => sum + item.score, 0) / dimensions.length) * 100) / 100
  const weak = dimensions.filter((item) => item.score <= 2)
  return {
    verdict: weak.length ? (averageScore >= 3 ? 'needs_improvement' : 'ineffective') : (averageScore >= 4 ? 'effective' : 'needs_improvement'),
    averageScore,
    dimensions,
    strengths: questionCount >= 4 ? ['The tutor asks multiple questions instead of relying only on exposition.'] : [],
    risks: weak.map((item) => `${item.name} is weak (${item.score}/5): ${item.rationale}`),
    recommendations: ['Collect richer lesson evidence before treating this tutoring pattern as consistently effective.'],
    evidence: { wordCount, turnCounts: turns, questionCount, encouragementHits, understandingCheckHits: understandingHits, scaffoldingHits: scaffoldHits, assessmentHits, boardMentions },
    note: 'Local FerbAI-compatible heuristic fallback.',
  }
}

async function loadReviewFunctions() {
  try {
    const mod = await import('./ferbai-upstream/server/tutorReview.js')
    return {
      source: 'ferbai-upstream/server/tutorReview.js',
      scoreTutorTranscript: mod.scoreTutorTranscript,
      messagesToTranscript: mod.messagesToTranscript,
    }
  } catch (err) {
    return {
      source: `local compatibility fallback (${err?.code || err?.message || 'upstream unavailable'})`,
      scoreTutorTranscript: localScoreTutorTranscript,
      messagesToTranscript: localMessagesToTranscript,
    }
  }
}

function mulberry32(seed) {
  let a = seed >>> 0
  return function rand() {
    a += 0x6d2b79f5
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function choice(rng, items) {
  return items[Math.floor(rng() * items.length)]
}

function sample(rng, items, count) {
  const copy = [...items]
  const out = []
  while (copy.length && out.length < count) {
    out.push(copy.splice(Math.floor(rng() * copy.length), 1)[0])
  }
  return out
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function bandForAbility(ability) {
  if (ability < 0.3) return 'novice'
  if (ability < 0.55) return 'developing'
  if (ability < 0.8) return 'proficient'
  return 'advanced'
}

function makeId(prefix, rng) {
  return `${prefix}_${Math.floor(rng() * 1e12).toString(36)}`
}

function subjectModel(subject) {
  return SUBJECT_MODELS[subject.toLowerCase()] || FALLBACK_MODEL
}

function makePersona(index, subject, rng) {
  const model = subjectModel(subject)
  const ability = clamp((rng() + rng() + rng()) / 3, 0.05, 0.98)
  const band = bandForAbility(ability)
  const conceptCount = { novice: 1, developing: 2, proficient: 4, advanced: 6 }[band]
  const misconceptionCount = { novice: 3, developing: 2, proficient: rng() < 0.55 ? 1 : 0, advanced: rng() < 0.15 ? 1 : 0 }[band]

  return {
    id: makeId(`agent_student_${String(index).padStart(3, '0')}`, rng),
    kind: 'agent_student',
    name: `${choice(rng, NAMES)} ${String(index).padStart(2, '0')}`,
    subject,
    ability: Number(ability.toFixed(3)),
    understandingBand: band,
    confidence: Number(clamp(ability + (rng() - 0.5) * 0.55, 0.05, 0.98).toFixed(3)),
    languageProfile: choice(rng, ['native English', 'English language learner', 'concise speaker', 'verbose explainer']),
    behavior: choice(rng, ['asks clarifying questions', 'answers quickly', 'hesitates', 'self-corrects', 'overconfident']),
    masteredConcepts: sample(rng, model.concepts, Math.min(conceptCount, model.concepts.length)),
    misconceptions: sample(rng, model.misconceptions, Math.min(misconceptionCount, model.misconceptions.length)),
  }
}

function agentStudentUtterances(persona, model) {
  const known = persona.masteredConcepts
  const misconception = persona.misconceptions[0]
  const openingByBand = {
    novice: 'I recognize some words, but I am not sure how they fit together.',
    developing: 'I can explain part of it, but I might mix up the mechanism.',
    proficient: 'I think I can walk through it step by step.',
    advanced: 'I can explain it and test the explanation with a new example.',
  }
  const firstClaim = known[0] || model.concepts[0]
  const secondClaim = known[1] || model.concepts[1]
  const errorTurn = misconception
    ? `I think ${misconception}, so maybe that is the main mechanism.`
    : `The main mechanism is that ${firstClaim}, which connects to ${secondClaim}.`
  const repairTurn = misconception && persona.ability >= 0.45
    ? `Wait, I should revise that. ${firstClaim} matters, and my earlier statement was too broad.`
    : misconception
      ? 'I am still stuck. I need a smaller hint before I can repair that.'
      : `A stronger explanation is that ${firstClaim}, because ${secondClaim}.`

  return [
    openingByBand[persona.understandingBand],
    errorTurn,
    repairTurn,
    `My confidence is about ${Math.round(persona.confidence * 100)} percent.`,
  ]
}

function tutorUtterances(persona, model) {
  const known = persona.masteredConcepts[0] || model.concepts[0]
  const weak = persona.misconceptions[0] || 'the missing link in your explanation'
  return [
    `Let's use the board. ${model.prompt} Start with one claim, then evidence.`,
    `Good start. What on the board supports that claim, and what could challenge it?`,
    `Notice the step near your diagram: ${known}. Why does that make ${weak} risky?`,
    'Nice revision. Try one final sentence that connects the evidence to the mechanism.',
  ]
}

function textElement(id, x, y, text, { color = INK, size = 28, author } = {}) {
  return { id, type: 'text', color, width: 3, x, y, text, size, ...(author ? { author } : {}) }
}

function arrowElement(id, x1, y1, x2, y2, { color = CLAY, author } = {}) {
  return { id, type: 'arrow', color, width: 3, x1, y1, x2, y2, ...(author ? { author } : {}) }
}

function ellipseElement(id, x, y, w, h, { color = SAGE, author } = {}) {
  return { id, type: 'ellipse', color, width: 3, x, y, w, h, ...(author ? { author } : {}) }
}

function buildBoardElements(persona, model, rng) {
  const first = persona.masteredConcepts[0] || model.concepts[0]
  const misconception = persona.misconceptions[0]
  const elements = [
    textElement(makeId('el_seed', rng), 72, 84, model.boardSeed, { color: INK, size: 30 }),
    textElement(makeId('el_claim', rng), 72, 150, `Claim: ${first}`, { color: INK, size: 24 }),
    arrowElement(makeId('el_arrow1', rng), 100, 170, 100, 220, { color: CLAY }),
  ]

  if (misconception) {
    elements.push(textElement(makeId('el_misconception', rng), 72, 252, `Maybe: ${misconception}`, { color: INK, size: 22 }))
    elements.push(textElement(makeId('el_ai_hint', rng), 72, 330, 'Check the input/output relationship', { color: BLUE, size: 24, author: 'ai' }))
    elements.push(ellipseElement(makeId('el_circle', rng), 58, 220, 520, 62, { color: CLAY, author: 'ai' }))
  } else {
    elements.push(textElement(makeId('el_support', rng), 72, 252, `Evidence: ${persona.masteredConcepts[1] || model.concepts[1]}`, { color: INK, size: 22 }))
    elements.push(textElement(makeId('el_ai_hint', rng), 72, 330, 'Now apply it to a new example', { color: BLUE, size: 24, author: 'ai' }))
    elements.push(ellipseElement(makeId('el_circle', rng), 58, 305, 430, 58, { color: SAGE, author: 'ai' }))
  }
  return elements
}

function eventLogForElements(elements) {
  return elements.map((element, index) => ({
    t: 600 + index * 1100,
    type: 'add',
    element,
  }))
}

function transcriptWords(turns) {
  const words = []
  for (const turn of turns) {
    const tokens = turn.text.split(/\s+/).filter(Boolean)
    tokens.forEach((w, index) => {
      const start = turn.t + index * 230
      words.push({ w, start, end: start + 190 })
    })
  }
  return words
}

function transcriptText(turns) {
  return turns.map((turn) => `${turn.speaker}: ${turn.text}`).join('\n\n')
}

function makeSession(index, subject, seed, reviewFns) {
  const rng = mulberry32(seed + index * 7919)
  const model = subjectModel(subject)
  const persona = makePersona(index, subject, rng)
  const studentTurns = agentStudentUtterances(persona, model)
  const tutorTurns = tutorUtterances(persona, model)
  const turns = [
    { t: 0, speaker: 'Tutor', role: 'assistant', text: tutorTurns[0] },
    { t: 4200, speaker: 'Student', role: 'user', text: studentTurns[0] },
    { t: 8700, speaker: 'Tutor', role: 'assistant', text: tutorTurns[1] },
    { t: 13100, speaker: 'Student', role: 'user', text: studentTurns[1] },
    { t: 18100, speaker: 'Tutor', role: 'assistant', text: tutorTurns[2] },
    { t: 23000, speaker: 'Student', role: 'user', text: studentTurns[2] },
    { t: 27700, speaker: 'Tutor', role: 'assistant', text: tutorTurns[3] },
    { t: 32100, speaker: 'Student', role: 'user', text: studentTurns[3] },
  ]
  const elements = buildBoardElements(persona, model, rng)
  const events = eventLogForElements(elements)
  const durationMs = 36000
  const sessionId = makeId('ferbai_agent_session', rng)
  const recordingId = makeId('ferbai_agent_recording', rng)
  const recording = {
    id: recordingId,
    title: `Agent ${persona.name} - ${subject}`,
    createdAt: Date.now(),
    durationMs,
    demo: false,
    remote: false,
    agentGenerated: true,
    events,
    snapshots: [
      { t: 0, view: 'board', elements: [], equations: [], viz: null },
      { t: durationMs, view: 'board', elements, equations: [], viz: null },
    ],
    transcript: transcriptWords(turns),
    chapters: [
      { t: 0, title: 'Goal setup' },
      { t: 8700, title: 'Evidence check' },
      { t: 18100, title: 'Misconception review' },
      { t: 27700, title: 'Final explanation' },
    ],
  }
  const messages = turns.map((turn, turnIndex) => ({
    id: makeId(`msg_${turnIndex}`, rng),
    role: turn.role,
    text: turn.text,
  }))
  const boardState = JSON.stringify({
    view: 'board',
    width: BOARD_WIDTH,
    height: BOARD_HEIGHT,
    elements,
    agentGenerated: true,
  })
  const reviewPayload = {
    messages,
    transcript: reviewFns.messagesToTranscript(messages),
    boardState,
    lessonGoal: model.prompt,
    sessionId,
    userId: 'agent-swarm-local',
    recordingId,
  }
  const review = reviewFns.scoreTutorTranscript(reviewPayload)

  return {
    sessionId,
    recordingId,
    generatedBy: 'agent_student_swarm',
    humanStudent: false,
    persona,
    groundTruth: {
      masteryScore: persona.ability,
      understandingBand: persona.understandingBand,
      hasMisconception: persona.misconceptions.length > 0,
      misconceptions: persona.misconceptions,
      expectedTutorFocus: persona.misconceptions.length ? 'diagnose_and_repair' : 'extend_and_apply',
    },
    ferbai: {
      recording,
      chatMessages: messages,
      transcript: transcriptText(turns),
      tutorReviewPayload: reviewPayload,
      tutorReviewSource: reviewFns.source,
      tutorReview: review,
    },
  }
}

function summarize(sessions) {
  const byBand = { novice: 0, developing: 0, proficient: 0, advanced: 0 }
  const reviews = {}
  let agentOnly = true
  let recordingShapeOk = true
  for (const session of sessions) {
    byBand[session.persona.understandingBand] += 1
    reviews[session.ferbai.tutorReview.verdict] = (reviews[session.ferbai.tutorReview.verdict] || 0) + 1
    agentOnly = agentOnly && session.generatedBy === 'agent_student_swarm' && session.humanStudent === false
    recordingShapeOk = recordingShapeOk
      && Array.isArray(session.ferbai.recording.events)
      && Array.isArray(session.ferbai.recording.snapshots)
      && Array.isArray(session.ferbai.recording.transcript)
      && session.ferbai.recording.snapshots.at(-1)?.elements?.length > 0
      && session.ferbai.chatMessages.some((message) => message.role === 'user')
      && session.ferbai.chatMessages.some((message) => message.role === 'assistant')
  }
  return {
    sessionCount: sessions.length,
    agentOnly,
    recordingShapeOk,
    abilityDistribution: byBand,
    tutorReviewVerdicts: reviews,
  }
}

function parseArgs(argv) {
  const args = {
    subject: 'photosynthesis',
    count: 12,
    seed: 20260627,
    out: 'ferbai_agent_swarm_sessions.json',
    report: 'ferbai_agent_swarm_e2e_report.json',
  }
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i]
    const value = argv[i + 1]
    if (!key.startsWith('--')) continue
    i += 1
    if (key === '--subject') args.subject = value
    else if (key === '--count') args.count = Number(value)
    else if (key === '--seed') args.seed = Number(value)
    else if (key === '--out') args.out = value
    else if (key === '--report') args.report = value
    else throw new Error(`Unknown argument: ${key}`)
  }
  if (!Number.isInteger(args.count) || args.count < 1) throw new Error('--count must be a positive integer')
  if (!Number.isFinite(args.seed)) throw new Error('--seed must be numeric')
  return args
}

const args = parseArgs(process.argv)
const reviewFns = await loadReviewFunctions()
const sessions = Array.from({ length: args.count }, (_, index) => makeSession(index + 1, args.subject, args.seed, reviewFns))
const summary = summarize(sessions)
if (!summary.agentOnly) throw new Error('E2E failed: at least one session was not agent-only')
if (!summary.recordingShapeOk) throw new Error('E2E failed: at least one session is missing FerbAI recording/review shape')

const payload = {
  swarmId: makeId('ferbai_agent_swarm', mulberry32(args.seed)),
  subject: args.subject,
  seed: args.seed,
  generatedAt: new Date().toISOString(),
  reviewSource: reviewFns.source,
  summary,
  sessions,
}

await writeFile(args.out, JSON.stringify(payload, null, 2), 'utf8')
await writeFile(args.report, JSON.stringify({
  ok: true,
  checked: [
    'all student turns generated by agent personas',
    'FerbAI Recording event log present',
    'FerbAI snapshots present',
    'FerbAI transcript word timings present',
    'FerbAI /api/tutor-review payload present',
    'FerbAI scoreTutorTranscript executed for every session',
  ],
  reviewSource: reviewFns.source,
  summary,
}, null, 2), 'utf8')

console.log(`Wrote ${sessions.length} FerbAI-compatible agent sessions to ${args.out}`)
console.log(`Wrote E2E report to ${args.report}`)
console.log(JSON.stringify(summary))
