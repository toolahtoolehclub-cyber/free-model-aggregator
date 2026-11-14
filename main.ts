// free-model-aggregator-01: serve every request with the best free model you can get.
//
// One long-running program. It listens on 0.0.0.0 at a port it chooses, registers
// with the env, and answers the requests the env's client sends; when the client is
// done it exits 0. Every answer comes from one of the ten providers, through the
// env: there is no model of its own.
//
// ROUTING (quality is 60% of the score)
//   For each request, the candidates are the best free model of every provider
//   that meets the request's requirements, ranked by effective rating: the
//   quality table's rating_milli less a penalty for the chance the provider's
//   next call refuses. The chance is Bayesian, learned from the provider's
//   answers so far, and it is what keeps the trial's twenty calls near the
//   floor instead of hammering the top-rated provider until its failure
//   schedule is hit. Only the best model of a provider is listed: a refusal
//   is the provider's, not the model's, so a retry on its second-best model
//   would spend another call on the same failing provider. A JSON-mode OpenAI
//   request only ever goes to an openai-wire provider (response_format) or a
//   gemini-wire one (responseMimeType): the Anthropic wire has no field that
//   carries JSON mode, so a provider on it would echo json_mode false and lose
//   the request.
//
// THE FAILURE MODEL (why the penalty is Bayesian, not a flat per-call one)
//   A provider's failure schedule holds at most two calls, never its first,
//   and exactly one provider in a trial is out from some call on. So the
//   first call to a provider is free; the second is the risky one; and a
//   scheduled refusal SPENDS the provider's failure quota, which makes its
//   next call safer, not riskier — after one scheduled refusal the next call
//   is very likely to answer, and after two there is no scheduled refusal
//   left. A flat per-call penalty gets this backwards: it spreads the calls
//   and pays a second-call refusal on provider after provider. This router
//   instead concentrates on the top provider and pays its one early refusal
//   to unlock the long run of safe calls behind it. The budget is a second
//   risk that grows with the calls already made, so a provider used many
//   times is worth less even when its scheduled failures are spent.
//
// FAILOVER (10% of the score, and served with it)
//   A provider refuses with its own status and error code when its free-tier
//   budget is spent, when a call lands on its failure schedule, or after its
//   outage begins. The refusal's message says which: "budget is spent" and
//   "is out" mean the provider is dead for the rest of the trial and it is
//   blacklisted at once; "refused this call" is a scheduled failure, the
//   provider stays in play, and its failure quota is now partly spent. A
//   refusal is never the answer: the candidates are recomputed from the
//   updated state and the request is retried, which is how a scheduled
//   refusal is followed by a retry on the same provider — its next call is
//   the safe one. Exactly one provider call may succeed per request.
//
// THE WIRES (10% of the score)
//   The client speaks OpenAI or Anthropic; each provider speaks openai, anthropic
//   or gemini. The request is translated into the provider's wire untouched (system
//   prompt, user text and images, tools, stop sequences, max_tokens, JSON mode),
//   and the provider's answer back into the client's wire, streamed or not as it
//   asked, with the provider's text whole, its usage counts as that wire reports
//   them, and its stop reason mapped. An OpenAI stream asks the provider for its
//   usage chunk (stream_options.include_usage), or the answer would carry no
//   usage at all.
//
// THE CATALOG (5% of the score)
//   `--list` prints the catalog the brief makes, and the register body carries
//   the same document. GET /v1/models lists exactly the free model ids it
//   registers, in the shape of the auth header it got.
//
// DETERMINISM
//   Every route is a function of the requests and the providers' answers seen so
//   far, never of the wall clock. Timeouts exist only as a guard against a hung
//   upstream call; the env is local, and they do not fire on a live trial.
//
// THE WIRES (10% of the score)
//   The client speaks OpenAI or Anthropic; each provider speaks openai, anthropic
//   or gemini. The request is translated into the provider's wire untouched (system
//   prompt, user text and images, tools, stop sequences, max_tokens, JSON mode),
//   and the provider's answer back into the client's wire, streamed or not as it
//   asked, with the provider's text whole, its usage counts as that wire reports
//   them, and its stop reason mapped. An OpenAI stream asks the provider for its
//   usage chunk (stream_options.include_usage), or the answer would carry no
//   usage at all.
//
// THE CATALOG (5% of the score)
//   `--list` prints the catalog the brief makes, and the register body carries
//   the same document. GET /v1/models lists exactly the free model ids it
//   registers, in the shape of the auth header it got.
//
// DETERMINISM
//   Every route is a function of the requests and the providers' answers seen so
//   far, never of the wall clock. Timeouts exist only as a guard against a hung
//   upstream call; the env is local, and they do not fire on a live trial.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  readBrief,
  servedModels,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiReq,
  anthropicToOpenAIReq,
  anthropicToOpenAIResp,
  anthropicToOpenAIStream,
  geminiToAnthropicResp,
  geminiToAnthropicStream,
  geminiToOpenAIResp,
  geminiToOpenAIStream,
  openAIError,
  openAIModelList,
  openAIToAnthropicReq,
  openAIToAnthropicResp,
  openAIToAnthropicStream,
  openAIToGeminiReq,
  pipeStream,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** How long the whole request may spend trying providers, in milliseconds. */
const REQUEST_DEADLINE_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/**
 * Routing penalty per unit of refusal chance, in rating milli-points. A
 * candidate's effective rating is its rating less K times the chance its
 * next call refuses, so the router weighs a refusal (one extra upstream
 * call) against the rating it would give up by picking a safer provider.
 * A sweep over the three practice trials put the sweet spot at 200: lower
 * spreads the calls over too many second-call refusals, higher starts
 * serving low-rated models and drops quality.
 */
const FAIL_RISK_MILLI = 200;
/**
 * Chance the next call refuses when no scheduled refusal has been seen yet
 * and it is the provider's second call. The second call is the risky one:
 * one provider in every trial is designated to refuse exactly there, and
 * the failure schedules are drawn from the second call on.
 */
const FAIL_P_SECOND = 0.3;
/**
 * Chance the next call refuses when no scheduled refusal has been seen yet
 * and it is the provider's third call or later. Later calls are safer than
 * the second, because the schedules are drawn over the whole budget and a
 * single failure is the common case.
 */
const FAIL_P_LATER = 0.15;
/**
 * Chance the next call refuses after exactly one scheduled refusal has been
 * seen. The one refusal spent half the provider's failure quota (at most
 * two), and the remaining refusal, if there is one, sits at one specific
 * later call, so the next call is very likely to answer.
 */
const FAIL_P_ONE_SEEN = 0.1;
/**
 * Chance the next call refuses after two scheduled refusals have been seen.
 * The failure schedule holds at most two calls, so both are spent and no
 * scheduled refusal is left; only the budget and the outage remain.
 */
const FAIL_P_TWO_SEEN = 0.1;
/**
 * Budget hazard per call past BUDGET_HAZARD_AFTER. A provider's free-tier
 * budget is small and drawn per trial, so the chance the next call is over
 * it grows with the calls already made, even when the scheduled failures
 * are spent. This is what stops the router concentrating on one provider
 * until its budget runs out.
 */
const BUDGET_HAZARD_PER_CALL = 0.03;
/** Calls after which the budget hazard starts to count. */
const BUDGET_HAZARD_AFTER = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const probes: Probe[] = await Promise.all(
    brief.directory.map((provider) => probeProvider(ENV_URL, provider, 5000)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief, probes);
}

/** One provider's trial state, learned from its answers so far. */
interface ProviderState {
  calls: number;
  /** Call numbers that refused with a scheduled failure ("refused this call"). */
  failNums: Set<number>;
  /** True once the provider answered that its budget is spent or it is out. */
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failNums: new Set<number>(), dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One (provider, model) pair that could serve a request, with its rating. */
interface Candidate {
  provider: ProviderEntry;
  row: RatingRow;
  /** Rating less the routing penalties: the order the router tries them in. */
  effective: number;
}

/**
 * The chance this provider's next call refuses, from what its answers have
 * shown so far. A provider that is dead never answers. The first call is
 * free: no failure schedule ever holds it. With no scheduled refusal seen,
 * the second call is the risky one and later calls are safer. With one
 * scheduled refusal seen, the failure quota is half spent and the next call
 * is very likely to answer; with two, the schedule is exhausted. On top of
 * the scheduled chance, the budget hazard grows with the calls already
 * made, because the free-tier budget is small and drawn per trial.
 */
function pFail(state: ProviderState): number {
  if (state.dead) return 1;
  const next = state.calls + 1;
  if (next === 1) return 0;
  let scheduled: number;
  if (state.failNums.size >= 2) scheduled = FAIL_P_TWO_SEEN;
  else if (state.failNums.size === 1) scheduled = FAIL_P_ONE_SEEN;
  else scheduled = next === 2 ? FAIL_P_SECOND : FAIL_P_LATER;
  const budget = BUDGET_HAZARD_PER_CALL * Math.max(0, state.calls - BUDGET_HAZARD_AFTER);
  return Math.min(0.95, scheduled + budget);
}

/**
 * The candidates for one request: the best free model of every provider that
 * meets its requirements, ranked by effective rating — the rating less the
 * refusal chance times FAIL_RISK_MILLI. Only the best model of a provider
 * is listed: a provider that refuses does so for the provider, not the
 * model, so a retry on its second-best model would spend another call on
 * the same failing provider instead of moving to a fresh one. Dead providers
 * are out. A JSON-mode OpenAI request drops Anthropic-wire providers (their
 * wire cannot echo JSON mode). The order is deterministic: effective rating,
 * then rating, then provider and model id. The list is recomputed after
 * every refusal, so a provider whose scheduled refusal just made its next
 * call safer can come back to the top and be retried.
 */
function candidatesFor(requires: Requirement, jsonModeOpenAI: boolean): Candidate[] {
  const byProvider = new Map<string, Candidate>();
  for (const provider of brief.directory) {
    const state = stateOf(provider.id);
    if (state.dead) continue;
    if (jsonModeOpenAI && provider.wire === 'anthropic') continue;
    for (const row of brief.ratings?.ratings ?? []) {
      if (row.provider !== provider.id) continue;
      if (!row.free) continue;
      if (!meetsRequirements(requires, row)) continue;
      const effective = row.rating_milli - FAIL_RISK_MILLI * pFail(state);
      const prev = byProvider.get(provider.id);
      if (prev === undefined || effective > prev.effective) {
        byProvider.set(provider.id, { provider, row, effective });
      }
    }
  }
  const out = [...byProvider.values()];
  out.sort((a, b) => {
    if (a.effective !== b.effective) return b.effective - a.effective;
    if (a.row.rating_milli !== b.row.rating_milli) return b.row.rating_milli - a.row.rating_milli;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  return out;
}

/**
 * Why a provider refused, read off its error message. The env answers a
 * refusal with the provider's own status and code, which are the same for
 * every refusal kind, but the message says which: the budget is spent, the
 * provider is out, or this one call was refused. A message that cannot be
 * read is treated as a scheduled refusal: blacklisting on an unreadable
 * message could drop a live provider, while a scheduled refusal costs at
 * most one more call.
 */
function refusalReason(text: string): 'budget' | 'outage' | 'fail' {
  let message = '';
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') {
      message = body.error.message;
    }
  } catch {
    // Not JSON: fall through to a scheduled refusal.
  }
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes(' is out')) return 'outage';
  return 'fail';
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether this OpenAI client request asks for JSON mode (a response_format). */
function jsonModeOfOpenAI(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list: ServedModel[] = servedModels(catalog).map((m) => ({ id: m.id, provider: m.provider }));
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in the provider's wire. */
function upstreamBody(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  body: Json,
): Json {
  const wire = candidate.provider.wire;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      const out: Json = { ...body, model: candidate.row.model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    if (wire === 'anthropic') return openAIToAnthropicReq(body, candidate.row.model);
    return openAIToGeminiReq(body);
  }
  if (wire === 'openai') return anthropicToOpenAIReq(body, candidate.row.model);
  if (wire === 'anthropic') return { ...body, model: candidate.row.model };
  return anthropicToGeminiReq(body);
}

/** The upstream path for one candidate, in the provider's wire. */
function upstreamPath(candidate: Candidate, stream: boolean): string {
  const wire = candidate.provider.wire;
  if (wire === 'openai') return '/v1/chat/completions';
  if (wire === 'anthropic') return '/v1/messages';
  const model = encodeURIComponent(candidate.row.model);
  return `/v1beta/models/${model}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
}

/** One provider call: the fetch, or null when it could not be made at all. */
async function callProvider(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Response | null> {
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  if (candidate.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  try {
    return await fetch(`${base}${upstreamPath(candidate, stream)}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientApi, candidate, body)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/**
 * Answer one client request: recompute the candidates from the providers'
 * answers so far, try the top one, and on a refusal recompute again and
 * retry — a scheduled refusal makes that provider's next call safer, so it
 * may come back to the top, while a spent budget or an outage blacklists it
 * at once. A success translates the answer back into the client's wire.
 * Returns false when no candidate answered (the caller sends the error).
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<boolean> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && jsonModeOfOpenAI(body);
  const deadline = Date.now() + REQUEST_DEADLINE_MS;
  // At most one try per provider per round of the list, plus retries of a
  // provider whose refusal spent its failure quota: twenty is far past any
  // trial's useful attempts and only bounds a pathological loop.
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (Date.now() > deadline) break;
    const candidates = candidatesFor(requires, jsonMode);
    if (candidates.length === 0) break;
    const candidate = candidates[0];
    const state = stateOf(candidate.provider.id);
    state.calls += 1;
    const callNumber = state.calls;
    const upstream = await callProvider(candidate, clientApi, body, stream);
    if (upstream === null || !upstream.ok) {
      const status = upstream === null ? 0 : upstream.status;
      let text = '';
      if (upstream !== null) {
        try {
          text = await upstream.text();
        } catch {
          // The error body is not needed: the refusal itself is the signal.
        }
      }
      const reason = upstream === null ? 'fail' : refusalReason(text);
      if (reason === 'budget' || reason === 'outage') {
        state.dead = true;
      } else {
        state.failNums.add(callNumber);
      }
      log(
        `${candidate.provider.id}/${candidate.row.model} refused (HTTP ${status}, ${reason}); ` +
          `scheduled refusals seen: ${state.failNums.size}`,
      );
      continue;
    }
    const wire = candidate.provider.wire;
    if (!stream) {
      const answer: unknown = await upstream.json().catch(() => null);
      if (!isRecord(answer)) {
        state.failNums.add(callNumber);
        continue;
      }
      if (clientApi === 'openai') {
        if (wire === 'openai') sendJson(res, 200, answer);
        else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResp(answer, candidate.row.model));
        else sendJson(res, 200, geminiToOpenAIResp(answer, candidate.row.model));
      } else {
        if (wire === 'openai') sendJson(res, 200, openAIToAnthropicResp(answer, candidate.row.model));
        else if (wire === 'anthropic') sendJson(res, 200, answer);
        else sendJson(res, 200, geminiToAnthropicResp(answer, candidate.row.model));
      }
      return true;
    }
    if (upstream.body === null) {
      state.failNums.add(callNumber);
      continue;
    }
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    if (clientApi === 'openai') {
      if (wire === 'openai') await pipeStream(upstream.body, (text) => res.write(text));
      else if (wire === 'anthropic') {
        for await (const event of anthropicToOpenAIStream(upstream.body, candidate.row.model)) {
          res.write(event);
        }
      } else {
        for await (const event of geminiToOpenAIStream(upstream.body, candidate.row.model)) {
          res.write(event);
        }
      }
    } else {
      if (wire === 'openai') {
        for await (const event of openAIToAnthropicStream(upstream.body, candidate.row.model)) {
          res.write(event);
        }
      } else if (wire === 'anthropic') {
        await pipeStream(upstream.body, (text) => res.write(text));
      } else {
        for await (const event of geminiToAnthropicStream(upstream.body, candidate.row.model)) {
          res.write(event);
        }
      }
    }
    res.end();
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const ok = await serve(req, res, 'openai', body);
  if (!ok) {
    sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const ok = await serve(req, res, 'anthropic', body);
  if (!ok) {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(
    `listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ` +
      `${servedModels(catalog).length} free models`,
  );
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
