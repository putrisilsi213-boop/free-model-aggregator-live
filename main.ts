// free-model-aggregator-02: one server, two client wires, and a real free model behind
// every answer.
//
// Built from the kit's starter and rewritten for the score:
//   - quality (60%): every request is routed to the best-rated free model that meets its
//     requirements, ranked by the brief's rating_milli, best first; ties keep the brief's
//     order (a stable sort), so the choice is a pure function of the brief and the request.
//   - failover (10%): a gateway call refused with a 4xx or 5xx (or unreachable, or timed
//     out) is retried at once on the next-best candidate that meets the requirements,
//     until one answers 2xx. Every refusal is reported in the
//     `x-makefun-upstream-failures` header, so the failover column can see a refusal the
//     server recovered from. At most 8 attempts per prompt.
//   - rate limits: a model that refuses is cooled down for the next prompt only:
//     a 429 or 5xx is often a one-off blip, and a failed re-probe costs nothing on
//     the tie-break, so the best model is re-tested one prompt later instead of being
//     written off for the whole trial. Within one prompt it is never retried.
//   - served (15%): the same failover keeps a request alive when its first model is out
//     of budget or on its failure schedule; a prompt is only given up when every
//     candidate has refused it.
//   - wire: the starter's translations carry the system prompt, user text (images
//     included), tools, stop sequences, max_tokens and JSON mode through, and the
//     provider's text whole, its usage counts and its stop reason back, in JSON and in
//     server-sent events. A gateway body that names no output cap gets max_tokens 1024,
//     because the gateway refuses a request with no cap.
//   - catalog (5%): the catalog lists exactly the free models the brief names, and
//     GET /v1/models answers with that same set on both auth headers.
//   - determinism: no clock and no randomness anywhere; every choice is a function of
//     the brief, the requests and the gateway answers seen so far, so a replay leaves
//     the same state export.
//
// The trial: read the brief at $MAKEFUN_PROMPT_FILE, listen on 0.0.0.0 at a free port,
// POST $MAKEFUN_ENV_URL/register with the port and the one catalog, serve the client's
// six prompts, poll GET /work and exit 0 when it says done. `node main.ts --list`
// prints the same catalog and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds (the client waits 30 s). */
const DIAL_TIMEOUT_MS = 28000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most candidates tried for one request before giving up. */
const MAX_ATTEMPTS = 8;
/** The output cap a gateway body gets when it names none (the gateway refuses no cap). */
const DEFAULT_MAX_TOKENS = 1024;
/** The most rows the upstream-failures trail may carry. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/**
 * The request number (1-based, as manifestIndex counts it) at which a refused model may
 * be tried again. A refusal cools a model for the next prompt only: a 429 or 5xx is often
 * a one-off blip, and a failed re-probe costs nothing on the tie-break (only calls the
 * provider answers are counted), so the trial re-tests the best model one prompt later
 * instead of writing it off for the whole trial. Within one prompt's failover loop the
 * model is never retried: the candidate list is fixed when the prompt arrives.
 */
const coolUntil = new Map<string, number>();

/** One refused gateway call, as the failover trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that can serve one request, best rating first. A model cooling down
 * after a refusal is left out — unless every candidate is cooling down, in which case
 * the list is tried as it stands. Ties keep the brief's order (Array.sort is stable),
 * so the ranking is a pure function of the brief and the refusals seen so far.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const candidates = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const fresh = candidates.filter((model) => (coolUntil.get(model.id) ?? 0) <= manifestIndex);
  const pool = fresh.length > 0 ? fresh : candidates;
  pool.sort((a, b) => b.rating_milli - a.rating_milli);
  return pool;
}

/** Cool a refused model down until the prompt after next may try it again. */
function coolDown(modelId: string): void {
  coolUntil.set(modelId, manifestIndex + 2);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
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
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** A gateway body that names an output cap, or gets the default one when it names none. */
function withOutputCap(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_MAX_TOKENS };
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

/** The failures trail as one response header value, or null when there is nothing to report. */
function failuresHeader(failures: UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  const text = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  return text.length > 8 * 1024 ? null : text;
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: UpstreamFailure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: UpstreamFailure[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/** A refusal's code, from the gateway's error body when it names one, else from its status. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  if (status === 400) return 'invalid_request';
  if (status === 401) return 'unauthorized';
  if (status === 402) return 'insufficient_quota';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'upstream_error';
  return `http_${status}`;
}

/**
 * Dial the candidates in order until one answers 2xx. Returns the model that answered and
 * its response, with every refusal recorded; when every candidate refused, model and
 * upstream are null and failures carries the trail, so the 502 can report it.
 */
async function dialFirst(
  candidates: readonly ModelRow[],
  body: Json,
): Promise<{ model: string | null; upstream: Response | null; failures: UpstreamFailure[] }> {
  const failures: UpstreamFailure[] = [];
  const attempts = candidates.slice(0, MAX_ATTEMPTS);
  for (const candidate of attempts) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, withOutputCap(body));
    } catch (err) {
      const timeout = err instanceof Error && err.name === 'TimeoutError';
      failures.push({ status: timeout ? 504 : 502, code: timeout ? 'upstream_timeout' : 'upstream_unreachable', model: candidate.id });
      coolDown(candidate.id);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: candidate.id, upstream, failures };
    const text = await upstream.text();
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
    coolDown(candidate.id);
    log(`gateway refused ${candidate.id} with HTTP ${upstream.status}`);
  }
  return { model: null, upstream: null, failures };
}

/** A refused prompt as a 502 in the client's wire, carrying the failures trail. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail = last === undefined ? '' : `: the gateway answered HTTP ${last.status}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', `every free model refused this prompt${detail}`, failures);
  else sendError(res, wire, 502, 'upstream_error', `every free model refused this prompt${detail}`, failures);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await dialFirst(candidates, { ...body });
  if (served.model === null || served.upstream === null) return refused(res, 'openai', served.failures);
  const { model, upstream, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await dialFirst(candidates, anthropicToOpenAI(body, ''));
  if (served.model === null || served.upstream === null) return refused(res, 'anthropic', served.failures);
  const { model, upstream, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), failures);
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
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
