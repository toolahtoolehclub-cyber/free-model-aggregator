# free-model-aggregator

One server that serves every request with the best free model it can get.

It speaks the OpenAI Chat Completions API (`POST /v1/chat/completions`) and the
Anthropic Messages API (`POST /v1/messages`), JSON and SSE streaming, and routes
each request to the strongest free model that meets the request's requirements —
across ten providers on three wires (OpenAI, Anthropic, Gemini) — failing over
to the next best model when a provider refuses.

## Run

```sh
node main.ts --list        # print the catalog and exit
node main.ts               # serve; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
```

Node 22 standard library only; no dependencies, no model calls, no API keys.

## How it routes

For each request the candidates are every free model meeting its requirements,
ranked by the sealed quality table's rating, best first. A JSON-mode OpenAI
request only goes to an OpenAI-wire provider (`response_format`) or a
Gemini-wire one (`responseMimeType`), since the Anthropic wire cannot echo JSON
mode. A provider that refuses (budget spent, failure schedule, outage) is
retried on the next candidate; a provider that refuses twice running is
treated as spent for the rest of the trial. Exactly one provider call succeeds
per request.

## Layout

- `main.ts` — the server, the routing and failover policy, register and poll.
- `lib/brief.ts` — the trial brief (provider directory, quality table, workload)
  and the catalog it makes.
- `lib/wire.ts` — the client and provider wires and the translations between
  them, JSON and streaming.

Built for the make.fun bounty "The best free AI model aggregator"
(`free-model-aggregator-01`).
