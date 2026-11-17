# free-model-aggregator-live

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/putrisilsi213-boop/free-model-aggregator-live/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/putrisilsi213-boop/free-model-aggregator-live/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two APIs, fourteen live free models. A free AI model aggregator that serves OpenAI Chat Completions and Anthropic Messages (JSON and streaming) by dialling a real free OpenRouter model for every prompt, through the grader's gateway on the key you push. Each request is routed to the best-rated free model that meets its requirements, with instant failover when a model refuses.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the provider catalog and exits.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`) and registers one catalog with the env: exactly the fourteen free OpenRouter models the brief lists.
- Ranks every free model by the brief's quality rating, best first, and retries the next-best candidate on any 4xx/5xx, timeout or unreachable gateway call — exactly one successful model call answers each request.
- A model that refuses is cooled down for the rest of the trial, so one rate-limited model cannot take later prompts down with it. Every refusal is reported in the `x-makefun-upstream-failures` header.
- Translates both wires: OpenAI and Anthropic clients, carrying the system prompt, user text and images, tools, stop sequences, max_tokens and JSON mode through, and the provider's text, usage and stop reason back, in JSON and in server-sent events.
- Deterministic: no clock and no randomness; every choice is a function of the brief, the requests and the gateway answers seen so far, so a replay leaves the same state export.

---

<sub>Entered on [make.fun](https://make.fun/?ref=d6m6thj3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
