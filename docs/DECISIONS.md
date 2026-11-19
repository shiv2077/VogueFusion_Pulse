# Architecture Decision Records

Each record says what was decided, what was rejected, and what it costs.
Where a record cites a test or a measurement, that is the evidence for the
claim; numbers come from runs on this machine (RTX 3060 Laptop, 6 GB VRAM,
22 GB RAM) unless they are labelled as assumptions; each assumption is
listed under Open measurements in `docs/ARCHITECTURE.md`, which also
describes the system these decisions add up to.

Format: Context / Decision / Alternatives rejected / Consequences / Revisit when.

---

## ADR-001: ComfyUI as the generation backend, driven through its API

**Context.** Generation is SDXL (DreamShaper XL Lightning) conditioned on up
to four reference images through IP-Adapter Plus, optionally with the
project's LoRA. The pipeline was designed and iterated as a ComfyUI node
graph. The graph itself was lost from the repo but survived inside the PNG
metadata of the nine committed outputs; `scripts/recover_workflows.py`
extracts it.

**Decision.** Keep ComfyUI. The GPU worker queues a saved API-format graph
(`comfyui/workflows/v1.api.json`, `v2.api.json`) over HTTP and follows it
over the websocket. Request values are written into the parsed graph by
role (the KSampler and whatever feeds its positive and negative inputs),
never by string templating (`src/voguefusion/workflow.py`). ComfyUI listens
on 127.0.0.1 only and is never reachable by clients.

**Alternatives rejected.**
- Re-implement the graph with `diffusers`. Two representations of one
  pipeline to keep in sync, and IP-Adapter embedding combination would have
  to be rebuilt by hand.
- A hosted generation API. No control over the LoRA; cost moves from
  amortised GPU to per call.

**Consequences.** ComfyUI's behaviour is now part of the contract, so its
version and the IP-Adapter node pack are pinned (ComfyUI v0.45.0
`989d93d0`, ComfyUI_IPAdapter_plus `a0f451a`). Outputs are not bit-identical
to the 2024 originals: `scripts/replay.py` re-runs three committed graphs
with their exact seeds and the results show the same design with shifted
motif placement (mean per-pixel ΔE2000 13.8 to 14.7 against the originals,
versus 19.6 to 34.4 against the other committed outputs). On this machine
the replay is deterministic: re-running `image_9` after restarting ComfyUI
gave the identical score (14.72). ComfyUI also
caches identical graphs in memory (a repeated replay returned in 0.29 s), so
every measurement uses a fresh seed. Prompt text cannot alter the graph
(`tests/services/test_workflow.py::test_hostile_prompt_text_cannot_change_graph_structure`).

**Revisit when** the graph stops changing and a `diffusers` pipeline would
be less code, or when several tenants need isolation that one shared
ComfyUI process cannot give.

---

## ADR-002: Asynchronous jobs with a durable queue in front of the GPU

**Context.** One generation takes 10 to 41 s on this GPU (see
`docs/ARCHITECTURE.md`, Capacity). The GPU serialises work: a second
concurrent generation only competes for the same 6 GB of VRAM, which a
single generation already fills to about 5.5 GB.

**Decision.** `POST /v1/designs` validates, writes a job and returns 202
with a job id and a `Location` to poll. One GPU worker per GPU claims jobs
from the queue. Job states: `queued`, `running`, `succeeded`, `failed`
(permanent error, not retried) and `dead` (retries exhausted). A claim takes
a lease; the worker renews it on every progress event and at least every
`queue.heartbeat_s`. A job whose lease lapses goes back to the queue, or to
`dead` when out of attempts. Every worker write is conditional on its lease
owner and attempt number, so a worker that lost its lease cannot overwrite
the job's new owner (fencing).

When queued plus running jobs of a kind reach `max_depth`, submit returns
429 with `Retry-After = ceil(depth * p50 service time / workers)`, using the
measured service time once jobs have finished.

**Alternatives rejected.**
- Hold the HTTP request open for the generation. Tens of seconds per
  request, does not survive a client disconnect, and gives no backpressure.
- Send work straight to ComfyUI's own queue. ComfyUI's queue is in memory,
  has no depth limit, no retry policy, and is lost on restart.

**Consequences.** Every client interaction is two calls (submit, then
poll). Backlog has a hard ceiling: with `max_depth` 8 and about 12.5 s of
sustained service time for v1, an admitted job waits at most about 88 s
before it starts (the load test's p95 wait under overload was 89 s). Tests:
`tests/services/test_stores.py` (leases, fencing, retry, backoff, depth),
`tests/services/test_api_and_workers.py::test_worker_killed_mid_job_recovers`
(SIGKILL a real worker process mid-job; the job finishes on attempt 2) and
`::test_backpressure_returns_429_with_retry_after`.

**Revisit when** there are several GPUs and LoRA swaps become expensive
enough to route jobs to workers that already have the right LoRA loaded.

---

## ADR-003: Content-addressed blob storage

**Context.** Reference uploads, generated images and their downstream
consumers (the cost worker, the UI) need a shared place to read and write
images that survives restarts.

**Decision.** Blobs are stored under their SHA-256:
`$VOGUEFUSION_DATA/blobs/<aa>/<sha256>`. A write goes to a temporary file in
the same directory, is fsynced, renamed into place, and the directory is
fsynced. Reads can verify the hash. Uploaded references are decoded and
re-encoded as plain RGB PNG before storing, so the stored bytes are exactly
the pixels that were validated.

**Alternatives rejected.**
- Images in the database: large immutable bytes gain nothing from
  transactions and would bloat the job database.
- Paths named by job id: no deduplication, and no built-in integrity check.

**Consequences.** Identical uploads and identical outputs are stored once.
`GET /v1/artifacts/{sha}` can be cached forever (`immutable`, ETag = hash).
A crash mid-write cannot leave a partial file under a valid name
(`test_stores.py::test_interrupted_blob_write_leaves_nothing_at_final_address`).
Nothing is ever deleted: there is no garbage collection, which would need a
reachability pass over job results.

**Revisit when** disk use matters (add reachability GC), or when a second
machine needs the blobs (same key layout on an object store).

---

## ADR-004: CIEDE2000 for colour matching, and clustering in Lab

**Context.** The original solver compared colours by Euclidean distance in
raw sRGB, which is not perceptually uniform: an identical 0.05 sRGB step
spans ΔE2000 0.32 to 8.13 depending on where it starts
(`tests/color_science/test_delta_e2000.py`).

**Decision.** The solver's tolerance is CIEDE2000 with the CIE default
weights `k_L = k_C = k_H = 1` (`src/costing/color/perceptual.py`).
Dominant-colour extraction also clusters in CIE Lab instead of sRGB, so
cluster boundaries follow perceived difference (`src/costing/extract.py`).
The default tolerance is ΔE 2.0 (`costing.delta_e_tolerance`).

**Alternatives rejected.**
- CIE76: better than sRGB, but over-weights differences in high-chroma
  regions, where textile prints live.
- `k_L = 2` (common in textiles): kept CIE defaults so the implementation
  reproduces the Sharma, Wu and Dalal (2005) test vectors exactly.
- Metric trees for nearest-dye search: CIEDE2000 violates the triangle
  inequality, so tree pruning would be unsound. With 20 dyes, nothing needs
  indexing (ADR-008 enumerates).

**Consequences.** Matching costs a vectorised ΔE2000 evaluation per
candidate, which is negligible at this catalogue size.

**Revisit when** QC pass/fail data exists to tune the weights.

---

## ADR-005: Three-channel Kubelka-Munk mixing, substrate as an input

**Context.** The original solver predicted a mixture as the weighted
average of dye sRGB values. That models light on a screen, not dye on
cloth: averaging can never produce a mixture darker than its inputs.

**Decision.** Single-constant Kubelka-Munk on three channels
(`mix_subtractive`, used by `src/costing/solver.py`). Each dye's K/S is
measured against ideal white; the substrate's K/S is added separately and
the substrate is a required solver input (white or cotton, in config).

The catalogue (`trend_and_cost_prediction/data/dye_detail.json`, prices in
INR per kg of dye from the supplier price list in the same folder) gives
each dye an sRGB swatch. Several
are CSS named colours, with channels at exactly 0 or 255. A 0 channel means
K/S of about 500,000, which lets a millionth of a unit of that dye produce
almost any colour. Swatch reflectance is therefore clamped to
[0.03, 0.95] (`costing.swatch_reflectance_floor`/`ceiling`; ASSUMED (OM-1)
range of a dyed textile).

Unit concentration (the depth at which the swatch colour applies) is
ASSUMED (OM-1) to be 20 g of dye per kg of fabric
(`costing.reference_depth_g_per_kg`).

**Alternatives rejected.**
- RGB averaging: wrong physics, not a precision problem.
- Dye K/S as excess over the substrate it was measured on: makes dye
  constants substrate-dependent; guarded against by
  `test_subtractive_mixing.py::test_substrate_gap_shrinks_monotonically_with_load`.
- 31-band spectral Kubelka-Munk: correct, but needs per-dye reflectance
  curves from the supplier, which do not exist here.

**Consequences.** Because the swatches are placeholders and the reference
depth is assumed, **absolute rupee figures from the quote are not
trustworthy**: a full saree quotes around INR 1 of dye, because the
swatches are far more saturated than the extracted design colours and
recipes come out at about 1 g/kg. Comparisons between designs priced with
the same catalogue are more meaningful than any single figure. Fixing this
needs dye swatches measured at a known depth.

**Revisit when** the supplier provides reflectance curves (go spectral) or
measured swatches at a known depth (replace the clamp and the reference
depth).

---

## ADR-006: `colour-science` as a test-only oracle

**Context.** `perceptual.py` implements sRGB/Lab conversion, CIEDE2000 and
Kubelka-Munk from the published formulas.

**Decision.** `colour-science` is used only in tests, to cross-check the
implementation. The runtime dependency is numpy (plus scipy for the
solver's local refinement).

**Alternatives rejected.** Depending on `colour-science` at runtime: a large
general library for four functions.

**Consequences.** The by-hand formulas are checked against the oracle and
against published test vectors on every test run.

**Revisit when** the spectral migration needs illuminant and colour-matching
data that would otherwise be vendored by hand.

---

## ADR-007: Minimise cost subject to a colour tolerance

**Context.** The original solver minimised colour error and reported cost
afterwards, so cost had no influence on the recipe.

**Decision.** Minimise cost subject to ΔE2000 ≤ tolerance. Candidates are
ranked by feasibility rules: anything within tolerance beats anything
outside it; feasible candidates are ordered by cost, infeasible ones by
ΔE. If nothing is within tolerance the closest recipe is returned with
`feasible: false`, and the quote reports `all_feasible: false`.

**Alternatives rejected.**
- A weighted sum of ΔE and cost: the weight would be an unexamined exchange
  rate between visible mismatch and rupees.
- Returning the whole cost/accuracy Pareto front by default: pushes the
  decision onto every caller.

**Consequences.** A solver that minimises cost under a tolerance moves to
the edge of the tolerance: most answers sit at ΔE 2.0 minus a 1e-4 safety
margin. The quoted recipe is therefore the cheapest that is acceptable, not
the closest. That is the intended business rule, but production use would
probably aim inside the tolerance to absorb dye-lot variation.
`tests/color_science/test_dye_solver_contract.py` checks substitution of a
perceptually identical cheaper dye, cheapest-within-tolerance against a
brute-force grid, and infeasibility reporting.

**Revisit when** the business can name a real cost of a ΔE unit.

---

## ADR-008: Exhaustive search over dye sets instead of a genetic algorithm

**Context.** The project originally used a genetic algorithm (population
100, 500 generations, unseeded). A price quote must be reproducible: the
same design must get the same quote.

A GA with the new constrained objective, plus a local SLSQP refinement of
its best candidates, was built and measured during this rebuild (prototype
not kept). On the real 20-dye catalogue the quoted cost for the same target
colour varied across 10 seeds by a factor of 3 to 42; on the 5-dye contract
catalogue the spread was 0.1% to 21%. The GA's job (choosing which dyes)
was the part it did unreliably.

**Decision.** Enumerate every dye set of up to `max_dyes` = 3 dyes: 1,350
sets for 20 dyes. K/S is linear in concentration, so each set gets a
closed-form least-squares concentration in K/S space, all in one batched
solve. The cheapest few sets within tolerance and the few closest to the
target are then refined with SLSQP to the cheapest concentrations within
tolerance. The result is deterministic.

**Alternatives rejected.**
- Keep the GA and fix the seed: makes one run repeatable, but the answer
  would still depend on an arbitrary seed and swing by multiples.
- Mixed-integer programming: CIEDE2000 is non-convex and not expressible
  in a standard MIP solver without approximation.

**Consequences.** About 0.35 s per colour on this machine, most of it in
SLSQP. Determinism and catalogue-order invariance are tested
(`test_dye_solver_contract.py::test_deterministic_on_repeat`,
`::test_cost_independent_of_catalogue_order`), and the result is never
worse than a dense brute-force grid over one- and two-dye recipes
(`::test_cost_matches_brute_force_grid_over_one_and_two_dye_recipes`).
Enumeration grows as C(n, 3): about 1.3 million sets at 200 dyes, where
screening would still be a few seconds of batched linear algebra, but the
refinement shortlist would need care. The contract tests originally written
for a GA (a seed parameter and a cost-variance bound across seeds) were
replaced by determinism tests; the reasons are at the top of that test file.

**Revisit when** the catalogue grows past a few hundred dyes or recipes may
use more than 3 dyes; then a beam search or GA over dye sets, with the same
closed-form screening as its fitness, becomes worth its non-determinism.

---

## ADR-009: Trend analysis as an offline batch job; what the data supports

**Context.** The README described trend analysis over 200 Pinterest images
per season for five years. The code that existed was deleted in commit
`d12a2df`; recovered with `git show d12a2df^:fashion_trends/<file>`, it shows:

- The scrape searched Pinterest for `dresses women casual <season> <year>`
  requesting 150 images per query, for 25 folders. A YOLO filter kept 99 to
  196 per folder, 3,202 in total (`distribution.csv`).
- Fall and Autumn were scraped as separate seasons; they are the same
  season.
- The season label comes from the search query, not from when a design was
  popular.
- The clustering notebook (ResNet50 features, k-means, and ImageNet class
  names as "keywords") never finished: its only output stops while
  processing the first season.
- The scraped images were gitignored and are gone.

So no trend result was ever produced, and none can be reproduced.

**Decision.** `scripts/build_trends.py` is an offline batch job that writes
a versioned prompt-conditioning store (`$VOGUEFUSION_DATA/trends/<version>.json`
and `current.json`, atomic replace). The API serves the current store at
`GET /v1/suggestions`; nothing in the request path waits on the batch job.
Two inputs:

- The 82 LoRA training captions (default). These have no dates, so the
  store holds domain keywords, labelled as such, not trends. Auto-tagger
  misfires (for example `pokemon (creature)` on 24 of 82 saree images) are
  excluded by a list in config.
- A seasonal CSV (`season,keywords`), for which the job computes trend lift:
  a keyword's share of items in the most recent seasons divided by its share
  before them, with Fall and Autumn merged (`src/voguefusion/trends.py`,
  tested in closed form).

No new scraping was done.

**Alternatives rejected.**
- Scraping at request time (the old inspiration page does this through
  Google Images; it is kept, renamed honestly, as an ad hoc search that
  feeds nothing).
- A random per-image train/test split for any future model: near-duplicate
  re-pins would leak across the split; split by source or scrape batch
  instead.

**Consequences.** The trend feature is honest but thin until a dated,
licensed dataset exists. Scraping Pinterest or Google Images carries
terms-of-service exposure; that is a product and legal decision.

**Revisit when** a licensed seasonal dataset is available: feed it through
`--seasonal`.

---

## ADR-010: LoRA fine-tune of SDXL

**Context.** Generations need to look like saree and kurti prints.

**Decision.** A LoRA on SDXL 1.0, published on Civitai as "Kurti/Saree
Pattern" (model 1030795, version 1156167) and downloaded with hash
verification. Its safetensors metadata records the training run:

| Setting | Value |
|---|---|
| Rank / alpha | 32 / 16 (confirmed by tensor shapes: 986 `lora_down` tensors, all rank 32) |
| Images | 82, repeated 3 times (246 per epoch) |
| Epochs / steps | 10 / 660 (the Civitai page states 615) |
| Optimizer, LR | Adafactor; 5e-4 UNet, 5e-5 text encoder; cosine with restarts |
| Other | bf16, noise offset 0.1, min-SNR gamma 5, clip skip 1, seed 394287687 |

**Alternatives rejected.** A full SDXL fine-tune: far too little data (82
images). Textual inversion: weaker at the repeat structure of print
patterns (inference; not recorded at the time).

**Consequences.** The training captions are auto-tagger output with
irrelevant tags on many images (`pokemon (creature)` 24 times,
`multiple girls` 18). Retraining with cleaned captions is the obvious
improvement and is out of scope here. The LoRA is used by workflow v2.

**Revisit when** retraining: clean the captions first.

---

## ADR-011: SQLite in WAL mode as the queue and job store

**Context.** One machine, one GPU, at most a few generations a minute. The
queue needs durability, atomic claims across processes, and to be
inspectable.

**Decision.** One SQLite table holds the queue and job state
(`src/voguefusion/jobstore.py`), in WAL mode with `synchronous=FULL`. Each
mutation is a `BEGIN IMMEDIATE` transaction, which makes claim atomic
across processes. All queue SQL is in that one module.

**Alternatives rejected.**
- Redis or RabbitMQ with Celery: another process to run and monitor, and
  job state would live in two places (broker plus a results store).
- Postgres: right at larger scale, unnecessary for one machine.

**Consequences.** Throughput is bounded by SQLite's single writer, which is
orders of magnitude above what one GPU can consume. Concurrency is tested
(`test_stores.py::test_concurrent_claims_never_hand_out_a_job_twice`). The
replacement path is Postgres with `SELECT ... FOR UPDATE SKIP LOCKED` in
`claim()`, contained to one file.

**Revisit when** a second machine needs the queue.

---

## ADR-012: Client polling, not push

**Context.** Clients need to learn when a job finishes.

**Decision.** Clients poll `GET /v1/designs/{id}` (the web app once a
second). Job status includes a substate (`loading_model`,
`encoding_references`, `sampling`, ...) taken from ComfyUI's executing-node
messages, so a cold start is visibly different from a stuck job.

**Alternatives rejected.** Websockets or server-sent events to the client:
connection state on the server and reconnect logic in the client. Webhooks:
need a reachable client endpoint.

**Consequences.** One cheap indexed read per poll per client. Reconnects
need nothing special.

**Revisit when** there are enough concurrent watchers that polling load
matters; then add webhooks for server-to-server clients.

---

## ADR-013: Separate worker pools per resource

**Context.** Generation is GPU-bound and slow; costing is CPU-bound and
takes about 3 s.

**Decision.** Jobs carry a kind (`generate`, `cost`). Each kind has its own
worker processes, depth limit, lease and retry settings. The cost worker
never touches the GPU (`CUDA_VISIBLE_DEVICES=` for every process except
ComfyUI).

**Alternatives rejected.** One pool for everything: a quote would wait
behind a backlog of generations.

**Consequences.** A stuck generation does not delay quotes
(`test_api_and_workers.py::test_cost_is_not_blocked_by_a_stuck_gpu_job`).

**Revisit when** another kind of work appears, such as upscaling.

---

## ADR-014: Idempotent submit and request coalescing

**Context.** Clients retry on timeouts. Identical requests should not cost
GPU time twice.

**Decision.** Two mechanisms:

- `Idempotency-Key` header: a repeat with the same key and body returns the
  original job; the same key with a different body is 409. If the request
  has no seed, the server derives it from the key, so a retry normalises to
  the same input as the original. Without a key the seed is random.
- Input key: the SHA-256 of the normalised input (workflow file hash,
  prompt, negative, seed, batch, reference hashes; for costs, the artifact,
  fabric, solver version, catalogue hash and costing config hash). A submit
  whose input key matches a queued, running or succeeded job returns that
  job; failed and dead jobs are not reused.

**Alternatives rejected.** Client-side deduplication only: does not protect
the GPU from different clients.

**Consequences.** Repeating a finished request returns the stored result
without using the GPU (`test_api_and_workers.py::test_cached_result_does_not_touch_the_gpu`).
Changing the solver, the catalogue or the costing config changes the input
key, so stale cached quotes are not served.

**Revisit when** results must expire (they currently never do).

---

## ADR-015: Provenance on every result

**Context.** A quote or an image must be explainable later.

**Decision.** Each generate result records the seed, workflow name, the
workflow file's hash, the hashes of the models it used (from the manifest
the downloader verified against) and the ComfyUI prompt id. Each quote
records the solver version; its input key covers the catalogue and costing
config hashes. Models are only accepted when their SHA-256 matches
`comfyui/models.json`.

**Consequences.** A job whose workflow file changed between submit and run
fails permanently instead of running a different graph.

---

## ADR-016: One static web app in the original Workbench style; the legacy app is superseded

**Context.** The 2024 project had two frontends: a dark "Workbench" web app
for generation (its code was never committed, only screenshots in
`assets/images/frontend*.png`) and a Streamlit app (`trend_and_cost_prediction/`)
whose pages held the solver and cost arithmetic and passed results between
pages through the session, which caused two of the three costing bugs.

**Decision.** One web app, `ui/web/` (HTML, CSS and plain JavaScript, no
build step), served by the API at `/`. Its look follows the Workbench
screenshots (colours sampled from them). The sidebar has Generate (the
Workbench itself), Colors, Dye Recipes, Cost and Status; a generated design
goes to costing with one click. The page has no business logic: every
number it shows comes from the API.

The legacy app is left in place for reference and is no longer run. Its
three costing bugs each have a regression test against the new code
(`tests/costing/test_costing.py::test_regression_*`). Its live Google Images
search page was not carried over: it would put scraping in the API's
request path (ADR-009).

**Alternatives rejected.**
- Keep a Streamlit app beside the Workbench: two UIs with different looks
  and a hand-off between them through a URL.
- React or Next.js: a Node toolchain for five small views.

**Consequences.** Same origin as the API, so no CORS. Any client (the web
app, `scripts/e2e.py`, the load test) exercises the same API.
`scripts/ui_check.py` drives the whole flow in headless Chrome as part of
`scripts/verify.sh`.

**Revisit when** the legacy directory is deleted; the dye catalogue should
then move to `data/`.
