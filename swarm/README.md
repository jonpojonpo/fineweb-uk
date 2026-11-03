# Volunteer labelling swarm

Volunteers label pinned FineWeb-Edu row groups on their own machines, in the style of SETI@home. A coordinator hands out work units and accepts a unit once two different volunteers agree on it.

Each worker downloads text directly from Hugging Face and uploads only document IDs, text hashes and label probabilities. The coordinator never receives or stores page text.

## Run a coordinator

```bash
python swarm/coordinator.py init --db swarm.db --kind croydon --shards 0-139 --span 50
SWARM_INVITE=choose-a-code python swarm/coordinator.py serve --db swarm.db --results swarm_results
```

It listens on `127.0.0.1:8770`. To accept volunteers from outside, put it behind an HTTPS reverse proxy, because tokens travel in request headers. `/stats` shows progress and a leaderboard.

For teacher labelling, use `--kind croydon-teacher --span 3`: a unit of 3 row groups holds about 90 `.uk` documents and takes a 4090 about 2 minutes.

## Run a worker

```bash
pip install -r requirements.txt torch "transformers>=5.10"
python swarm/worker.py --server https://your-coordinator --name your-pseudonym --invite CODE --croydon jonpojonpo/croydon-v0
python swarm/worker.py --server https://your-coordinator --name your-pseudonym --invite CODE --teacher http://127.0.0.1:8092
```

- **Croydon:** runs on a GPU or CPU, and uses mostly bandwidth, about 3 MB per row group.
- **Teacher (Qwen):** needs a local `llama-server` running the weights named in `pins.env`, on a GPU with about 20 GB of memory.

Stop at any time. An unfinished unit returns to the queue when its one-hour lease expires. Your token is saved in `~/.config/fineweb-uk-swarm/`.

## What it protects against

| Risk | Protection |
|---|---|
| Faulty or random results | Two different volunteers must agree, or the unit goes to another volunteer. After 4 failed rounds it is marked `disputed` |
| Malformed uploads | Strict checks on fields, probabilities and size; page text is rejected |
| A malicious coordinator | The worker only fetches files matching the pinned sample, using the repo and revision in its own `pins.env`, and treats responses as data |
| Lost work | Leases expire and units are reissued |
| One person confirming their own work from the same connection | Each unit goes to two different networks: a keyed hash of the IPv4 address or IPv6 /64 is stored, never the address itself. Households and offices sharing one connection simply get different units |

## What it doesn't protect against

- **One person with two accounts on different networks,** for example using a VPN or mobile data, can still confirm their own results. Two accounts on the same IPv4 address or IPv6 /64 can't work on the same unit. Use invite codes, and audit a random sample of accepted units on trusted hardware before using them.
- **The same honest mistake:** two volunteers running the same model agree even when the model is wrong. Agreement shows the work was done, not that the labels are correct.
- **Different model versions:** the model version is not cryptographically proven. A volunteer can run something else and only the comparison will catch it.
- **Worker code:** volunteers run this repo's Python and, for Croydon workers, a downloaded weights file. Pin a release and load weights only from a source you trust.

## Before launching publicly

- Publish the Croydon weights and pin them as `CROYDON_REPO` and `CROYDON_REV` in `pins.env`.
- Host the coordinator behind HTTPS and back up its database.
- Check whether Hugging Face is happy with the download volume. A full `sample/100BT` pass with two volunteers per unit is about 570 GB.
- Tell volunteers they will download and process third-party web pages, which may include copyrighted, adult or upsetting material.
