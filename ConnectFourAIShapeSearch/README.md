# Connect Four Shape Search

Browser lab that searches for Connect Four bot architectures that learn well. Everything runs locally in a modern browser - no build step, no backend, no install.

Open [`c4-shapesearch.html`](c4-shapesearch.html) as a file (double-click or File > Open). Do not require an HTTP server.

## What it does

The lab evolves and trains learning agents against a rising ladder of opponents, then lets you promote favorites and watch games.

| Tab | Purpose |
| --- | --- |
| **Shape Search** | Population of bots trains and is graded vs Random, then Minimax depth 1, 2, ... Survivors morph / clone into the next round after a tournament. |
| **Train One** | Pull one bot out of Search and climb the same ladder with focused training, export/import, and Save to Search. |
| **Observe** | Play or watch games: Random, Minimax, MCTS, the current trainee, and Search bots. |

Progress auto-saves in the browser (IndexedDB `ConnectFourShapeSearch`, with a localStorage fallback).

## Shape Search (short)

1. Round 1 grades vs **Random**; later rounds vs **Minimax** at increasing strength.
2. Bots train on their own clock; short evals (20 games) can extend to 100 once they clear WR > 50%.
3. A bot "beats" the round opponent after **10 consecutive** evals with points win rate **> 50%**.
4. When at least one bot has beaten the round **ten times in a row** and everyone has hit a short train floor, a live round-robin tournament runs; the **top 10** seed the next population, and the **top 3** also spawn mutants (or NEAT clones).

`toBeat_s` in the population table is the train time when a bot first cleared a **100-game** WR > 50% that round.

## Train One

- Start from a live Search bot or a tournament snapshot.
- **Export** / **Import** JSON to move a bot between machines or sessions.
- **Save to Search** overwrites the original Search bot, or adds an imported bot under the name you chose.
- Contender tournaments highlight the trainee row with dashed lines above and below.

## Algorithms

Includes tabular agents (Q-learning, SARSA, MENACE, genetic MENACE), neural nets (backprop / REINFORCE policy and value, several widths and depths), NEAT / NEAT-value, and TensorFlow.js hybrids. Baselines for play and tournaments: Random, timed Minimax, and a light MCTS-style agent.

TF.js is vendored at `vendor/tf.min.js` (no CDN). Other agents do not need it.

## Tests

Requires Node.js. From the repo root:

```bash
node tests/test-shape-lab.js
node tests/test-minimax-block.js
```

Optional long Minimax self-check (about an hour at 1s/move):

```bash
node tests/h2h-minimax-new-vs-old.js
```

## Layout

```text
c4-shapesearch.html   # UI entry
c4-shapesearch.css
c4_js/                # game, minimax, engine, shape lab, play UI
c4-network.js         # neural net helpers
c4-opponents.js
vendor/tf.min.js      # TensorFlow.js 4.22.0 (offline)
tests/                # unit tests and benches
docs/                 # notes (e.g. example boards)
```

## Notes

- Leave the tab open while Search or Train One is running; work pauses when you leave the tab (Search auto-pauses).
- Clearing site data for the page wipes saved populations and trainees.
- This is a research / hobby lab, not a polished product UI.
