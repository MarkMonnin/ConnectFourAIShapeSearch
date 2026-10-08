(function (global) {
  "use strict";

  var ACTION_COUNT = 7;
  var C = global.C4_CONSTANTS;
  var Q_MAX_STATES = (C && C.Q_MAX_STATES) || 10000;
  var MENACE_MAX_BOXES = (C && C.MENACE_MAX_BOXES) || 10000;
  var EVICT_FRACTION = 0.1;

  function pickMax(values, moves) {
    var best = -Infinity;
    var picks = [];
    for (var i = 0; i < moves.length; i += 1) {
      var move = moves[i];
      if (values[move] > best) {
        best = values[move];
        picks = [move];
      } else if (values[move] === best) {
        picks.push(move);
      }
    }
    return picks[Math.floor(Math.random() * picks.length)];
  }

  function pickBead(beads, moves) {
    var total = 0;
    for (var i = 0; i < moves.length; i += 1) {
      total += beads[moves[i]];
    }
    if (total <= 0) {
      for (var j = 0; j < moves.length; j += 1) {
        beads[moves[j]] = 1;
      }
      return moves[Math.floor(Math.random() * moves.length)];
    }
    var roll = Math.random() * total;
    for (var k = 0; k < moves.length; k += 1) {
      roll -= beads[moves[k]];
      if (roll <= 0) {
        return moves[k];
      }
    }
    return moves[moves.length - 1];
  }

  function zeroActions() {
    var v = [];
    for (var i = 0; i < ACTION_COUNT; i += 1) {
      v.push(0);
    }
    return v;
  }

  /** Immediate win, else block opponent's immediate win. Null if neither. */
  function pickTacticalMove(board, player, moves) {
    var G = global.C4_GAME;
    if (!G || !moves || !moves.length) {
      return null;
    }
    var other = G.other(player);
    var i;
    var next;
    for (i = 0; i < moves.length; i += 1) {
      next = G.applyMove(board, player, moves[i]);
      if (G.findWinner(next) === player) {
        return moves[i];
      }
    }
    for (i = 0; i < moves.length; i += 1) {
      next = G.applyMove(board, other, moves[i]);
      if (G.findWinner(next) === other) {
        return moves[i];
      }
    }
    return null;
  }

  function evictLru(table, lastUsed, related, maxKeep) {
    var keys = Object.keys(table);
    if (keys.length <= maxKeep) {
      return;
    }
    keys.sort(function (a, b) {
      return (lastUsed[a] || 0) - (lastUsed[b] || 0);
    });
    var toRemove = Math.max(1, Math.floor(keys.length * EVICT_FRACTION));
    if (keys.length - toRemove > maxKeep) {
      toRemove = keys.length - maxKeep;
    }
    for (var i = 0; i < toRemove; i += 1) {
      var key = keys[i];
      delete table[key];
      delete lastUsed[key];
      if (related) {
        for (var r = 0; r < related.length; r += 1) {
          delete related[r][key];
        }
      }
    }
  }

  function QBrain() {
    this.qTable = {};
    this.visits = {};
    this.lastUsedEpisode = {};
    this.episodeCounter = 0;
    this.epsilon = 0.2;
    this.lruCapReached = false;
  }

  QBrain.prototype.touchState = function (stateKey) {
    this.lastUsedEpisode[stateKey] = this.episodeCounter;
    this.episodeCounter += 1;
  };

  QBrain.prototype.stateKey = function (boardKey, player) {
    return boardKey + "|" + player;
  };

  QBrain.prototype.storageCapReached = function () {
    return !!this.lruCapReached;
  };

  QBrain.prototype.ensureQ = function (stateKey) {
    if (!this.qTable[stateKey]) {
      if (Object.keys(this.qTable).length >= Q_MAX_STATES) {
        this.lruCapReached = true;
        evictLru(this.qTable, this.lastUsedEpisode, [this.visits], Q_MAX_STATES - 1);
      }
      this.qTable[stateKey] = zeroActions();
    }
    this.touchState(stateKey);
    return this.qTable[stateKey];
  };

  QBrain.prototype.ensureVisits = function (stateKey) {
    if (!this.visits[stateKey]) {
      this.visits[stateKey] = zeroActions();
    }
    return this.visits[stateKey];
  };

  QBrain.prototype.chooseMove = function (board, player, boardKeyFn, legalMovesFn, explore) {
    var moves = legalMovesFn(board);
    if (moves.length === 0) {
      return null;
    }
    var tactical = pickTacticalMove(board, player, moves);
    if (tactical !== null) {
      return tactical;
    }
    if (explore && Math.random() < this.epsilon) {
      this.episodeCounter += 1;
      return moves[Math.floor(Math.random() * moves.length)];
    }
    var stateKey = this.stateKey(boardKeyFn(board), player);
    var q = this.ensureQ(stateKey);
    /* Use raw Q values. Do not treat unvisited actions as -Infinity: that made
       greedy play (eval) forever repeat the first tried move in each state,
       so Q/SARSA could not beat Random after short Shape Search training. */
    this.ensureVisits(stateKey);
    var best = -Infinity;
    var picks = [];
    for (var i = 0; i < moves.length; i += 1) {
      var move = moves[i];
      var value = q[move];
      if (value > best) {
        best = value;
        picks = [move];
      } else if (value === best) {
        picks.push(move);
      }
    }
    return picks[Math.floor(Math.random() * picks.length)];
  };

  QBrain.prototype.getScores = function (board, player, boardKeyFn) {
    var stateKey = this.stateKey(boardKeyFn(board), player);
    this.ensureQ(stateKey);
    return this.qTable[stateKey].slice();
  };

  QBrain.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    if (!trajectory.length) {
      return;
    }
    var terminal = 0;
    if (winner === mark) {
      terminal = 1;
    } else if (winner && winner !== mark) {
      terminal = -1;
    }
    /* Reverse one-step Q: bootstrap max Q(s', ·) so credit reaches early plies.
       (Old every-visit MC with -4 on every step crushed openings and barely
       beat Random.) */
    var i;
    for (i = trajectory.length - 1; i >= 0; i -= 1) {
      var step = trajectory[i];
      var q = this.ensureQ(step.stateKey);
      var visits = this.ensureVisits(step.stateKey);
      visits[step.action] += 1;
      var alpha = 1 / visits[step.action];
      var target;
      if (i + 1 < trajectory.length) {
        var nxt = trajectory[i + 1];
        var nxtQ = this.ensureQ(nxt.stateKey);
        var legal = nxt.moves && nxt.moves.length ? nxt.moves : null;
        var best = -Infinity;
        var a;
        if (legal) {
          for (a = 0; a < legal.length; a += 1) {
            if (nxtQ[legal[a]] > best) {
              best = nxtQ[legal[a]];
            }
          }
        } else {
          for (a = 0; a < nxtQ.length; a += 1) {
            if (nxtQ[a] > best) {
              best = nxtQ[a];
            }
          }
        }
        target = best === -Infinity ? 0 : best;
      } else {
        target = terminal;
      }
      q[step.action] += alpha * (target - q[step.action]);
    }
  };

  QBrain.prototype.stateCount = function () {
    return Object.keys(this.qTable).length;
  };

  QBrain.prototype.trimToFraction = function (fraction) {
    var target = Math.floor(Q_MAX_STATES * fraction);
    evictLru(this.qTable, this.lastUsedEpisode, [this.visits], target);
  };

  QBrain.prototype.toJSON = function () {
    return {
      qTable: this.qTable,
      visits: this.visits,
      lastUsedEpisode: this.lastUsedEpisode,
      episodeCounter: this.episodeCounter,
      epsilon: this.epsilon,
      lruCapReached: this.lruCapReached
    };
  };

  QBrain.prototype.toJSONMaxStates = function (maxStates) {
    var keys = Object.keys(this.qTable);
    if (keys.length <= maxStates) {
      return this.toJSON();
    }
    keys.sort(function (a, b) {
      return (this.lastUsedEpisode[a] || 0) - (this.lastUsedEpisode[b] || 0);
    }.bind(this));
    var keep = keys.slice(keys.length - maxStates);
    var qTable = {};
    var visits = {};
    var lastUsedEpisode = {};
    var i;
    var k;
    for (i = 0; i < keep.length; i += 1) {
      k = keep[i];
      qTable[k] = this.qTable[k];
      if (this.visits[k]) {
        visits[k] = this.visits[k];
      }
      if (this.lastUsedEpisode[k]) {
        lastUsedEpisode[k] = this.lastUsedEpisode[k];
      }
    }
    return {
      qTable: qTable,
      visits: visits,
      lastUsedEpisode: lastUsedEpisode,
      episodeCounter: this.episodeCounter,
      epsilon: this.epsilon,
      lruCapReached: this.lruCapReached
    };
  };

  QBrain.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.qTable = data.qTable || {};
    this.visits = data.visits || {};
    this.lastUsedEpisode = data.lastUsedEpisode || {};
    this.episodeCounter = data.episodeCounter || 0;
    this.epsilon = typeof data.epsilon === "number" ? data.epsilon : 0.2;
    this.lruCapReached = !!data.lruCapReached;
  };

  function MenaceBrain() {
    this.boxes = {};
    this.lastUsedEpisode = {};
    this.episodeCounter = 0;
    this.lruCapReached = false;
  }

  MenaceBrain.prototype.touchState = function (stateKey) {
    this.lastUsedEpisode[stateKey] = this.episodeCounter;
    this.episodeCounter += 1;
  };

  MenaceBrain.prototype.stateKey = function (boardKey, player) {
    return boardKey + "|" + player;
  };

  MenaceBrain.prototype.storageCapReached = function () {
    return !!this.lruCapReached;
  };

  MenaceBrain.prototype.ensureBox = function (stateKey, moves) {
    if (!this.boxes[stateKey]) {
      if (Object.keys(this.boxes).length >= MENACE_MAX_BOXES) {
        this.lruCapReached = true;
        evictLru(this.boxes, this.lastUsedEpisode, null, MENACE_MAX_BOXES - 1);
      }
      this.boxes[stateKey] = zeroActions();
      for (var i = 0; i < moves.length; i += 1) {
        this.boxes[stateKey][moves[i]] = 1;
      }
    }
    this.touchState(stateKey);
    return this.boxes[stateKey];
  };

  MenaceBrain.prototype.chooseMove = function (board, player, boardKeyFn, legalMovesFn, explore) {
    var moves = legalMovesFn(board);
    if (moves.length === 0) {
      return null;
    }
    var tactical = pickTacticalMove(board, player, moves);
    if (tactical !== null) {
      return tactical;
    }
    var beads = this.ensureBox(this.stateKey(boardKeyFn(board), player), moves);
    if (explore) {
      return pickBead(beads, moves);
    }
    return pickMax(beads, moves);
  };

  MenaceBrain.prototype.getScores = function (board, player, boardKeyFn, legalMovesFn) {
    var moves = legalMovesFn(board);
    return this.ensureBox(this.stateKey(boardKeyFn(board), player), moves).slice();
  };

  MenaceBrain.prototype.learnFromTrajectory = function (trajectory, winner) {
    for (var i = 0; i < trajectory.length; i += 1) {
      var step = trajectory[i];
      var beads = this.ensureBox(step.stateKey, step.moves);
      if (!winner) {
        continue;
      }
      /* Classic MENACE deltas: +3 on win, -1 on loss (floor at 0). */
      if (winner === step.player) {
        beads[step.action] += 3;
      } else if (beads[step.action] > 0) {
        beads[step.action] -= 1;
      }
    }
  };

  MenaceBrain.prototype.boxCount = function () {
    return Object.keys(this.boxes).length;
  };

  MenaceBrain.prototype.trimToFraction = function (fraction) {
    var target = Math.floor(MENACE_MAX_BOXES * fraction);
    evictLru(this.boxes, this.lastUsedEpisode, null, target);
  };

  MenaceBrain.prototype.toJSON = function () {
    return {
      boxes: this.boxes,
      lastUsedEpisode: this.lastUsedEpisode,
      episodeCounter: this.episodeCounter,
      lruCapReached: this.lruCapReached
    };
  };

  MenaceBrain.prototype.toJSONMaxBoxes = function (maxBoxes) {
    var keys = Object.keys(this.boxes);
    if (keys.length <= maxBoxes) {
      return this.toJSON();
    }
    keys.sort(function (a, b) {
      return (this.lastUsedEpisode[a] || 0) - (this.lastUsedEpisode[b] || 0);
    }.bind(this));
    var keep = keys.slice(keys.length - maxBoxes);
    var boxes = {};
    var lastUsedEpisode = {};
    var i;
    var k;
    for (i = 0; i < keep.length; i += 1) {
      k = keep[i];
      boxes[k] = this.boxes[k];
      if (this.lastUsedEpisode[k]) {
        lastUsedEpisode[k] = this.lastUsedEpisode[k];
      }
    }
    return {
      boxes: boxes,
      lastUsedEpisode: lastUsedEpisode,
      episodeCounter: this.episodeCounter,
      lruCapReached: this.lruCapReached
    };
  };

  MenaceBrain.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.boxes = data.boxes || {};
    this.lastUsedEpisode = data.lastUsedEpisode || {};
    this.episodeCounter = data.episodeCounter || 0;
    this.lruCapReached = !!data.lruCapReached;
  };

  function SeededRandom() {
    this.lastSeed = 0;
  }

  SeededRandom.prototype.createRng = function (seed) {
    var state = seed >>> 0;
    return function next() {
      state = (state + 0x6D2B79F5) >>> 0;
      var t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  SeededRandom.prototype.newSeed = function () {
    this.lastSeed = (Math.random() * 0x100000000) >>> 0;
    return this.lastSeed;
  };

  SeededRandom.prototype.openingColumnFromSeed = function (seed) {
    var rng = this.createRng(seed);
    return Math.floor(rng() * 7);
  };

  SeededRandom.prototype.gameSetupFromSeed = function (seed) {
    var rng = this.createRng(seed);
    return {
      opening: Math.floor(rng() * 7),
      learnerMark: rng() < 0.5 ? "X" : "O",
      rng: rng
    };
  };

  SeededRandom.prototype.chooseMove = function (moves, seed) {
    var rng = this.createRng(seed);
    rng();
    rng();
    return moves[Math.floor(rng() * moves.length)];
  };

  var NEMESIS_CAP = 500;
  var NEMESIS_SUCCESSES_REQUIRED = 10;
  var NEMESIS_COOLDOWN_EPISODES = 1000;
  var LOSS_REPLAY_COUNT = 5;

  function normalizeNemesisLevel(level) {
    if (level === "random" || level === null || level === undefined) {
      return "random";
    }
    return Math.round(parseFloat(level) * 10) / 10;
  }

  function nemesisEntryKey(seed, level) {
    return String(seed) + "|" + normalizeNemesisLevel(level);
  }

  function normalizeNemesisEntry(raw) {
    if (typeof raw === "number") {
      return { seed: raw, level: "random", successes: 0 };
    }
    return {
      seed: raw.seed,
      level: normalizeNemesisLevel(raw.level),
      successes: typeof raw.successes === "number" ? raw.successes :
        (typeof raw.draws === "number" ? raw.draws : (raw.wins || 0))
    };
  }

  function RollingResults(capacity) {
    this.capacity = capacity;
    this.items = [];
  }

  RollingResults.prototype.push = function (result) {
    this.items.push(result);
    if (this.items.length > this.capacity) {
      this.items.shift();
    }
  };

  RollingResults.prototype.counts = function () {
    var wins = 0;
    var draws = 0;
    var losses = 0;
    for (var i = 0; i < this.items.length; i += 1) {
      if (this.items[i] === "win") {
        wins += 1;
      } else if (this.items[i] === "draw") {
        draws += 1;
      } else {
        losses += 1;
      }
    }
    return { wins: wins, draws: draws, losses: losses, size: this.items.length };
  };

  RollingResults.prototype.toJSON = function () {
    return { capacity: this.capacity, items: this.items.slice() };
  };

  RollingResults.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.capacity = data.capacity || this.capacity;
    this.items = data.items || [];
  };

  function NemesisTracker() {
    this.seeds = [];
    this.cooldown = [];
    this.totalEpisodes = 0;
  }

  NemesisTracker.prototype.findEntryIndex = function (seed, level) {
    var key = nemesisEntryKey(seed, level);
    var i;
    for (i = 0; i < this.seeds.length; i += 1) {
      if (nemesisEntryKey(this.seeds[i].seed, this.seeds[i].level) === key) {
        return i;
      }
    }
    return -1;
  };

  NemesisTracker.prototype.findSeedIndex = function (seed) {
    var i;
    for (i = 0; i < this.seeds.length; i += 1) {
      if (this.seeds[i].seed === seed) {
        return i;
      }
    }
    return -1;
  };

  NemesisTracker.prototype.isOnCooldown = function (seed, level) {
    var key = nemesisEntryKey(seed, level);
    for (var i = 0; i < this.cooldown.length; i += 1) {
      var cd = this.cooldown[i];
      var cdLevel = cd.level !== undefined ? cd.level : "random";
      if (nemesisEntryKey(cd.seed, cdLevel) === key) {
        return this.totalEpisodes < cd.releaseEpisode;
      }
    }
    return false;
  };

  NemesisTracker.prototype.removeFromCooldown = function (seed, level) {
    var key = nemesisEntryKey(seed, level);
    var kept = [];
    for (var i = 0; i < this.cooldown.length; i += 1) {
      var cd = this.cooldown[i];
      var cdLevel = cd.level !== undefined ? cd.level : "random";
      if (nemesisEntryKey(cd.seed, cdLevel) !== key) {
        kept.push(cd);
      }
    }
    this.cooldown = kept;
  };

  NemesisTracker.prototype.pickEligibleEntry = function () {
    for (var i = 0; i < this.seeds.length; i += 1) {
      var entry = this.seeds[i];
      if (!this.isOnCooldown(entry.seed, entry.level)) {
        return entry;
      }
    }
    return null;
  };

  NemesisTracker.prototype.activeCount = function () {
    return this.seeds.length;
  };

  NemesisTracker.prototype.addNemesis = function (seed, level) {
    level = normalizeNemesisLevel(level);
    var idx = this.findEntryIndex(seed, level);
    var entry;
    if (idx >= 0) {
      entry = this.seeds.splice(idx, 1)[0];
      entry.successes = 0;
    } else {
      entry = { seed: seed, level: level, successes: 0 };
    }
    this.removeFromCooldown(seed, level);
    this.seeds.unshift(entry);
    if (this.seeds.length > NEMESIS_CAP) {
      this.seeds.length = NEMESIS_CAP;
    }
  };

  NemesisTracker.prototype.addNemesisSeed = function (seed) {
    this.addNemesis(seed, "random");
  };

  NemesisTracker.prototype.recordNemesisSuccess = function (seed, level) {
    var idx = this.findEntryIndex(seed, level);
    if (idx < 0) {
      return false;
    }
    var entry = this.seeds[idx];
    entry.successes += 1;
    if (entry.successes >= NEMESIS_SUCCESSES_REQUIRED) {
      this.seeds.splice(idx, 1);
      this.cooldown.push({
        seed: seed,
        level: normalizeNemesisLevel(level),
        releaseEpisode: this.totalEpisodes + NEMESIS_COOLDOWN_EPISODES
      });
      return true;
    }
    return false;
  };

  NemesisTracker.prototype.recordNemesisLoss = function (seed, level) {
    var idx = this.findEntryIndex(seed, level);
    if (idx < 0) {
      return;
    }
    var entry = this.seeds.splice(idx, 1)[0];
    entry.successes = 0;
    this.seeds.unshift(entry);
  };

  NemesisTracker.prototype.resetNemesisSuccesses = function (seed, level) {
    var idx = this.findEntryIndex(seed, level);
    if (idx >= 0) {
      this.seeds[idx].successes = 0;
    }
  };

  NemesisTracker.prototype.tickEpisode = function () {
    this.totalEpisodes += 1;
  };

  NemesisTracker.prototype.toJSON = function () {
    return {
      seeds: this.seeds,
      cooldown: this.cooldown,
      totalEpisodes: this.totalEpisodes
    };
  };

  NemesisTracker.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.seeds = [];
    if (data.seeds) {
      for (var i = 0; i < data.seeds.length; i += 1) {
        this.seeds.push(normalizeNemesisEntry(data.seeds[i]));
      }
    }
    this.cooldown = data.cooldown || [];
    this.totalEpisodes = data.totalEpisodes || 0;
  };

  global.C4_OPPONENTS = {
    QBrain: QBrain,
    MenaceBrain: MenaceBrain,
    SeededRandom: SeededRandom,
    NemesisTracker: NemesisTracker,
    normalizeNemesisLevel: normalizeNemesisLevel,
    RollingResults: RollingResults,
    Q_MAX_STATES: Q_MAX_STATES,
    MENACE_MAX_BOXES: MENACE_MAX_BOXES,
    EVICT_FRACTION: EVICT_FRACTION,
    NEMESIS_CAP: NEMESIS_CAP,
    NEMESIS_SUCCESSES_REQUIRED: NEMESIS_SUCCESSES_REQUIRED,
    NEMESIS_COOLDOWN_EPISODES: NEMESIS_COOLDOWN_EPISODES,
    LOSS_REPLAY_COUNT: LOSS_REPLAY_COUNT
  };
})(window);
