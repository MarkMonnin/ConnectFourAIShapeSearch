/**
 * Shape Search + Train One lab (front-end meta-GA and single-bot ladder).
 * Depends on C4_APP shape helpers, C4_GAME, C4_MINIMAX, C4_CONSTANTS.
 */
(function (global) {
  "use strict";

  var APP = global.C4_APP;
  var G = global.C4_GAME;
  var C = global.C4_CONSTANTS;
  var M = global.C4_MINIMAX;

  var STORAGE_KEY = "learnC4_shape_v1";
  var IDB_NAME = "ConnectFourShapeSearch";
  var IDB_STORE = "checkpoints";
  var AUTO_SAVE_MS = 30000;
  var SLICE_MS = 10000;
  var MIN_DWELL_SEC = 0.5;
  var EVAL_GAMES = 20;
  var EVAL_GAMES_LONG = 100;
  var BEAT_WR = 0.5;
  /** Consecutive evals with WR > 50% required before fill credit (beaten). */
  var BEAT_STREAK_NEED = 10;
  /** Restore saved weights if eval WR falls this far below personal best. */
  var BEST_WR_DIP = 0.10;
  var NEMESIS_MAX = 24;
  var WARMUP_MS = 100;
  var TRAIN_BETWEEN_EVAL_MS = 50;
  var TRAIN_TO_EVAL_RATIO = 4;
  /** Learned bots kept into the next round after each tournament. */
  var MAX_SURVIVORS = 10;
  /** Only these top finishers spawn mutants / NEAT clones. */
  var MAX_MUTANT_PARENTS = 3;
  /** @deprecated aliases - always keep top MAX_SURVIVORS. */
  var MAX_SURVIVORS_FIRST = MAX_SURVIVORS;
  var MAX_SURVIVORS_LATER = MAX_SURVIVORS;
  /** Learned bots kept in tournament history (baselines may add extra rows). */
  var HISTORY_LEARNED_LIMIT = 10;
  /** Include Minimax/MCTS in history only when they finish in this overall band. */
  var HISTORY_BASELINE_TOP_N = 5;
  var MAX_NEAT_COPIES = 3;
  /** Minimum train_s every bot must reach each round before the round can end. */
  /** Min train wall seconds once a beater exists (MM1+ rounds). */
  var MIN_TRAIN_SEC = 10;
  /** R1 vs Random uses a shorter floor so the first round finishes sooner. */
  var MIN_TRAIN_SEC_RANDOM = 3;
  /** @deprecated alias - same as MIN_TRAIN_SEC (no longer earliest-beat + 10). */
  var FILL_AFTER_BEAT_SEC = MIN_TRAIN_SEC;
  var NEAR_WIDTH_DUP = 3;
  var CHECKPOINT_FORMAT = "c4-shape-lab-v1";
  var EXPORT_FORMAT = "c4-shape-bot-v1";
  var LR_MIN_NN = 0.003;
  var LR_MAX_NN = 0.1;
  var LR_MIN_TF = 0.0001;
  var LR_MAX_TF = 0.01;

  var nextId = 1;
  var neatSerialByFamily = { neat: 0, neat_value: 0 };

  function uid(prefix) {
    var id = prefix + "_" + nextId;
    nextId += 1;
    return id;
  }

  function isNeatFamily(family) {
    return family === "neat" || family === "neat_value";
  }

  function neatDisplayName(family, serial) {
    if (family === "neat_value") {
      return "NEAT value " + serial;
    }
    return "NEAT " + serial;
  }

  function noteNeatSerial(family, serial) {
    if (!isNeatFamily(family) || typeof serial !== "number") {
      return;
    }
    if (serial > (neatSerialByFamily[family] || 0)) {
      neatSerialByFamily[family] = serial;
    }
  }

  function assignNeatIdentity(ind) {
    if (!ind || !isNeatFamily(ind.family)) {
      return ind;
    }
    if (typeof ind.neatSerial !== "number") {
      neatSerialByFamily[ind.family] = (neatSerialByFamily[ind.family] || 0) + 1;
      ind.neatSerial = neatSerialByFamily[ind.family];
    } else {
      noteNeatSerial(ind.family, ind.neatSerial);
    }
    /* Keep a user-chosen name. Generated labels are "NEAT" / "NEAT 3". */
    var generated = neatDisplayName(ind.family, ind.neatSerial);
    var generic = familyLabel(ind.family, ind.layerSizes);
    if (!ind.label || ind.label === generic || /^NEAT( value)? \d+$/.test(ind.label)) {
      ind.label = generated;
    }
    return ind;
  }

  function cloneSizes(sizes) {
    return sizes ? sizes.slice() : null;
  }

  function sizeSignature(family, sizes) {
    return family + ":" + (sizes ? sizes.join("x") : "na");
  }

  function idAge(id) {
    if (id === null || id === undefined) {
      return Number.MAX_SAFE_INTEGER;
    }
    var m = String(id).match(/_(\d+)$/);
    if (!m) {
      return Number.MAX_SAFE_INTEGER;
    }
    return parseInt(m[1], 10);
  }

  /**
   * Same family + same depth/IO; each hidden width within NEAR_WIDTH_DUP units.
   * Example: [78,64,78,7] near [78,67,75,7].
   */
  function layersNearDuplicate(a, b) {
    if (!a && !b) {
      return true;
    }
    if (!a || !b || a.length !== b.length) {
      return false;
    }
    if (a[0] !== b[0] || a[a.length - 1] !== b[b.length - 1]) {
      return false;
    }
    var i;
    for (i = 1; i < a.length - 1; i += 1) {
      if (Math.abs(a[i] - b[i]) > NEAR_WIDTH_DUP) {
        return false;
      }
    }
    return true;
  }

  function isNearDuplicateShape(familyA, sizesA, familyB, sizesB) {
    if (familyA !== familyB) {
      return false;
    }
    /* Tabular / NEAT seats are distinct individuals (null layerSizes). */
    if (!sizesA || !sizesB) {
      return false;
    }
    return layersNearDuplicate(sizesA, sizesB);
  }

  function findNearDuplicateIndex(nextPop, family, sizes) {
    var i;
    for (i = 0; i < nextPop.length; i += 1) {
      if (isNearDuplicateShape(family, sizes, nextPop[i].family, nextPop[i].layerSizes)) {
        return i;
      }
    }
    return -1;
  }

  /** Keep oldest id when two near-duplicate shapes collide. */
  function tryAddUniqueShape(nextPop, candidate) {
    var idx = findNearDuplicateIndex(nextPop, candidate.family, candidate.layerSizes);
    if (idx < 0) {
      nextPop.push(candidate);
      return true;
    }
    var existing = nextPop[idx];
    if (idAge(candidate.id) < idAge(existing.id)) {
      nextPop[idx] = candidate;
      return true;
    }
    return false;
  }

  function familyLabel(family, sizes) {
    var base = {
      nn_policy: "Backprop policy",
      nn_value: "Backprop value",
      reinforce_policy: "REINFORCE policy",
      reinforce_value: "REINFORCE value",
      tfjs_policy: "TF.js hybrid",
      tfjs_value: "TF.js value",
      qtable: "Q-Learning",
      sarsa: "SARSA",
      menace: "MENACE",
      neat: "NEAT",
      neat_value: "NEAT value",
      genetic_menace: "Genetic MENACE"
    }[family] || family;
    if (!sizes) {
      return base;
    }
    var hidden = Math.max(0, sizes.length - 2);
    return base + " " + hidden + "L [" + sizes.join(",") + "]";
  }

  function isMorphable(family) {
    return /^(nn_|reinforce_|tfjs_)/.test(family);
  }

  function gen0Specs() {
    var morph = [
      "nn_policy", "nn_value", "reinforce_policy", "reinforce_value",
      "tfjs_policy", "tfjs_value"
    ];
    var tabular = [
      { family: "qtable", layerSizes: null },
      { family: "sarsa", layerSizes: null },
      { family: "menace", layerSizes: null },
      { family: "genetic_menace", layerSizes: null }
    ];
    var out = [];
    var i;
    for (i = 0; i < morph.length; i += 1) {
      var fam = morph[i];
      var base = APP.defaultLayerSizes(fam);
      var narrow = mutateWidth(base, 2 / 3);
      var wide = mutateWidth(base, 1.5);
      out.push({ family: fam, layerSizes: narrow, learningRate: APP.defaultLearningRate(fam) });
      out.push({ family: fam, layerSizes: base, learningRate: APP.defaultLearningRate(fam) });
      out.push({ family: fam, layerSizes: wide, learningRate: APP.defaultLearningRate(fam) });
    }
    for (i = 0; i < tabular.length; i += 1) {
      out.push(tabular[i]);
    }
    /* Several independent NEAT seats so the family has a fair shot in R1. */
    for (i = 0; i < MAX_NEAT_COPIES; i += 1) {
      out.push({ family: "neat", layerSizes: null });
    }
    for (i = 0; i < MAX_NEAT_COPIES; i += 1) {
      out.push({ family: "neat_value", layerSizes: null });
    }
    return out;
  }

  function mutateWidth(sizes, factor) {
    if (!sizes || sizes.length < 3) {
      return null;
    }
    var out = sizes.slice();
    var i;
    for (i = 1; i < out.length - 1; i += 1) {
      out[i] = Math.max(8, Math.min(512, Math.round(out[i] * factor)));
    }
    return out;
  }

  function mutateDepth(sizes, delta, family) {
    if (!sizes || sizes.length < 3) {
      return null;
    }
    var input = sizes[0];
    var output = sizes[sizes.length - 1];
    var hidden = sizes.slice(1, -1);
    if (delta > 0) {
      if (hidden.length >= 6) {
        return null;
      }
      var mid = hidden.length ? hidden[Math.floor(hidden.length / 2)] : 64;
      hidden.push(Math.max(8, Math.round(mid * 0.85)));
    } else {
      if (hidden.length <= 1) {
        return null;
      }
      hidden.pop();
    }
    return [input].concat(hidden).concat([output]);
  }

  function clampLr(family, lr) {
    if (family === "tfjs_policy" || family === "tfjs_value") {
      return Math.max(LR_MIN_TF, Math.min(LR_MAX_TF, lr));
    }
    return Math.max(LR_MIN_NN, Math.min(LR_MAX_NN, lr));
  }

  function flipHeadFamily(family) {
    var map = {
      nn_policy: "nn_value",
      nn_value: "nn_policy",
      reinforce_policy: "reinforce_value",
      reinforce_value: "reinforce_policy",
      tfjs_policy: "tfjs_value",
      tfjs_value: "tfjs_policy"
    };
    return map[family] || null;
  }

  function sizesForHeadFlip(family, sizes) {
    var dest = flipHeadFamily(family);
    if (!dest || !sizes || sizes.length < 3) {
      return null;
    }
    var def = APP.defaultLayerSizes(dest);
    var hidden = sizes.slice(1, -1);
    return [def[0]].concat(hidden).concat([def[def.length - 1]]);
  }

  function mutateFrom(individual) {
    if (!isMorphable(individual.family)) {
      return [];
    }
    var kids = [];
    var wider = mutateWidth(individual.layerSizes, 1.5);
    var narrower = mutateWidth(individual.layerSizes, 2 / 3);
    var deeper = mutateDepth(individual.layerSizes, 1, individual.family);
    var shallower = mutateDepth(individual.layerSizes, -1, individual.family);
    var baseLr = typeof individual.learningRate === "number" ?
      individual.learningRate : APP.defaultLearningRate(individual.family);
    function pushKid(family, sizes, lr) {
      kids.push({
        family: family,
        layerSizes: sizes,
        learningRate: clampLr(family, lr),
        parentId: individual.id
      });
    }
    if (wider) {
      pushKid(individual.family, wider, baseLr);
    }
    if (narrower) {
      pushKid(individual.family, narrower, baseLr);
    }
    if (deeper) {
      pushKid(individual.family, deeper, baseLr);
    }
    if (shallower) {
      pushKid(individual.family, shallower, baseLr);
    }
    pushKid(individual.family, cloneSizes(individual.layerSizes), clampLr(individual.family, baseLr * 2));
    pushKid(individual.family, cloneSizes(individual.layerSizes), clampLr(individual.family, baseLr * 0.5));
    /* No policy/value head-swap mutants - those rarely keep useful skill. */
    return kids;
  }

  function initTrainSchedule(ind) {
    ind.trainBetweenTargetMs = TRAIN_BETWEEN_EVAL_MS;
    ind.trainBetweenLocked = false;
    ind.trainSinceEvalMs = 0;
  }

  /**
   * After an eval of duration evalMs: start at 50ms play between grades,
   * double each cycle until play target >= 4x eval, then hold at 4x eval.
   */
  function adaptTrainBetweenTarget(ind, evalMs) {
    var fourX = Math.max(TRAIN_BETWEEN_EVAL_MS, TRAIN_TO_EVAL_RATIO * Math.max(0, evalMs));
    if (ind.trainBetweenLocked) {
      ind.trainBetweenTargetMs = fourX;
      return;
    }
    if (ind.trainBetweenTargetMs >= fourX) {
      ind.trainBetweenTargetMs = fourX;
      ind.trainBetweenLocked = true;
      return;
    }
    ind.trainBetweenTargetMs = Math.min(ind.trainBetweenTargetMs * 2, fourX);
    if (ind.trainBetweenTargetMs >= fourX) {
      ind.trainBetweenLocked = true;
    }
  }

  function ensureTrainSchedule(ind) {
    if (typeof ind.trainBetweenTargetMs !== "number") {
      ind.trainBetweenTargetMs = TRAIN_BETWEEN_EVAL_MS;
    }
    if (typeof ind.trainBetweenLocked !== "boolean") {
      ind.trainBetweenLocked = false;
    }
    if (typeof ind.trainSinceEvalMs !== "number") {
      ind.trainSinceEvalMs = 0;
    }
  }

  /** One train/evolve step; returns wall seconds spent. Optional peer for dual-learn. */
  function shapeTrainStep(agent, family, useSelfPlay, peerInd) {
    var t0 = performance.now();
    if (peerInd && peerInd.agent && APP.shapeTrainVsPeer) {
      APP.shapeTrainVsPeer(agent, family, peerInd.agent, peerInd.family, true);
    } else if (family === "neat" || family === "neat_value" || family === "genetic_menace") {
      APP.shapeEvolveOne(agent, family);
    } else if (useSelfPlay) {
      APP.shapeTrainVsSelf(agent, family, true);
    } else {
      APP.shapeTrainVsRandom(agent, family, true, true);
    }
    return (performance.now() - t0) / 1000;
  }

  function saveBestWeights(ind, wr) {
    ind.bestWinRate = wr;
    /* Deep clone so later training cannot mutate the saved snapshot. */
    ind.bestBlob = JSON.parse(JSON.stringify(APP.shapeAgentToDict(ind.agent, ind.family)));
  }

  function restoreBestWeights(ind) {
    if (!ind.bestBlob) {
      return false;
    }
    var restored = APP.shapeAgentFromDict(ind.bestBlob);
    if (!restored) {
      return false;
    }
    ind.agent = restored;
    if (typeof ind.learningRate === "number") {
      APP.applyShapeLearningRate(ind.agent, ind.family, ind.learningRate);
    }
    return true;
  }

  function wrBeats(wr) {
    return typeof wr === "number" && wr > BEAT_WR;
  }

  function ensureBeatStreak(ind) {
    if (typeof ind.beatStreak !== "number" || ind.beatStreak < 0) {
      ind.beatStreak = 0;
    }
  }

  /** Update streak from one eval WR. Returns new streak. */
  function noteBeatStreak(ind, wr) {
    ensureBeatStreak(ind);
    if (wrBeats(wr)) {
      ind.beatStreak = Math.min(BEAT_STREAK_NEED, ind.beatStreak + 1);
    } else {
      ind.beatStreak = 0;
    }
    return ind.beatStreak;
  }

  function streakComplete(ind) {
    ensureBeatStreak(ind);
    return ind.beatStreak >= BEAT_STREAK_NEED;
  }

  function isRandomMmLevel(level) {
    return typeof level !== "number" || level <= 0;
  }

  function formatMmTag(level) {
    if (isRandomMmLevel(level)) {
      return "Random";
    }
    var n = typeof level === "number" ? level : 0;
    return "MM" + n;
  }

  function isEvolveFamily(family) {
    return family === "neat" || family === "neat_value" || family === "genetic_menace";
  }

  /**
   * Mix for Shape Search train steps.
   * Not-yet-better-than-MM: nemesis / MM / frozen / random.
   * Better-than-MM: peer (other beaters) / self / frozen / nemesis / MM / random.
   */
  function pickShapeTrainKind(flags, roll) {
    var r = typeof roll === "number" ? roll : Math.random();
    var beats = !!flags.beatsMm;
    var hasPeer = !!flags.hasPeer;
    var hasFrozen = !!flags.hasFrozen;
    var hasNemesis = !!flags.hasNemesis;
    var canSelf = flags.canSelf !== false;
    if (beats) {
      if (hasPeer && r < 0.30) {
        return "peer";
      }
      if (canSelf && r < 0.60) {
        return "self";
      }
      if (hasFrozen && r < 0.75) {
        return "frozen";
      }
      if (hasNemesis && r < 0.85) {
        return "nemesis";
      }
      if (r < 0.95) {
        return "mm";
      }
      return "random";
    }
    if (hasNemesis && r < 0.25) {
      return "nemesis";
    }
    if (r < 0.55) {
      return "mm";
    }
    if (hasFrozen && r < 0.75) {
      return "frozen";
    }
    return "random";
  }

  function ensureNemesis(ind) {
    if (!ind.nemesisSeeds || !Array.isArray(ind.nemesisSeeds)) {
      ind.nemesisSeeds = [];
    }
  }

  function addNemesisSeed(ind, seed, level) {
    ensureNemesis(ind);
    if (typeof seed !== "number") {
      return;
    }
    var lv = typeof level === "number" ? level : 1;
    var i;
    for (i = 0; i < ind.nemesisSeeds.length; i += 1) {
      if (ind.nemesisSeeds[i].seed === seed && ind.nemesisSeeds[i].level === lv) {
        return;
      }
    }
    ind.nemesisSeeds.push({ seed: seed, level: lv });
    if (ind.nemesisSeeds.length > NEMESIS_MAX) {
      ind.nemesisSeeds.shift();
    }
  }

  function pickNemesisSeed(ind) {
    ensureNemesis(ind);
    if (!ind.nemesisSeeds.length) {
      return null;
    }
    return ind.nemesisSeeds[Math.floor(Math.random() * ind.nemesisSeeds.length)];
  }

  function loadFrozenAgent(ind) {
    if (!ind.bestBlob) {
      return null;
    }
    var agent = APP.shapeAgentFromDict(ind.bestBlob);
    if (!agent) {
      return null;
    }
    if (typeof ind.learningRate === "number") {
      APP.applyShapeLearningRate(agent, ind.family, ind.learningRate);
    }
    return agent;
  }

  function pointsWinRate(ev) {
    if (!ev) {
      return 0;
    }
    var g = typeof ev.games === "number" && ev.games > 0 ? ev.games :
      (ev.wins || 0) + (ev.draws || 0) + (ev.losses || 0);
    if (g <= 0) {
      return 0;
    }
    return APP.shapeEvalPoints(ev) / g;
  }

  function ensureEvalMeta(ind) {
    if (typeof ind.evalGamesTarget !== "number" || ind.evalGamesTarget < EVAL_GAMES) {
      ind.evalGamesTarget = EVAL_GAMES;
    }
    if (ind.bestWinRate === undefined) {
      ind.bestWinRate = null;
    }
    if (ind.bestBlob === undefined) {
      ind.bestBlob = null;
    }
    if (ind.lastEvalWinRate === undefined) {
      ind.lastEvalWinRate = null;
    }
    if (!ind.peerPlayCounts || typeof ind.peerPlayCounts !== "object") {
      ind.peerPlayCounts = {};
    }
    ensureNemesis(ind);
  }

  /**
   * Warmup: one train-only step. Returns dt seconds, or null if not in warmup.
   */
  function stepWarmup(ind, useSelfPlay) {
    if (!(ind.warmupLeftMs > 0)) {
      return null;
    }
    var w0 = performance.now();
    var dt = shapeTrainStep(ind.agent, ind.family, useSelfPlay);
    ind.warmupLeftMs = Math.max(0, ind.warmupLeftMs - (performance.now() - w0));
    if (ind.warmupLeftMs <= 0) {
      ind.trainSinceEvalMs = 0;
    }
    return dt;
  }

  function clampBotSearchOffset(offset) {
    var off = typeof offset === "number" && isFinite(offset) ? Math.floor(offset) : 0;
    if (off > 0) {
      return 0;
    }
    if (off < -10) {
      return -10;
    }
    return off;
  }

  function createIndividual(spec, id) {
    var family = spec.family;
    var sizes = cloneSizes(spec.layerSizes);
    var lr = typeof spec.learningRate === "number" ?
      spec.learningRate : APP.defaultLearningRate(family);
    var agent = spec.agent || APP.createShapeAgent(family, sizes, lr);
    if (spec.agent && typeof lr === "number") {
      APP.applyShapeLearningRate(agent, family, lr);
    }
    var ind = {
      id: id || uid("bot"),
      family: family,
      layerSizes: sizes,
      learningRate: lr,
      label: familyLabel(family, sizes),
      agent: agent,
      secondsUsed: 0,
      evalSeconds: 0,
      beaten: false,
      beatStreak: 0,
      secondsToBeat: null,
      passedMm1: false,
      lastEvalWins: null,
      lastEvalDraws: null,
      lastEvalGames: null,
      lastEvalWinRate: null,
      evalGamesTarget: EVAL_GAMES,
      bestWinRate: null,
      bestBlob: null,
      peerPlayCounts: {},
      nemesisSeeds: [],
      warmupLeftMs: WARMUP_MS,
      live: true
    };
    initTrainSchedule(ind);
    assignNeatIdentity(ind);
    return ind;
  }

  function resetIndividualProgress(ind) {
    ind.secondsUsed = 0;
    ind.evalSeconds = 0;
    ind.beaten = false;
    ind.beatStreak = 0;
    ind.secondsToBeat = null;
    ind.passedMm1 = false;
    ind.lastEvalWins = null;
    ind.lastEvalDraws = null;
    ind.lastEvalGames = null;
    ind.lastEvalWinRate = null;
    /* Each round restarts short grades; bestBlob / bestWinRate stay permanent. */
    ind.evalGamesTarget = EVAL_GAMES;
    ind.nemesisSeeds = [];
    ensureEvalMeta(ind);
    ind.warmupLeftMs = WARMUP_MS;
    initTrainSchedule(ind);
  }

  /**
   * NEAT / NEAT-value parents each get MAX_NEAT_COPIES-1 weight-clone seats.
   * Morph parents get mutants instead (via mutateFrom); other families neither.
   * Callers pass only the top MAX_MUTANT_PARENTS finishers that remain after dedupe.
   */
  function cloneNeatSurvivors(nextPop, survivors) {
    var extras = Math.max(0, MAX_NEAT_COPIES - 1);
    var i;
    for (i = 0; i < survivors.length; i += 1) {
      var parent = survivors[i];
      if (!isNeatFamily(parent.family)) {
        continue;
      }
      var c;
      for (c = 0; c < extras; c += 1) {
        var agent = APP.shapeAgentFromDict(
          APP.shapeAgentToDict(parent.agent, parent.family)
        );
        nextPop.push(createIndividual({
          family: parent.family,
          layerSizes: null,
          agent: agent
        }));
      }
    }
  }

  function survivorLimitForRound(/* round */) {
    return MAX_SURVIVORS;
  }

  /** Top tournament finishers (by rank order) that are still in the survivor list. */
  function pickMutantParents(rankedTop, survivors, limit) {
    var n = typeof limit === "number" ? limit : MAX_MUTANT_PARENTS;
    var byId = {};
    var i;
    for (i = 0; i < survivors.length; i += 1) {
      byId[survivors[i].id] = survivors[i];
    }
    var parents = [];
    for (i = 0; i < rankedTop.length && parents.length < n; i += 1) {
      var bot = byId[rankedTop[i].id];
      if (bot) {
        parents.push(bot);
      }
    }
    return parents;
  }

  /** Copy overlapping weight/bias slices (Lamarckian size/head mutants). */
  function copyOverlapNet(srcNet, dstNet) {
    if (!srcNet || !dstNet || !srcNet.weights || !dstNet.weights) {
      return false;
    }
    var layers = Math.min(srcNet.weights.length, dstNet.weights.length);
    var L;
    for (L = 0; L < layers; L += 1) {
      var rows = Math.min(srcNet.weights[L].length, dstNet.weights[L].length);
      var r;
      for (r = 0; r < rows; r += 1) {
        var cols = Math.min(srcNet.weights[L][r].length, dstNet.weights[L][r].length);
        var c;
        for (c = 0; c < cols; c += 1) {
          dstNet.weights[L][r][c] = srcNet.weights[L][r][c];
        }
        if (srcNet.biases && dstNet.biases &&
            typeof srcNet.biases[L][r] === "number") {
          dstNet.biases[L][r] = srcNet.biases[L][r];
        }
      }
    }
    return true;
  }

  function cloneAgentFromParent(parent, family, layerSizes, learningRate) {
    var sameArch = parent.family === family &&
      ((parent.layerSizes === null && layerSizes === null) ||
        (parent.layerSizes && layerSizes &&
          parent.layerSizes.join(",") === layerSizes.join(",")));
    var agent;
    if (sameArch && parent.agent) {
      agent = APP.shapeAgentFromDict(APP.shapeAgentToDict(parent.agent, parent.family));
    } else {
      agent = APP.createShapeAgent(family, layerSizes, learningRate);
      if (parent.agent && parent.agent.net && agent.net) {
        copyOverlapNet(parent.agent.net, agent.net);
      }
    }
    if (typeof learningRate === "number") {
      APP.applyShapeLearningRate(agent, family, learningRate);
    }
    return agent;
  }

  function snapshotIndividual(ind) {
    return {
      id: ind.id,
      family: ind.family,
      layerSizes: cloneSizes(ind.layerSizes),
      learningRate: ind.learningRate,
      label: ind.label,
      blob: APP.shapeAgentToDict(ind.agent, ind.family)
    };
  }

  function individualFromSnapshot(snap) {
    var agent = APP.shapeAgentFromDict(snap.blob);
    if (typeof snap.learningRate === "number") {
      APP.applyShapeLearningRate(agent, snap.family, snap.learningRate);
    }
    var ind = {
      id: snap.id,
      family: snap.family,
      layerSizes: cloneSizes(snap.layerSizes),
      learningRate: snap.learningRate,
      label: snap.label || familyLabel(snap.family, snap.layerSizes),
      agent: agent,
      secondsUsed: 0,
      evalSeconds: 0,
      beaten: false,
      beatStreak: 0,
      secondsToBeat: null,
      passedMm1: false,
      lastEvalWins: null,
      lastEvalDraws: null,
      lastEvalGames: null,
      warmupLeftMs: WARMUP_MS,
      live: false
    };
    initTrainSchedule(ind);
    return ind;
  }

  /* --- Round-robin tournament among free agents --- */
  var TOURN_UI_MIN_MS = 500;

  function playTournamentMatch(a, b, markA, moveBudgetMs, botMaxDepth) {
    var board = G.emptyBoard();
    var current = C.X;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move;
      var entry = current === markA ? a : b;
      if (entry.family === "minimax") {
        move = entry.agent.chooseMove(board, current, moveBudgetMs);
      } else if (entry.family === "mcts") {
        move = entry.agent.chooseMove(board, current, false, moveBudgetMs);
      } else if (entry.family === "random") {
        move = entry.agent.chooseMove(board, current);
      } else {
        move = APP.shapeAgentChoose(
          entry.agent, entry.family, board, current, false, moveBudgetMs, botMaxDepth
        );
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    return G.findWinner(board);
  }

  function isMinimaxEntry(entry) {
    return !!entry && (entry.id === "minimax" || entry.family === "minimax");
  }

  function isRoundBaselineEntry(entry) {
    return isMinimaxEntry(entry) ||
      (!!entry && (entry.id === "random" || entry.family === "random"));
  }

  function noteH2hWin(session, winnerId, loserId) {
    if (!session.winsAgainst[winnerId]) {
      session.winsAgainst[winnerId] = {};
    }
    session.winsAgainst[winnerId][loserId] =
      (session.winsAgainst[winnerId][loserId] || 0) + 1;
  }

  function applyMatchResult(session, a, b, markA, winner) {
    if (!winner) {
      session.points[a.id] += 0.5;
      session.points[b.id] += 0.5;
      session.draws[a.id] += 1;
      session.draws[b.id] += 1;
    } else if (winner === markA) {
      session.points[a.id] += 1;
      session.wins[a.id] += 1;
      session.losses[b.id] += 1;
      noteH2hWin(session, a.id, b.id);
      if (isRoundBaselineEntry(b) && !isRoundBaselineEntry(a)) {
        session.winsVsMinimax[a.id] += 1;
      }
    } else {
      session.points[b.id] += 1;
      session.wins[b.id] += 1;
      session.losses[a.id] += 1;
      noteH2hWin(session, b.id, a.id);
      if (isRoundBaselineEntry(a) && !isRoundBaselineEntry(b)) {
        session.winsVsMinimax[b.id] += 1;
      }
    }
  }

  /** Suffix for Latest/Current tournament lines. */
  function formatBeatMinimaxSuffix(row) {
    if (!row || isRoundBaselineEntry(row)) {
      return "";
    }
    var n = row.winsVsMinimax || 0;
    if (n < 1) {
      return "";
    }
    var vs = row.vsBaselineName || "Minimax";
    return " | beat " + vs + " in " + n + " game" + (n === 1 ? "" : "s");
  }

  function createRoundRobinSession(entries, moveBudgetMs, botMaxDepth) {
    var points = {};
    var wins = {};
    var draws = {};
    var losses = {};
    var winsVsMinimax = {};
    var winsAgainst = {};
    var i;
    for (i = 0; i < entries.length; i += 1) {
      points[entries[i].id] = 0;
      wins[entries[i].id] = 0;
      draws[entries[i].id] = 0;
      losses[entries[i].id] = 0;
      winsVsMinimax[entries[i].id] = 0;
      winsAgainst[entries[i].id] = {};
    }
    var n = entries.length;
    var pairCount = n < 2 ? 0 : (n * (n - 1)) / 2;
    return {
      entries: entries,
      moveBudgetMs: moveBudgetMs,
      botMaxDepth: typeof botMaxDepth === "number" && botMaxDepth > 0 ? botMaxDepth : null,
      points: points,
      wins: wins,
      draws: draws,
      losses: losses,
      winsVsMinimax: winsVsMinimax,
      winsAgainst: winsAgainst,
      vsBaselineName: entries.some(function (e) {
        return e.id === "random" || e.family === "random";
      }) ? "Random" : "Minimax",
      i: 0,
      j: 1,
      leg: 0,
      startedAt: performance.now(),
      gamesDone: 0,
      gamesTotal: pairCount * 2,
      done: n < 2
    };
  }

  function roundRobinStandings(session) {
    var inProgress = !session.done;
    function gamesPlayed(id) {
      return session.wins[id] + session.draws[id] + session.losses[id];
    }
    function winPct(id) {
      var g = gamesPlayed(id);
      return g > 0 ? session.wins[id] / g : 0;
    }
    var ranked = session.entries.slice().sort(function (x, y) {
      if (inProgress) {
        var dPct = winPct(y.id) - winPct(x.id);
        if (Math.abs(dPct) > 1e-12) {
          return dPct;
        }
        var dGames = gamesPlayed(y.id) - gamesPlayed(x.id);
        if (dGames !== 0) {
          return dGames;
        }
      }
      return session.points[y.id] - session.points[x.id];
    });
    return {
      durationMs: performance.now() - session.startedAt,
      moveBudgetMs: session.moveBudgetMs,
      gamesDone: session.gamesDone,
      gamesTotal: session.gamesTotal,
      inProgress: inProgress,
      winsAgainst: session.winsAgainst || {},
      ranked: ranked.map(function (e) {
        var g = gamesPlayed(e.id);
        return {
          id: e.id,
          label: e.label,
          family: e.family,
          points: session.points[e.id],
          wins: session.wins[e.id],
          draws: session.draws[e.id],
          losses: session.losses[e.id],
          winsVsMinimax: session.winsVsMinimax[e.id] || 0,
          vsBaselineName: session.vsBaselineName || "Minimax",
          games: g,
          winPct: g > 0 ? session.wins[e.id] / g : 0
        };
      })
    };
  }

  /**
   * Learned opponent ids that scored strictly more tournament points than
   * `traineeId`. Baselines excluded.
   */
  function peersWhoBeatTrainee(tourn, traineeId) {
    var out = [];
    if (!tourn || !tourn.ranked || !traineeId) {
      return out;
    }
    var ranked = tourn.ranked;
    var traineePts = null;
    var i;
    for (i = 0; i < ranked.length; i += 1) {
      if (ranked[i].id === traineeId) {
        traineePts = ranked[i].points;
        break;
      }
    }
    if (typeof traineePts !== "number") {
      return out;
    }
    for (i = 0; i < ranked.length; i += 1) {
      var row = ranked[i];
      if (!row || row.id === traineeId || isBaselineRank(row)) {
        continue;
      }
      if (typeof row.points === "number" && row.points > traineePts) {
        out.push(row.id);
      }
    }
    return out;
  }

  /** Play one game; returns true when the whole RR is finished. */
  function stepRoundRobinSession(session) {
    if (session.done) {
      return true;
    }
    var entries = session.entries;
    if (session.i >= entries.length - 1) {
      session.done = true;
      return true;
    }
    var a = entries[session.i];
    var b = entries[session.j];
    var markA = session.leg === 0 ? C.X : C.O;
    var winner = playTournamentMatch(a, b, markA, session.moveBudgetMs, session.botMaxDepth);
    applyMatchResult(session, a, b, markA, winner);
    session.gamesDone += 1;
    session.leg += 1;
    if (session.leg > 1) {
      session.leg = 0;
      session.j += 1;
      if (session.j >= entries.length) {
        session.i += 1;
        session.j = session.i + 1;
        if (session.i >= entries.length - 1) {
          session.done = true;
        }
      }
    }
    return session.done;
  }

  function runRoundRobin(entries, moveBudgetMs, botMaxDepth) {
    var session = createRoundRobinSession(entries, moveBudgetMs, botMaxDepth);
    while (!session.done) {
      stepRoundRobinSession(session);
    }
    return roundRobinStandings(session);
  }

  function isBaselineRank(r) {
    return !!r && (r.id === "minimax" || r.id === "mcts" || r.id === "random" ||
      r.family === "minimax" || r.family === "mcts" || r.family === "random");
  }

  function topLearnedRanks(ranked, limit) {
    var n = typeof limit === "number" ? limit : MAX_SURVIVORS_FIRST;
    return (ranked || []).filter(function (r) {
      return !isBaselineRank(r);
    }).slice(0, n);
  }

  /**
   * History rows: up to `learnedLimit` learned bots (default 10), plus Minimax/MCTS
   * when they place in the overall top HISTORY_BASELINE_TOP_N. Order follows standings.
   * Example: Minimax #1 and MCTS #10 -> top 10 learned + Minimax (MCTS omitted).
   */
  function historyTopRanks(ranked, learnedLimit) {
    var lim = typeof learnedLimit === "number" ? learnedLimit : HISTORY_LEARNED_LIMIT;
    var rows = ranked || [];
    var includeBase = {};
    var i;
    for (i = 0; i < Math.min(HISTORY_BASELINE_TOP_N, rows.length); i += 1) {
      if (isBaselineRank(rows[i])) {
        includeBase[rows[i].id] = true;
      }
    }
    var out = [];
    var learned = 0;
    for (i = 0; i < rows.length; i += 1) {
      var r = rows[i];
      if (isBaselineRank(r)) {
        if (includeBase[r.id]) {
          out.push(r);
        }
      } else if (learned < lim) {
        out.push(r);
        learned += 1;
      }
    }
    return out;
  }

  function makeBaselineEntries(mmLevel, moveBudgetMs) {
    var mctsAgent = {
      chooseMove: function (board, player, explore, budgetMs) {
        /* Lightweight MCTS-like: random playouts until budget. */
        var moves = G.legalMoves(board);
        if (!moves.length) {
          return null;
        }
        if (moves.length === 1) {
          return moves[0];
        }
        var deadline = performance.now() + (budgetMs || moveBudgetMs || 5);
        var best = moves[0];
        var bestScore = -1;
        var m;
        for (m = 0; m < moves.length; m += 1) {
          var wins = 0;
          var trials = 0;
          while (performance.now() < deadline) {
            var b = G.applyMove(board, player, moves[m]);
            var cur = G.other(player);
            var guard = 0;
            while (!G.findWinner(b) && !G.isDraw(b) && guard < 42) {
              var mv = G.legalMoves(b);
              if (!mv.length) {
                break;
              }
              b = G.applyMove(b, cur, mv[Math.floor(Math.random() * mv.length)]);
              cur = G.other(cur);
              guard += 1;
            }
            if (G.findWinner(b) === player) {
              wins += 1;
            }
            trials += 1;
            if (trials > 40) {
              break;
            }
          }
          var score = trials ? wins / trials : 0;
          if (score > bestScore) {
            bestScore = score;
            best = moves[m];
          }
        }
        return best;
      }
    };
    if (isRandomMmLevel(mmLevel)) {
      var randomAgent = {
        chooseMove: function (board) {
          var moves = G.legalMoves(board);
          if (!moves.length) {
            return null;
          }
          return moves[Math.floor(Math.random() * moves.length)];
        }
      };
      return [
        { id: "random", family: "random", label: "Random", agent: randomAgent },
        { id: "mcts", family: "mcts", label: "MCTS", agent: mctsAgent }
      ];
    }
    var mmAgent = new (function MinimaxShim() {
      this.lastDepth = 0;
      this.chooseMove = function (board, player, budgetMs) {
        var opp = new M.FractionalMinimaxOpponent(M.normalizeLevel(mmLevel));
        if (typeof budgetMs === "number" && budgetMs > 0) {
          var deadline = performance.now() + budgetMs;
          var timed = M.chooseMoveUntilDeadline(board, player, deadline, Math.random.bind(Math));
          this.lastDepth = timed.depth || 0;
          return timed.move;
        }
        this.lastDepth = 0;
        return opp.chooseMove(board, player);
      };
    })();
    return [
      { id: "minimax", family: "minimax", label: "Minimax", agent: mmAgent },
      { id: "mcts", family: "mcts", label: "MCTS", agent: mctsAgent }
    ];
  }

  /* --- Shape Search controller --- */
  function ShapeSearch() {
    this.round = 1;
    this.mmLevel = 0;
    /** 0 = Match Minimax depth; -1..-10 = shallower learner search. */
    this.botSearchOffset = 0;
    this.population = [];
    this.paused = false;
    this.userPaused = false;
    this.bestBeatSec = null;
    this.lastTournament = null;
    this.tournamentSnapshots = [];
    this.history = [];
    this.statusLine = "Shape Search ready";
    this._currentId = null;
    this._dwellId = null;
    this._sliceTarget = null;
    this._leaderAdvanceTarget = null;
    this._sweepIndex = -1;
    this.liveTourn = null;
    this._tournLastUiAt = 0;
    this._tournUiDirty = false;
    this.initPopulation();
  }

  ShapeSearch.prototype.botMaxDepth = function (mmLevel) {
    var level = typeof mmLevel === "number" ? mmLevel : this.mmLevel;
    return APP.botSearchDepthForLevel(level, this.botSearchOffset);
  };

  ShapeSearch.prototype.setBotSearchOffset = function (offset) {
    this.botSearchOffset = clampBotSearchOffset(offset);
  };

  ShapeSearch.prototype.initPopulation = function () {
    var specs = gen0Specs();
    var i;
    this.population = [];
    for (i = 0; i < specs.length; i += 1) {
      this.population.push(createIndividual(specs[i]));
    }
  };

  ShapeSearch.prototype.setPaused = function (paused, fromUser) {
    this.paused = !!paused;
    if (fromUser) {
      this.userPaused = !!paused;
    }
  };

  ShapeSearch.prototype.resumeIfAllowed = function () {
    if (!this.userPaused) {
      this.paused = false;
    }
  };

  ShapeSearch.prototype.minTrainSec = function () {
    return isRandomMmLevel(this.mmLevel) ? MIN_TRAIN_SEC_RANDOM : MIN_TRAIN_SEC;
  };

  /**
   * UI floor: R1 Random = 3s, later rounds = 10s.
   * Round end is gated by streak (hasBeater) + all bots at/above that floor.
   */
  ShapeSearch.prototype.maxTrainSec = function () {
    return this.minTrainSec();
  };

  ShapeSearch.prototype.hasBeater = function () {
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (this.population[i].beaten) {
        return true;
      }
    }
    return false;
  };

  /** Done training this round: at least min train_s, and some bot has 10/10 streak. */
  ShapeSearch.prototype.botTrainComplete = function (ind) {
    if (!ind || ind.secondsUsed < this.minTrainSec()) {
      return false;
    }
    return this.hasBeater();
  };

  /**
   * Lowest toBeat_s among bots that have one (first WR>50% train time; kept on dips).
   */
  ShapeSearch.prototype.recomputeBestBeatSec = function () {
    var best = null;
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      var ind = this.population[i];
      if (typeof ind.secondsToBeat === "number") {
        if (best === null || ind.secondsToBeat < best) {
          best = ind.secondsToBeat;
        }
      }
    }
    this.bestBeatSec = best;
    return best;
  };

  /**
   * First 100-game WR>50% train_s this round. Never updates again until round reset.
   */
  ShapeSearch.prototype.noteToBeatSec = function (ind) {
    if (!ind || typeof ind.secondsUsed !== "number") {
      return;
    }
    if (ind.secondsToBeat !== null) {
      return;
    }
    ind.secondsToBeat = ind.secondsUsed;
    this.recomputeBestBeatSec();
  };

  ShapeSearch.prototype.clearBeatForDip = function (ind) {
    if (!ind || !ind.beaten) {
      return false;
    }
    ind.beaten = false;
    ind.beatStreak = 0;
    /* Keep secondsToBeat until the next Shape Search round. */
    this.recomputeBestBeatSec();
    return true;
  };

  /** Soft RR among bots whose last eval WR beats current MM (>50%). */
  ShapeSearch.prototype.pickPeerFor = function (ind) {
    ensureEvalMeta(ind);
    if (typeof ind.lastEvalWinRate !== "number" || !wrBeats(ind.lastEvalWinRate)) {
      return null;
    }
    var peers = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      var p = this.population[i];
      if (p.id === ind.id) {
        continue;
      }
      ensureEvalMeta(p);
      if (typeof p.lastEvalWinRate === "number" && wrBeats(p.lastEvalWinRate)) {
        peers.push(p);
      }
    }
    if (!peers.length) {
      return null;
    }
    peers.sort(function (a, b) {
      var ca = ind.peerPlayCounts[a.id] || 0;
      var cb = ind.peerPlayCounts[b.id] || 0;
      if (ca !== cb) {
        return ca - cb;
      }
      /* Prefer peers behind on train clock so popular beaters do not race ahead. */
      if (a.secondsUsed !== b.secondsUsed) {
        return a.secondsUsed - b.secondsUsed;
      }
      return idAge(a.id) - idAge(b.id);
    });
    return peers[0];
  };

  ShapeSearch.prototype.notePeerPlay = function (a, b) {
    ensureEvalMeta(a);
    ensureEvalMeta(b);
    a.peerPlayCounts[b.id] = (a.peerPlayCounts[b.id] || 0) + 1;
    b.peerPlayCounts[a.id] = (b.peerPlayCounts[a.id] || 0) + 1;
  };

  ShapeSearch.prototype.botDueForEval = function (ind) {
    if (!ind) {
      return false;
    }
    ensureEvalMeta(ind);
    ensureTrainSchedule(ind);
    if (ind.warmupLeftMs > 0) {
      return false;
    }
    return ind.trainSinceEvalMs >= ind.trainBetweenTargetMs;
  };

  ShapeSearch.prototype.botsNeedingEval = function () {
    var out = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (this.botDueForEval(this.population[i])) {
        out.push(this.population[i]);
      }
    }
    return out;
  };

  ShapeSearch.prototype.evalContext = function (ind) {
    var evalLevel = (this.round > 1 && !ind.passedMm1 && this.mmLevel >= 1) ?
      1 : this.mmLevel;
    var onRoundMm = !(this.round > 1 && !ind.passedMm1 && evalLevel === 1);
    return { evalLevel: evalLevel, onRoundMm: onRoundMm };
  };

  ShapeSearch.prototype.evalIfDue = function (ind) {
    if (!this.botDueForEval(ind)) {
      return false;
    }
    var ctx = this.evalContext(ind);
    this.runIndividualEval(ind, ctx.evalLevel, ctx.onRoundMm);
    return true;
  };

  /**
   * Curriculum train step at the same ply cap as eval.
   * Not-better-than-MM: mix MM, nemesis replay, frozen-best, random.
   * Better-than-MM: mix self, other beaters, frozen-best, plus a little MM/random.
   */
  ShapeSearch.prototype.trainIndividual = function (ind, useSelfPlay) {
    ensureEvalMeta(ind);
    var ctx = this.evalContext(ind);
    var depth = this.botMaxDepth(ctx.evalLevel);
    var mmLevel = ctx.evalLevel;
    var t0 = performance.now();
    if (isEvolveFamily(ind.family)) {
      APP.shapeEvolveOne(ind.agent, ind.family);
      var evo = (performance.now() - t0) / 1000;
      ind.secondsUsed += evo;
      ind.trainSinceEvalMs += evo * 1000;
      return evo;
    }
    var peer = wrBeats(ind.lastEvalWinRate) ? this.pickPeerFor(ind) : null;
    var frozen = loadFrozenAgent(ind);
    var kind = pickShapeTrainKind({
      beatsMm: wrBeats(ind.lastEvalWinRate),
      hasPeer: !!peer,
      hasFrozen: !!frozen,
      hasNemesis: !isRandomMmLevel(mmLevel) && !!(ind.nemesisSeeds && ind.nemesisSeeds.length),
      canSelf: true
    });
    if (isRandomMmLevel(mmLevel) && (kind === "mm" || kind === "nemesis")) {
      kind = "random";
    }
    if (kind === "peer" && peer) {
      APP.shapeTrainVsPeer(
        ind.agent, ind.family, peer.agent, peer.family, true, depth, true
      );
      this.notePeerPlay(ind, peer);
      var peerDt = (performance.now() - t0) / 1000;
      ind.secondsUsed += peerDt;
      ind.trainSinceEvalMs += peerDt * 1000;
      peer.secondsUsed += peerDt;
      peer.trainSinceEvalMs += peerDt * 1000;
      this.evalIfDue(peer);
      return peerDt;
    }
    if (kind === "self" || (kind === "peer" && !peer)) {
      APP.shapeTrainVsSelf(ind.agent, ind.family, true, depth);
    } else if (kind === "frozen" && frozen) {
      APP.shapeTrainVsPeer(
        ind.agent, ind.family, frozen, ind.family, true, depth, false
      );
    } else if (kind === "nemesis") {
      var entry = pickNemesisSeed(ind);
      if (entry) {
        APP.shapeTrainVsMinimax(
          ind.agent, ind.family, entry.level, false, true, depth, entry.seed
        );
      } else {
        APP.shapeTrainVsMinimax(
          ind.agent, ind.family, mmLevel, true, true, depth, null
        );
      }
    } else if (kind === "mm") {
      var packed = APP.shapeTrainVsMinimax(
        ind.agent, ind.family, mmLevel, true, true, depth, null
      );
      if (packed && packed.result === "loss") {
        addNemesisSeed(ind, packed.seed, mmLevel);
      }
    } else {
      APP.shapeTrainVsRandom(ind.agent, ind.family, true, true, depth);
    }
    var dt = (performance.now() - t0) / 1000;
    ind.secondsUsed += dt;
    ind.trainSinceEvalMs += dt * 1000;
    return dt;
  };

  /** Eval a frozen blob without replacing the live agent. */
  function evalSavedBlob(ind, evalLevel, botDepth, games) {
    if (!ind.bestBlob) {
      return null;
    }
    var agent = APP.shapeAgentFromDict(ind.bestBlob);
    if (!agent) {
      return null;
    }
    if (typeof ind.learningRate === "number") {
      APP.applyShapeLearningRate(agent, ind.family, ind.learningRate);
    }
    var n = typeof games === "number" && games > 0 ? games : EVAL_GAMES_LONG;
    return APP.shapeEvalVsMinimaxTimed(agent, ind.family, evalLevel, n, botDepth, null);
  }

  /**
   * Keep-best: always save on a new high WR (cheap).
   * Expensive 1-ply playoffs (tie / >=10pt dip) only in legacy 100-game mode.
   */
  function applyKeepBestAfterEval(ind, ev, evalLevel, botDepth) {
    if (ind.bestWinRate === null || ev.winRate > ind.bestWinRate) {
      saveBestWeights(ind, ev.winRate);
      return "saved best";
    }
    if (ind.evalGamesTarget < EVAL_GAMES_LONG) {
      return null;
    }
    var tied = ev.winRate >= ind.bestWinRate;
    var dipped = ev.winRate <= ind.bestWinRate - BEST_WR_DIP;
    if (!tied && !dipped) {
      return null;
    }
    if (!ind.bestBlob || !APP.shapeCompareKeepBest) {
      if (tied) {
        saveBestWeights(ind, ev.winRate);
        return "saved tie";
      }
      return null;
    }
    var savedAgent = APP.shapeAgentFromDict(ind.bestBlob);
    if (!savedAgent) {
      return null;
    }
    if (typeof ind.learningRate === "number") {
      APP.applyShapeLearningRate(savedAgent, ind.family, ind.learningRate);
    }
    var who = APP.shapeCompareKeepBest(ind.agent, ind.family, savedAgent);
    if (who === "saved") {
      if (restoreBestWeights(ind)) {
        return "restored best";
      }
      return null;
    }
    saveBestWeights(ind, ev.winRate);
    return tied ? "saved tie" : "kept current";
  }

  function runBotEval(ind, evalLevel, botDepth) {
    ensureEvalMeta(ind);
    /*
     * Short grades (20) until WR>50%, then extend that eval to 100 and stay on
     * 100-game evals for the rest of the Shape Search round.
     */
    var startGames = ind.evalGamesTarget >= EVAL_GAMES_LONG ?
      EVAL_GAMES_LONG : EVAL_GAMES;
    var opts = null;
    if (startGames < EVAL_GAMES_LONG) {
      opts = { extendTo: EVAL_GAMES_LONG, extendIfWr: BEAT_WR };
    }
    var tEval0 = performance.now();
    var ev;
    if (isRandomMmLevel(evalLevel)) {
      ev = APP.shapeEvalVsRandom(
        ind.agent, ind.family, startGames, botDepth, opts
      );
    } else {
      ev = APP.shapeEvalVsMinimaxTimed(
        ind.agent, ind.family, evalLevel, startGames, botDepth, opts
      );
    }
    var evalMs = performance.now() - tEval0;
    ind.evalSeconds += evalMs / 1000;
    ev.winRate = pointsWinRate(ev);
    ind.lastEvalWins = ev.wins;
    ind.lastEvalDraws = ev.draws;
    ind.lastEvalGames = ev.games;
    ind.lastEvalWinRate = ev.winRate;
    if (ev.extended || ev.games >= EVAL_GAMES_LONG) {
      ind.evalGamesTarget = EVAL_GAMES_LONG;
    }
    adaptTrainBetweenTarget(ind, evalMs);
    ind.trainSinceEvalMs = 0;
    var tKeep0 = performance.now();
    var weightNote = applyKeepBestAfterEval(ind, ev, evalLevel, botDepth);
    ind.evalSeconds += (performance.now() - tKeep0) / 1000;
    return { ev: ev, evalMs: evalMs, weightNote: weightNote };
  }

  /** Run MM eval; update mmWR, streak, best weights, beaten at streak need. */
  ShapeSearch.prototype.runIndividualEval = function (ind, evalLevel, onRoundMm) {
    var packed = runBotEval(ind, evalLevel, this.botMaxDepth(evalLevel));
    var ev = packed.ev;
    noteBeatStreak(ind, ev.winRate);
    /* toBeat_s: first 100-game WR>50% vs whatever opponent this eval used. */
    if (ev.games >= EVAL_GAMES_LONG && wrBeats(ev.winRate)) {
      this.noteToBeatSec(ind);
    }
    if (packed.weightNote === "restored best") {
      this.statusLine = ind.label + " restored best (playoff won)";
    } else if (packed.weightNote === "kept current") {
      this.statusLine = ind.label + " kept current (playoff won)";
    } else if (packed.weightNote === "saved tie") {
      this.statusLine = ind.label + " saved tied best";
    }
    if (onRoundMm) {
      var minT = this.minTrainSec();
      if (streakComplete(ind)) {
        if (!ind.beaten) {
          ind.beaten = true;
          this.recomputeBestBeatSec();
          this.statusLine = ind.label + " beat " + formatMmTag(this.mmLevel) +
            " (" + BEAT_STREAK_NEED + "/" + BEAT_STREAK_NEED + ") in " +
            (typeof ind.secondsToBeat === "number" ?
              ind.secondsToBeat.toFixed(2) : ind.secondsUsed.toFixed(2)) +
            "s train (" +
            (ev.winRate * 100).toFixed(1) + "% WR)" +
            " | round ends when all bots >= " + minT + "s";
        }
      } else if (ind.beaten) {
        this.clearBeatForDip(ind);
        this.statusLine = ind.label + " streak reset vs " +
          formatMmTag(this.mmLevel) + " (" + ind.beatStreak + "/" + BEAT_STREAK_NEED + ")" +
          (this.hasBeater()
            ? (" | min train " + minT + "s")
            : " | waiting for a 10/10 beater");
      } else {
        this.statusLine = ind.label + " streak " + ind.beatStreak + "/" + BEAT_STREAK_NEED +
          " vs " + formatMmTag(this.mmLevel) +
          " (" + (ev.winRate * 100).toFixed(1) + "% WR)";
      }
    } else if (wrBeats(ev.winRate)) {
      ind.passedMm1 = true;
      this.statusLine = ind.label + " cleared MM1 -> self-play";
    }
    return ev;
  };

  /* Bots that still need wall training before the post-round tournament. */
  ShapeSearch.prototype.botsNeedingTrain = function () {
    var out = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (!this.botTrainComplete(this.population[i])) {
        out.push(this.population[i]);
      }
    }
    return out;
  };

  ShapeSearch.prototype.roundComplete = function () {
    if (!this.hasBeater()) {
      return false;
    }
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (!this.botTrainComplete(this.population[i])) {
        return false;
      }
    }
    return true;
  };

  /** Start live RR when the round fill is done (safe to call while paused). */
  ShapeSearch.prototype.maybeStartTournament = function () {
    if (this.liveTourn) {
      return false;
    }
    if (this.botsNeedingEval().length) {
      return false;
    }
    if (this.roundComplete() ||
        (!this.botsNeedingTrain().length &&
          this.population.some(function (p) { return p.beaten; }))) {
      this.finishRound();
      return !!this.liveTourn;
    }
    return false;
  };

  /** Next 0.5s train milestone for one bot (e.g. 0.3 -> 0.5, 0.5 -> 1.0). */
  function nextTrainSliceTarget(secondsUsed, maxT) {
    var step = MIN_DWELL_SEC;
    var target = (Math.floor(secondsUsed / step + 1e-9) + 1) * step;
    if (maxT !== null && maxT < target) {
      return maxT;
    }
    return target;
  }

  /**
   * First bot in population order that still needs training (the wave leader).
   * Target = that bot's train_s; everyone else catches up; then leader advances +0.5s.
   */
  ShapeSearch.prototype.firstNeedingBot = function (needing) {
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (this.findBot(needing, this.population[i].id)) {
        return this.population[i];
      }
    }
    return needing.length ? needing[0] : null;
  };

  function botsBehindLeader(needing, leader) {
    var out = [];
    var i;
    for (i = 0; i < needing.length; i += 1) {
      if (needing[i].id === leader.id) {
        continue;
      }
      if (needing[i].secondsUsed + 1e-9 < leader.secondsUsed) {
        out.push(needing[i]);
      }
    }
    return out;
  }

  /**
   * Sweep population top-to-bottom among bots still below the catch-up target.
   */
  ShapeSearch.prototype.pickSweepBot = function (workList) {
    if (!workList.length) {
      return null;
    }
    if (this._dwellId && typeof this._sliceTarget === "number") {
      var cur = this.findBot(workList, this._dwellId);
      if (cur && cur.secondsUsed < this._sliceTarget) {
        return cur;
      }
    }
    var start = typeof this._sweepIndex === "number" ? this._sweepIndex + 1 : 0;
    var pop = this.population;
    var n = pop.length;
    var k;
    for (k = 0; k < n; k += 1) {
      var idx = (start + k) % n;
      if (this.findBot(workList, pop[idx].id)) {
        this._sweepIndex = idx;
        return pop[idx];
      }
    }
    return workList[0];
  };

  ShapeSearch.prototype.findBot = function (bots, id) {
    var i;
    for (i = 0; i < bots.length; i += 1) {
      if (bots[i].id === id) {
        return bots[i];
      }
    }
    return null;
  };

  ShapeSearch.prototype.tick = function (budgetMs) {
    if (this.liveTourn) {
      this.tickLiveTournament(budgetMs);
      return;
    }
    if (this.paused) {
      return;
    }
    if (this.maybeStartTournament()) {
      return;
    }
    var needing = this.botsNeedingTrain();
    var evalDue = this.botsNeedingEval();
    if (!needing.length && !evalDue.length) {
      return;
    }
    var workMs = Math.max(40, budgetMs || 40);
    var deadline = performance.now() + workMs;
    var minT = this.minTrainSec();
    /* Cap slices at min only after a 10/10 beater exists; else keep climbing. */
    var trainCap = this.hasBeater() ? minT : null;
    var leader = this.firstNeedingBot(needing);
    var workList;
    var sliceTarget;
    var evalOnly = [];
    var ei;
    for (ei = 0; ei < evalDue.length; ei += 1) {
      if (!this.findBot(needing, evalDue[ei].id)) {
        evalOnly.push(evalDue[ei]);
      }
    }
    /* Filled bots (skipped for train) still take a pass for eval. */
    if (evalOnly.length) {
      workList = evalOnly;
      sliceTarget = evalOnly[0].secondsUsed;
      this._leaderAdvanceTarget = null;
    } else if (!leader) {
      return;
    } else if (typeof this._leaderAdvanceTarget === "number" &&
        leader.secondsUsed < this._leaderAdvanceTarget) {
      sliceTarget = this._leaderAdvanceTarget;
      if (trainCap !== null && sliceTarget > trainCap) {
        sliceTarget = trainCap;
      }
      workList = [leader];
    } else {
      this._leaderAdvanceTarget = null;
      var behind = botsBehindLeader(needing, leader);
      if (behind.length) {
        sliceTarget = leader.secondsUsed;
        workList = behind;
      } else {
        sliceTarget = nextTrainSliceTarget(leader.secondsUsed, trainCap);
        if (sliceTarget <= leader.secondsUsed) {
          if (this.botTrainComplete(leader)) {
            this._dwellId = null;
            this._sliceTarget = null;
            this._leaderAdvanceTarget = null;
            this.maybeStartTournament();
            return;
          }
          /* Past min with no beater yet, or still short of min: keep going. */
          sliceTarget = trainCap !== null ? trainCap : (leader.secondsUsed + MIN_DWELL_SEC);
          if (sliceTarget <= leader.secondsUsed) {
            sliceTarget = leader.secondsUsed + MIN_DWELL_SEC;
          }
        }
        this._leaderAdvanceTarget = sliceTarget;
        workList = [leader];
      }
    }
    this._sliceTarget = sliceTarget;
    var ind = this.pickSweepBot(workList);
    if (!ind) {
      return;
    }
    this._dwellId = ind.id;
    this._currentId = ind.id;
    var sliceEnd = performance.now() + Math.min(SLICE_MS, deadline - performance.now());

    while (performance.now() < sliceEnd && performance.now() < deadline) {
      var atCap = this.botTrainComplete(ind);
      var pastSlice = ind.secondsUsed >= sliceTarget;
      if (atCap || pastSlice) {
        this.evalIfDue(ind);
        this._dwellId = null;
        this._sliceTarget = null;
        if (this._leaderAdvanceTarget !== null &&
            leader &&
            ind.id === leader.id &&
            ind.secondsUsed >= this._leaderAdvanceTarget) {
          this._leaderAdvanceTarget = null;
        }
        if (atCap) {
          this._leaderAdvanceTarget = null;
        }
        break;
      }

      var useSelfPlay = this.round > 1 && !!ind.passedMm1;
      ensureEvalMeta(ind);
      var warmDt = stepWarmup(ind, false);
      if (warmDt !== null) {
        ind.secondsUsed += warmDt;
      } else {
        var ctx = this.evalContext(ind);
        ensureTrainSchedule(ind);
        if (ind.trainSinceEvalMs < ind.trainBetweenTargetMs) {
          this.trainIndividual(ind, useSelfPlay);
        } else {
          this.runIndividualEval(ind, ctx.evalLevel, ctx.onRoundMm);
          trainCap = this.hasBeater() ? minT : null;
          if (!this.botTrainComplete(ind)) {
            this.trainIndividual(ind, useSelfPlay);
          }
        }
      }

      needing = this.botsNeedingTrain();
      if (!needing.length) {
        this.evalIfDue(ind);
        this._dwellId = null;
        this._sliceTarget = null;
        this._leaderAdvanceTarget = null;
        break;
      }
      if (!this.findBot(needing, ind.id) || ind.secondsUsed >= sliceTarget) {
        this.evalIfDue(ind);
        this._dwellId = null;
        this._sliceTarget = null;
        if (ind.secondsUsed >= sliceTarget &&
            this._leaderAdvanceTarget !== null &&
            leader &&
            ind.id === leader.id) {
          this._leaderAdvanceTarget = null;
        }
        break;
      }
    }
    this.statusLine = "Round " + this.round + " " + formatMmTag(this.mmLevel) +
      " | " + ind.label + " train=" + ind.secondsUsed.toFixed(1) + "s" +
      (this.bestBeatSec !== null ? (" | best=" + this.bestBeatSec.toFixed(2) + "s") : "") +
      " | min=" + minT.toFixed(0) + "s" +
      (this.hasBeater() ? " | beater ready" : " | need 10/10 streak");
    this.maybeStartTournament();
  };

  ShapeSearch.prototype.buildTournamentEntries = function (bots, mmMs) {
    var entries = [];
    var i;
    for (i = 0; i < bots.length; i += 1) {
      entries.push({
        id: bots[i].id,
        family: bots[i].family,
        label: bots[i].label,
        agent: bots[i].agent
      });
    }
    var have = {};
    for (i = 0; i < entries.length; i += 1) {
      have[entries[i].id] = true;
    }
    for (i = 0; i < this.tournamentSnapshots.length; i += 1) {
      var snap = this.tournamentSnapshots[i];
      if (have[snap.id]) {
        continue;
      }
      var frozen = individualFromSnapshot(snap);
      entries.push({
        id: frozen.id,
        family: frozen.family,
        label: frozen.label + " (snap)",
        agent: frozen.agent
      });
    }
    return entries.concat(makeBaselineEntries(this.mmLevel, mmMs));
  };

  ShapeSearch.prototype.storeTournamentSnapshots = function (bots, result) {
    var top = topLearnedRanks(result.ranked, survivorLimitForRound(this.round));
    var snaps = [];
    var i;
    for (i = 0; i < top.length; i += 1) {
      var bot = null;
      var b;
      for (b = 0; b < bots.length; b += 1) {
        if (bots[b].id === top[i].id) {
          bot = bots[b];
          break;
        }
      }
      if (bot) {
        snaps.push(snapshotIndividual(bot));
      }
    }
    this.tournamentSnapshots = snaps;
  };

  /**
   * Start a live RR that advances one game at a time (UI updates between games,
   * at most every TOURN_UI_MIN_MS).
   */
  ShapeSearch.prototype.startLiveTournament = function (bots, storeSnapshots, onComplete) {
    if (this.liveTourn) {
      return false;
    }
    var mmMs = APP.calibrateTournamentMoveMs(this.mmLevel);
    var entries = this.buildTournamentEntries(bots, mmMs);
    var session = createRoundRobinSession(entries, mmMs, this.botMaxDepth(this.mmLevel));
    this.liveTourn = {
      session: session,
      bots: bots,
      storeSnapshots: !!storeSnapshots,
      onComplete: onComplete || null
    };
    this._tournLastUiAt = 0;
    this.lastTournament = roundRobinStandings(session);
    this._tournUiDirty = true;
    this.statusLine = "Tournament running (" + session.gamesDone + "/" +
      session.gamesTotal + " games, " + mmMs.toFixed(1) + "ms/move)";
    return true;
  };

  ShapeSearch.prototype.tickLiveTournament = function (budgetMs) {
    if (!this.liveTourn) {
      return;
    }
    var session = this.liveTourn.session;
    var deadline = performance.now() + Math.max(20, budgetMs || 120);
    var played = false;
    while (!session.done && performance.now() < deadline) {
      stepRoundRobinSession(session);
      played = true;
      this.lastTournament = roundRobinStandings(session);
      this.statusLine = "Tournament " + session.gamesDone + "/" + session.gamesTotal +
        " | " + session.moveBudgetMs.toFixed(1) + "ms/move";
      var now = performance.now();
      if (now - this._tournLastUiAt >= TOURN_UI_MIN_MS) {
        this._tournLastUiAt = now;
        this._tournUiDirty = true;
        break;
      }
    }
    if (session.done) {
      var result = roundRobinStandings(session);
      result.inProgress = false;
      this.lastTournament = result;
      if (this.liveTourn.storeSnapshots) {
        this.storeTournamentSnapshots(this.liveTourn.bots, result);
      }
      var onComplete = this.liveTourn.onComplete;
      this.liveTourn = null;
      this._tournUiDirty = true;
      if (onComplete) {
        onComplete.call(this, result);
      } else {
        this.statusLine = "Tournament done (" +
          (result.durationMs / 1000).toFixed(1) + "s)";
      }
    } else if (!played) {
      /* keep status visible */
    }
  };

  ShapeSearch.prototype.drainLiveTournament = function () {
    while (this.liveTourn) {
      this.tickLiveTournament(60000);
    }
  };

  ShapeSearch.prototype.applyRoundAdvance = function (tourn) {
    var keepN = survivorLimitForRound(this.round);
    var topRows = historyTopRanks(tourn.ranked, HISTORY_LEARNED_LIMIT).map(function (r) {
      return {
        id: r.id,
        label: r.label,
        family: r.family,
        points: r.points,
        wins: r.wins,
        draws: r.draws,
        losses: r.losses
      };
    });
    this.history.push({
      round: this.round,
      mmLevel: this.mmLevel,
      top: topRows
    });

    var top = topLearnedRanks(tourn.ranked, keepN);

    var byId = {};
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      byId[this.population[i].id] = this.population[i];
    }

    var nextPop = [];
    var topIdSet = {};
    for (i = 0; i < top.length; i += 1) {
      var src = byId[top[i].id];
      if (!src) {
        continue;
      }
      topIdSet[src.id] = true;
      resetIndividualProgress(src);
      src.live = true;
      assignNeatIdentity(src);
      tryAddUniqueShape(nextPop, src);
    }
    /* Survivors that remain after near-dup collapse (order follows nextPop). */
    var survivors = [];
    for (i = 0; i < nextPop.length; i += 1) {
      if (topIdSet[nextPop[i].id]) {
        survivors.push(nextPop[i]);
      }
    }
    /* Offspring only from top MAX_MUTANT_PARENTS finishers still present. */
    var mutantParents = pickMutantParents(top, survivors, MAX_MUTANT_PARENTS);

    /* NEAT parents get weight clones; morph parents get width/depth/LR mutants. */
    cloneNeatSurvivors(nextPop, mutantParents);

    for (i = 0; i < mutantParents.length; i += 1) {
      var parent = mutantParents[i];
      if (isNeatFamily(parent.family)) {
        continue;
      }
      var kids = mutateFrom(parent);
      var k;
      for (k = 0; k < kids.length; k += 1) {
        var kidSpec = kids[k];
        var childAgent = cloneAgentFromParent(
          parent, kidSpec.family, kidSpec.layerSizes, kidSpec.learningRate
        );
        var child = createIndividual({
          family: kidSpec.family,
          layerSizes: kidSpec.layerSizes,
          learningRate: kidSpec.learningRate,
          agent: childAgent
        });
        tryAddUniqueShape(nextPop, child);
      }
    }

    this.population = nextPop;
    this.round += 1;
    this.mmLevel += 1;
    this.bestBeatSec = null;
    this._dwellId = null;
    this._sliceTarget = null;
    this._leaderAdvanceTarget = null;
    this._sweepIndex = -1;
    this.statusLine = "Advanced to round " + this.round + " (" + formatMmTag(this.mmLevel) +
      ", kept " + survivors.length + "/" + keepN +
      ", mutants from top " + mutantParents.length + ")";
  };

  ShapeSearch.prototype.finishRound = function () {
    if (this.liveTourn) {
      return;
    }
    if (!this.population.some(function (p) { return p.beaten; })) {
      return;
    }
    var self = this;
    this.startLiveTournament(this.population, true, function (result) {
      self.applyRoundAdvance(result);
    });
  };

  ShapeSearch.prototype.manualTournament = function () {
    if (this.liveTourn) {
      return null;
    }
    this.startLiveTournament(this.population, false, null);
    return null;
  };

  /* --- Train One controller --- */
  function TrainOne() {
    this.trainee = null;
    this.mmLevel = 0;
    /** Shared with Shape Search UI (0 = Match Minimax; -1..-10). */
    this.botSearchOffset = 0;
    this.ladder = [];
    this.paused = false;
    this.userPaused = false;
    this.T_tourn = null;
    this.trainSinceTourn = 0;
    this.lastTournament = null;
    /** Contender ids with more tourn points than trainee last tournament. */
    this.peerBeatIds = [];
    this.statusLine = "Pick a bot to train";
    this.startedAt = null;
    this.totalTrainSec = 0;
  }

  TrainOne.prototype.botMaxDepth = function (mmLevel) {
    var level = typeof mmLevel === "number" ? mmLevel : this.mmLevel;
    return APP.botSearchDepthForLevel(level, this.botSearchOffset);
  };

  TrainOne.prototype.setBotSearchOffset = function (offset) {
    this.botSearchOffset = clampBotSearchOffset(offset);
  };

  TrainOne.prototype.setPaused = function (paused, fromUser) {
    this.paused = !!paused;
    if (fromUser) {
      this.userPaused = !!paused;
    }
  };

  TrainOne.prototype.resumeIfAllowed = function () {
    if (!this.userPaused) {
      this.paused = false;
    }
  };

  TrainOne.prototype.startFromSnapshotOrLive = function (source) {
    var lr = typeof source.learningRate === "number" ?
      source.learningRate : APP.defaultLearningRate(source.family);
    var agent;
    if (source.blob) {
      agent = APP.shapeAgentFromDict(source.blob);
    } else {
      agent = APP.createShapeAgent(source.family, source.layerSizes, lr);
      /* Clone live agent weights when possible. */
      if (source.agent && typeof source.agent.toDict === "function") {
        var blob = APP.shapeAgentToDict(source.agent, source.family);
        agent = APP.shapeAgentFromDict(blob);
      }
    }
    if (typeof lr === "number") {
      APP.applyShapeLearningRate(agent, source.family, lr);
    }
    this.trainee = {
      id: "train_" + source.id,
      sourceId: source.id,
      imported: false,
      family: source.family,
      layerSizes: cloneSizes(source.layerSizes),
      learningRate: lr,
      label: source.label,
      agent: agent,
      warmupLeftMs: WARMUP_MS,
      evalSeconds: 0,
      beatStreak: 0,
      beaten: false,
      nemesisSeeds: [],
      peerPlayCounts: {},
      lastEvalWinRate: null,
      bestWinRate: null,
      bestBlob: null,
      evalGamesTarget: EVAL_GAMES
    };
    initTrainSchedule(this.trainee);
    this.mmLevel = 0;
    this.ladder = [];
    this.T_tourn = null;
    this.trainSinceTourn = 0;
    this.lastTournament = null;
    this.peerBeatIds = [];
    this.startedAt = performance.now();
    this.totalTrainSec = 0;
    this.statusLine = "Training " + this.trainee.label;
  };

  /**
   * Keep learned weights / keep-best; restart ladder from Random with fresh
   * train clock and tournament schedule.
   */
  TrainOne.prototype.restartLadder = function () {
    if (!this.trainee) {
      return false;
    }
    var ind = this.trainee;
    ind.warmupLeftMs = WARMUP_MS;
    ind.evalSeconds = 0;
    ind.beatStreak = 0;
    ind.beaten = false;
    ind.nemesisSeeds = [];
    ind.peerPlayCounts = {};
    /* Keep agent, bestBlob, bestWinRate, lastEval* (brain + recent grade). */
    initTrainSchedule(ind);
    this.mmLevel = 0;
    this.ladder = [];
    this.T_tourn = null;
    this.trainSinceTourn = 0;
    this.peerBeatIds = [];
    this.lastTournament = null;
    this.totalTrainSec = 0;
    this.startedAt = performance.now();
    this.statusLine = "Restarted ladder for " + ind.label + " (weights kept)";
    return true;
  };

  /** Wipe weights and progress; same family/shape, fresh untrained agent. */
  TrainOne.prototype.resetTrainee = function () {
    if (!this.trainee) {
      return;
    }
    var lr = typeof this.trainee.learningRate === "number" ?
      this.trainee.learningRate : APP.defaultLearningRate(this.trainee.family);
    this.trainee.agent = APP.createShapeAgent(this.trainee.family, this.trainee.layerSizes, lr);
    this.trainee.learningRate = lr;
    this.trainee.warmupLeftMs = WARMUP_MS;
    this.trainee.evalSeconds = 0;
    this.trainee.beatStreak = 0;
    this.trainee.beaten = false;
    this.trainee.nemesisSeeds = [];
    this.trainee.peerPlayCounts = {};
    this.trainee.lastEvalWinRate = null;
    this.trainee.lastEvalWins = null;
    this.trainee.lastEvalDraws = null;
    this.trainee.lastEvalGames = null;
    this.trainee.bestWinRate = null;
    this.trainee.bestBlob = null;
    initTrainSchedule(this.trainee);
    this.mmLevel = 0;
    this.ladder = [];
    this.T_tourn = null;
    this.trainSinceTourn = 0;
    this.peerBeatIds = [];
    this.lastTournament = null;
    this.totalTrainSec = 0;
    this.startedAt = performance.now();
    this.statusLine = "Reset " + this.trainee.label + " (new weights)";
  };

  /**
   * Peer from last tournament: only contenders with more points than trainee.
   * Soft-RR among those still present in contenderEntries.
   */
  TrainOne.prototype.pickPeerWhoBeatTrainee = function (contenderEntries) {
    var ind = this.trainee;
    if (!ind || !this.peerBeatIds || !this.peerBeatIds.length) {
      return null;
    }
    ensureEvalMeta(ind);
    var allowed = {};
    var i;
    for (i = 0; i < this.peerBeatIds.length; i += 1) {
      allowed[this.peerBeatIds[i]] = true;
    }
    var candidates = [];
    var peers = contenderEntries || [];
    for (i = 0; i < peers.length; i += 1) {
      if (peers[i].id !== ind.id && peers[i].agent && allowed[peers[i].id]) {
        candidates.push(peers[i]);
      }
    }
    if (!candidates.length) {
      return null;
    }
    candidates.sort(function (a, b) {
      var ca = ind.peerPlayCounts[a.id] || 0;
      var cb = ind.peerPlayCounts[b.id] || 0;
      if (ca !== cb) {
        return ca - cb;
      }
      return idAge(a.id) - idAge(b.id);
    });
    return candidates[0];
  };

  /** Same curriculum mix as Shape Search (peer = higher-point prior-tourn bots). */
  TrainOne.prototype.curriculumTrainStep = function (contenderEntries) {
    var ind = this.trainee;
    ensureEvalMeta(ind);
    ensureNemesis(ind);
    var depth = this.botMaxDepth(this.mmLevel);
    var mmLevel = this.mmLevel;
    var t0 = performance.now();
    if (isEvolveFamily(ind.family)) {
      APP.shapeEvolveOne(ind.agent, ind.family);
      return (performance.now() - t0) / 1000;
    }
    var peer = wrBeats(ind.lastEvalWinRate) ?
      this.pickPeerWhoBeatTrainee(contenderEntries) : null;
    var frozen = loadFrozenAgent(ind);
    var kind = pickShapeTrainKind({
      beatsMm: wrBeats(ind.lastEvalWinRate),
      hasPeer: !!peer,
      hasFrozen: !!frozen,
      hasNemesis: !isRandomMmLevel(mmLevel) && !!(ind.nemesisSeeds && ind.nemesisSeeds.length),
      canSelf: true
    });
    if (isRandomMmLevel(mmLevel) && (kind === "mm" || kind === "nemesis")) {
      kind = "random";
    }
    if (kind === "peer" && peer) {
      APP.shapeTrainVsPeer(
        ind.agent, ind.family, peer.agent, peer.family, true, depth, false
      );
      ind.peerPlayCounts[peer.id] = (ind.peerPlayCounts[peer.id] || 0) + 1;
    } else if (kind === "self" || (kind === "peer" && !peer)) {
      APP.shapeTrainVsSelf(ind.agent, ind.family, true, depth);
    } else if (kind === "frozen" && frozen) {
      APP.shapeTrainVsPeer(
        ind.agent, ind.family, frozen, ind.family, true, depth, false
      );
    } else if (kind === "nemesis") {
      var entry = pickNemesisSeed(ind);
      if (entry) {
        APP.shapeTrainVsMinimax(
          ind.agent, ind.family, entry.level, false, true, depth, entry.seed
        );
      } else {
        APP.shapeTrainVsMinimax(
          ind.agent, ind.family, mmLevel, true, true, depth, null
        );
      }
    } else if (kind === "mm") {
      var packed = APP.shapeTrainVsMinimax(
        ind.agent, ind.family, mmLevel, true, true, depth, null
      );
      if (packed && packed.result === "loss") {
        addNemesisSeed(ind, packed.seed, mmLevel);
      }
    } else {
      APP.shapeTrainVsRandom(ind.agent, ind.family, true, true, depth);
    }
    return (performance.now() - t0) / 1000;
  };

  TrainOne.prototype.baselineIdForLevel = function () {
    return isRandomMmLevel(this.mmLevel) ? "random" : "minimax";
  };

  /** True when trainee scored strictly more tournament points than Random/MM. */
  TrainOne.prototype.traineeBeatsBaseline = function (tourn) {
    if (!tourn || !tourn.ranked || !this.trainee) {
      return false;
    }
    var tid = this.trainee.id;
    var bid = this.baselineIdForLevel();
    var traineePts = null;
    var basePts = null;
    var i;
    for (i = 0; i < tourn.ranked.length; i += 1) {
      var row = tourn.ranked[i];
      if (row.id === tid) {
        traineePts = row.points;
      }
      if (row.id === bid || row.family === bid) {
        basePts = row.points;
      }
    }
    return typeof traineePts === "number" && typeof basePts === "number" &&
      traineePts > basePts;
  };

  TrainOne.prototype.advanceLevel = function (tourn) {
    var ind = this.trainee;
    var wr = typeof ind.lastEvalWinRate === "number" ? ind.lastEvalWinRate : 0;
    this.ladder.push({
      mmLevel: this.mmLevel,
      seconds: this.totalTrainSec,
      winRate: wr,
      tournPoints: (function () {
        var i;
        for (i = 0; i < tourn.ranked.length; i += 1) {
          if (tourn.ranked[i].id === ind.id) {
            return tourn.ranked[i].points;
          }
        }
        return null;
      })()
    });
    var beatenTag = formatMmTag(this.mmLevel);
    this.mmLevel += 1;
    this.trainSinceTourn = 0;
    this.peerBeatIds = [];
    ind.warmupLeftMs = WARMUP_MS;
    ind.beatStreak = 0;
    ind.beaten = false;
    ind.nemesisSeeds = [];
    initTrainSchedule(ind);
    this.statusLine = "Beat " + beatenTag + " in tournament (more points than " +
      beatenTag + ") at train t=" + this.totalTrainSec.toFixed(1) + "s | now " +
      formatMmTag(this.mmLevel);
  };

  TrainOne.prototype.maybeRunTournament = function (contenderEntries) {
    var due = false;
    if (this.T_tourn === null) {
      /* First contender tournament after a short warm train. */
      due = this.totalTrainSec >= 1;
    } else {
      due = this.trainSinceTourn >= 5 * this.T_tourn;
    }
    if (!due) {
      return false;
    }
    var tourn = this.runContenderTournament(contenderEntries);
    this.trainSinceTourn = 0;
    this.peerBeatIds = peersWhoBeatTrainee(tourn, this.trainee.id);
    if (this.traineeBeatsBaseline(tourn)) {
      this.advanceLevel(tourn);
    } else {
      var bid = this.baselineIdForLevel();
      var tPts = "-";
      var bPts = "-";
      var i;
      for (i = 0; i < tourn.ranked.length; i += 1) {
        if (tourn.ranked[i].id === this.trainee.id) {
          tPts = String(tourn.ranked[i].points);
        }
        if (tourn.ranked[i].id === bid || tourn.ranked[i].family === bid) {
          bPts = String(tourn.ranked[i].points);
        }
      }
      this.statusLine = this.trainee.label + " tourn " + tPts + " pts vs " +
        formatMmTag(this.mmLevel) + " " + bPts + " pts (need more to advance)";
    }
    return true;
  };

  TrainOne.prototype.tick = function (budgetMs, contenderEntries) {
    if (this.paused || !this.trainee) {
      return;
    }
    var deadline = performance.now() + Math.max(5, budgetMs || 40);
    var ind = this.trainee;
    if (typeof ind.warmupLeftMs !== "number") {
      ind.warmupLeftMs = WARMUP_MS;
    }
    if (typeof ind.evalSeconds !== "number") {
      ind.evalSeconds = 0;
    }
    ensureTrainSchedule(ind);
    ensureNemesis(ind);
    var tournNote = null;

    while (performance.now() < deadline) {
      var warmDt = stepWarmup(ind, false);
      if (warmDt !== null) {
        this.totalTrainSec += warmDt;
        this.trainSinceTourn += warmDt;
        continue;
      }

      if (ind.trainSinceEvalMs < ind.trainBetweenTargetMs) {
        var trainDt = this.curriculumTrainStep(contenderEntries);
        this.totalTrainSec += trainDt;
        this.trainSinceTourn += trainDt;
        ind.trainSinceEvalMs += trainDt * 1000;
        if (this.maybeRunTournament(contenderEntries)) {
          tournNote = this.statusLine;
        }
        continue;
      }

      ensureEvalMeta(ind);
      var packed = runBotEval(ind, this.mmLevel, this.botMaxDepth(this.mmLevel));
      noteBeatStreak(ind, packed.ev.winRate);
      if (packed.weightNote === "restored best") {
        this.statusLine = ind.label + " restored best (playoff won)";
      } else if (packed.weightNote === "kept current") {
        this.statusLine = ind.label + " kept current (playoff won)";
      }

      var postDt = this.curriculumTrainStep(contenderEntries);
      this.totalTrainSec += postDt;
      this.trainSinceTourn += postDt;
      ind.trainSinceEvalMs += postDt * 1000;
      if (this.maybeRunTournament(contenderEntries)) {
        tournNote = this.statusLine;
      }
    }
    if (tournNote) {
      this.statusLine = tournNote;
    } else {
      this.statusLine = ind.label + " | " + formatMmTag(this.mmLevel) +
        " | train " + this.totalTrainSec.toFixed(1) + "s" +
        (this.T_tourn ? (" | next tourn in " +
          Math.max(0, 5 * this.T_tourn - this.trainSinceTourn).toFixed(1) + "s") :
          " | first tourn after 1s train");
    }
  };

  TrainOne.prototype.runContenderTournament = function (contenderEntries) {
    var entries = (contenderEntries || []).slice();
    var mmMs = APP.calibrateTournamentMoveMs(this.mmLevel);
    entries.push({
      id: this.trainee.id,
      family: this.trainee.family,
      label: this.trainee.label + " (trainee)",
      agent: this.trainee.agent
    });
    entries = entries.concat(makeBaselineEntries(this.mmLevel, mmMs));
    /* Dedupe by id */
    var seen = {};
    var uniq = [];
    var i;
    for (i = 0; i < entries.length; i += 1) {
      if (seen[entries[i].id]) {
        continue;
      }
      seen[entries[i].id] = true;
      uniq.push(entries[i]);
    }
    this.lastTournament = runRoundRobin(uniq, mmMs, this.botMaxDepth(this.mmLevel));
    this.T_tourn = this.lastTournament.durationMs / 1000;
    return this.lastTournament;
  };

  /* --- Persistence (IndexedDB + localStorage fallback) --- */
  function idbSupported() {
    return typeof indexedDB !== "undefined";
  }

  function openShapeDb(callback) {
    var req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = function (ev) {
      var db = ev.target.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = function () {
      callback(null, req.result);
    };
    req.onerror = function () {
      callback(req.error || new Error("indexedDB open failed"));
    };
  }

  function idbGetShape(callback) {
    if (!idbSupported()) {
      callback(null, null);
      return;
    }
    openShapeDb(function (err, db) {
      if (err) {
        callback(err, null);
        return;
      }
      var tx = db.transaction(IDB_STORE, "readonly");
      var req = tx.objectStore(IDB_STORE).get(STORAGE_KEY);
      req.onsuccess = function () {
        db.close();
        callback(null, req.result || null);
      };
      req.onerror = function () {
        db.close();
        callback(req.error, null);
      };
    });
  }

  function idbSetShape(json, callback) {
    if (!idbSupported()) {
      callback(new Error("indexedDB unavailable"));
      return;
    }
    openShapeDb(function (err, db) {
      if (err) {
        callback(err);
        return;
      }
      var tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(json, STORAGE_KEY);
      tx.oncomplete = function () {
        db.close();
        callback(null);
      };
      tx.onerror = function () {
        db.close();
        callback(tx.error);
      };
    });
  }

  function idbDeleteShape(callback) {
    if (!idbSupported()) {
      if (callback) {
        callback(null);
      }
      return;
    }
    openShapeDb(function (err, db) {
      if (err) {
        if (callback) {
          callback(err);
        }
        return;
      }
      var tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).delete(STORAGE_KEY);
      tx.oncomplete = function () {
        db.close();
        if (callback) {
          callback(null);
        }
      };
      tx.onerror = function () {
        db.close();
        if (callback) {
          callback(tx.error);
        }
      };
    });
  }

  function bumpNextIdFromString(id) {
    if (!id || typeof id !== "string") {
      return;
    }
    var m = id.match(/_(\d+)$/);
    if (m) {
      var n = parseInt(m[1], 10);
      if (n >= nextId) {
        nextId = n + 1;
      }
    }
  }

  function serializeIndividual(ind) {
    return {
      id: ind.id,
      family: ind.family,
      layerSizes: cloneSizes(ind.layerSizes),
      learningRate: ind.learningRate,
      label: ind.label,
      neatSerial: typeof ind.neatSerial === "number" ? ind.neatSerial : null,
      secondsUsed: ind.secondsUsed || 0,
      evalSeconds: ind.evalSeconds || 0,
      beaten: !!ind.beaten,
      beatStreak: typeof ind.beatStreak === "number" ? ind.beatStreak : 0,
      secondsToBeat: ind.secondsToBeat === undefined ? null : ind.secondsToBeat,
      passedMm1: !!ind.passedMm1,
      lastEvalWins: typeof ind.lastEvalWins === "number" ? ind.lastEvalWins : null,
      lastEvalDraws: typeof ind.lastEvalDraws === "number" ? ind.lastEvalDraws : null,
      lastEvalGames: typeof ind.lastEvalGames === "number" ? ind.lastEvalGames : null,
      lastEvalWinRate: typeof ind.lastEvalWinRate === "number" ? ind.lastEvalWinRate : null,
      evalGamesTarget: typeof ind.evalGamesTarget === "number" ? ind.evalGamesTarget : EVAL_GAMES,
      bestWinRate: typeof ind.bestWinRate === "number" ? ind.bestWinRate : null,
      bestBlob: ind.bestBlob || null,
      peerPlayCounts: ind.peerPlayCounts || {},
      nemesisSeeds: Array.isArray(ind.nemesisSeeds) ? ind.nemesisSeeds.slice() : [],
      warmupLeftMs: typeof ind.warmupLeftMs === "number" ? ind.warmupLeftMs : WARMUP_MS,
      trainSinceEvalMs: typeof ind.trainSinceEvalMs === "number" ? ind.trainSinceEvalMs : 0,
      trainBetweenTargetMs: typeof ind.trainBetweenTargetMs === "number" ?
        ind.trainBetweenTargetMs : TRAIN_BETWEEN_EVAL_MS,
      trainBetweenLocked: !!ind.trainBetweenLocked,
      blob: APP.shapeAgentToDict(ind.agent, ind.family)
    };
  }

  function deserializeIndividual(row) {
    bumpNextIdFromString(row.id);
    var lr = typeof row.learningRate === "number" ?
      row.learningRate : APP.defaultLearningRate(row.family);
    var agent = row.blob ? APP.shapeAgentFromDict(row.blob) :
      APP.createShapeAgent(row.family, row.layerSizes, lr);
    if (typeof lr === "number") {
      APP.applyShapeLearningRate(agent, row.family, lr);
    }
    var ind = {
      id: row.id,
      family: row.family,
      layerSizes: cloneSizes(row.layerSizes),
      learningRate: lr,
      label: row.label || familyLabel(row.family, row.layerSizes),
      neatSerial: typeof row.neatSerial === "number" ? row.neatSerial : undefined,
      agent: agent,
      secondsUsed: row.secondsUsed || 0,
      evalSeconds: row.evalSeconds || 0,
      beaten: !!row.beaten,
      beatStreak: typeof row.beatStreak === "number" ? row.beatStreak : 0,
      secondsToBeat: row.secondsToBeat === undefined ? null : row.secondsToBeat,
      passedMm1: !!row.passedMm1,
      lastEvalWins: typeof row.lastEvalWins === "number" ? row.lastEvalWins : null,
      lastEvalDraws: typeof row.lastEvalDraws === "number" ? row.lastEvalDraws : null,
      lastEvalGames: typeof row.lastEvalGames === "number" ? row.lastEvalGames : null,
      lastEvalWinRate: typeof row.lastEvalWinRate === "number" ? row.lastEvalWinRate : null,
      evalGamesTarget: typeof row.evalGamesTarget === "number" ? row.evalGamesTarget : EVAL_GAMES,
      bestWinRate: typeof row.bestWinRate === "number" ? row.bestWinRate : null,
      bestBlob: row.bestBlob || null,
      peerPlayCounts: row.peerPlayCounts || {},
      nemesisSeeds: Array.isArray(row.nemesisSeeds) ? row.nemesisSeeds.slice() : [],
      warmupLeftMs: typeof row.warmupLeftMs === "number" ? row.warmupLeftMs : WARMUP_MS,
      trainSinceEvalMs: typeof row.trainSinceEvalMs === "number" ? row.trainSinceEvalMs : 0,
      trainBetweenTargetMs: typeof row.trainBetweenTargetMs === "number" ?
        row.trainBetweenTargetMs : TRAIN_BETWEEN_EVAL_MS,
      trainBetweenLocked: !!row.trainBetweenLocked,
      live: true
    };
    ensureEvalMeta(ind);
    ensureBeatStreak(ind);
    if (ind.beaten && ind.beatStreak < BEAT_STREAK_NEED) {
      ind.beatStreak = BEAT_STREAK_NEED;
    }
    assignNeatIdentity(ind);
    return ind;
  }

  /* --- Lab facade --- */
  function ShapeLab(options) {
    options = options || {};
    this.onSnapshot = options.onSnapshot || null;
    this.search = new ShapeSearch();
    this.trainOne = new TrainOne();
    this.activeTab = "search";
    this.running = false;
    this.rafId = null;
    this.lastUi = 0;
    this.lastSaveAt = 0;
    this.lastSaveStatus = null;
    this._autoSaveTimer = null;
    this._unloadBound = false;
  }

  ShapeLab.prototype.unionCatalog = function () {
    var out = [];
    var seen = {};
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      var p = this.search.population[i];
      seen[p.id] = true;
      out.push({
        id: p.id,
        family: p.family,
        layerSizes: cloneSizes(p.layerSizes),
        learningRate: p.learningRate,
        label: p.label + " (live)",
        agent: p.agent,
        live: true
      });
    }
    for (i = 0; i < this.search.tournamentSnapshots.length; i += 1) {
      var s = this.search.tournamentSnapshots[i];
      if (seen[s.id]) {
        continue;
      }
      seen[s.id] = true;
      out.push({
        id: s.id,
        family: s.family,
        layerSizes: cloneSizes(s.layerSizes),
        learningRate: s.learningRate,
        label: (s.label || familyLabel(s.family, s.layerSizes)) + " (tournament)",
        blob: s.blob,
        live: false
      });
    }
    return out;
  };

  /** Observe dropdown: baselines, Train One trainee (if any), then Search bots. */
  ShapeLab.prototype.getPlayCatalog = function () {
    var out = [
      { id: "random", name: "Random", family: "random" },
      { id: "minimax", name: "Minimax", family: "minimax" },
      { id: "mcts", name: "MCTS", family: "mcts" }
    ];
    var t = this.trainOne && this.trainOne.trainee;
    if (t) {
      out.push({
        id: t.id,
        name: (t.label || familyLabel(t.family, t.layerSizes)) + " (trainee)",
        family: t.family
      });
    }
    var cat = this.unionCatalog();
    var i;
    for (i = 0; i < cat.length; i += 1) {
      out.push({
        id: cat[i].id,
        name: cat[i].label,
        family: cat[i].family
      });
    }
    return out;
  };

  ShapeLab.prototype.ensurePlayBaselines = function () {
    if (this._playBaselines) {
      return this._playBaselines;
    }
    /* Tournament baseline agents: Random (MM0), timed Minimax, budget MCTS. */
    var map = {};
    var packs = [makeBaselineEntries(0, 50), makeBaselineEntries(1, 50)];
    var p;
    var i;
    for (p = 0; p < packs.length; p += 1) {
      for (i = 0; i < packs[p].length; i += 1) {
        map[packs[p][i].id] = packs[p][i];
      }
    }
    this._playBaselines = map;
    return map;
  };

  ShapeLab.prototype.contenderEntriesForTrain = function () {
    var cat = this.unionCatalog();
    var entries = [];
    var i;
    for (i = 0; i < cat.length; i += 1) {
      var c = cat[i];
      var agent = c.agent;
      if (!agent && c.blob) {
        agent = APP.shapeAgentFromDict(c.blob);
      }
      if (!agent) {
        continue;
      }
      entries.push({
        id: c.id,
        family: c.family,
        label: c.label,
        agent: agent
      });
    }
    return entries;
  };

  ShapeLab.prototype.setActiveTab = function (name) {
    var prev = this.activeTab;
    this.activeTab = name;
    if (name !== "search") {
      this.search.setPaused(true, false);
    } else {
      this.search.resumeIfAllowed();
    }
    if (name !== "train") {
      this.trainOne.setPaused(true, false);
    } else {
      this.trainOne.resumeIfAllowed();
    }
    if (prev !== name) {
      this.saveLocal();
      this.publish();
    }
  };

  ShapeLab.prototype.saveStatusLine = function () {
    if (!this.lastSaveStatus) {
      return "not saved yet";
    }
    if (this.lastSaveStatus.pending) {
      return "saving...";
    }
    if (!this.lastSaveStatus.ok) {
      return "save FAILED";
    }
    var ago = this.lastSaveAt ?
      Math.round((performance.now() - this.lastSaveAt) / 1000) : 0;
    return "saved " + ago + "s ago";
  };

  ShapeLab.prototype.setBotSearchOffset = function (offset) {
    var off = clampBotSearchOffset(offset);
    this.search.setBotSearchOffset(off);
    this.trainOne.setBotSearchOffset(off);
    return off;
  };

  ShapeLab.prototype.getSnapshot = function () {
    var searchRows = this.search.population.map(function (p) {
      return {
        id: p.id,
        label: p.label,
        secondsUsed: p.secondsUsed,
        beaten: p.beaten,
        beatStreak: typeof p.beatStreak === "number" ? p.beatStreak : 0,
        beatStreakNeed: BEAT_STREAK_NEED,
        secondsToBeat: p.secondsToBeat,
        lastEvalWins: typeof p.lastEvalWins === "number" ? p.lastEvalWins : null,
        lastEvalDraws: typeof p.lastEvalDraws === "number" ? p.lastEvalDraws : null,
        lastEvalGames: typeof p.lastEvalGames === "number" ? p.lastEvalGames : null,
        lastEvalWinRate: typeof p.lastEvalWinRate === "number" ? p.lastEvalWinRate : null,
        evalGamesTarget: typeof p.evalGamesTarget === "number" ? p.evalGamesTarget : EVAL_GAMES,
        bestWinRate: typeof p.bestWinRate === "number" ? p.bestWinRate : null
      };
    });
    var currentBot = null;
    if (this.search._currentId) {
      var i;
      for (i = 0; i < this.search.population.length; i += 1) {
        if (this.search.population[i].id === this.search._currentId) {
          currentBot = this.search.population[i];
          break;
        }
      }
    }
    return {
      activeTab: this.activeTab,
      saveStatusLine: this.saveStatusLine(),
      search: {
        round: this.search.round,
        mmLevel: this.search.mmLevel,
        botSearchOffset: this.search.botSearchOffset,
        botMaxDepth: this.search.botMaxDepth(),
        paused: this.search.paused,
        userPaused: this.search.userPaused,
        statusLine: this.search.statusLine,
        bestBeatSec: this.search.bestBeatSec,
        maxTrainSec: this.search.maxTrainSec(),
        currentLabel: currentBot ? currentBot.label : "-",
        currentUsed: currentBot ? currentBot.secondsUsed : 0,
        rows: searchRows,
        lastTournament: this.search.lastTournament,
        tournamentLive: !!this.search.liveTourn,
        tournamentHeading: this.search.liveTourn ? "Current tournament" : "Latest tournament",
        history: this.search.history.slice(-5)
      },
      trainOne: {
        hasTrainee: !!this.trainOne.trainee,
        label: this.trainOne.trainee ? this.trainOne.trainee.label : null,
        mmLevel: this.trainOne.mmLevel,
        ladder: this.trainOne.ladder.slice(),
        statusLine: this.trainOne.statusLine,
        paused: this.trainOne.paused,
        userPaused: this.trainOne.userPaused,
        T_tourn: this.trainOne.T_tourn,
        lastTournament: this.trainOne.lastTournament,
        totalTrainSec: this.trainOne.totalTrainSec
      },
      catalog: this.unionCatalog().map(function (c) {
        return { id: c.id, label: c.label, family: c.family };
      })
    };
  };

  ShapeLab.prototype.publish = function () {
    if (this.onSnapshot) {
      this.onSnapshot(this.getSnapshot());
    }
  };

  ShapeLab.prototype.tick = function () {
    if (this.search.liveTourn) {
      this.search.tickLiveTournament(120);
      return;
    }
    /* Fill-complete rounds start the tournament even if Search is paused. */
    if (this.search.maybeStartTournament()) {
      return;
    }
    if (this.activeTab === "search" && !this.search.paused) {
      this.search.tick(120);
    } else if (this.activeTab === "train" && !this.trainOne.paused) {
      this.trainOne.tick(120, this.contenderEntriesForTrain());
    }
  };

  ShapeLab.prototype.start = function () {
    var self = this;
    if (this.running) {
      return;
    }
    this.running = true;
    this.bindUnloadSave();
    this.scheduleAutoSave();
    function loop() {
      if (!self.running) {
        return;
      }
      try {
        self.tick();
        var now = performance.now();
        var tournDirty = self.search._tournUiDirty;
        var uiInterval = self.search.liveTourn ? TOURN_UI_MIN_MS : 300;
        if (tournDirty || now - self.lastUi > uiInterval) {
          self.publish();
          self.lastUi = now;
          self.search._tournUiDirty = false;
        }
      } catch (err) {
        if (typeof console !== "undefined" && console.error) {
          console.error(err);
        }
        self.search.statusLine = "Error: " + (err && err.message ? err.message : err);
        self.publish();
      }
      self.rafId = requestAnimationFrame(loop);
    }
    this.rafId = requestAnimationFrame(loop);
    this.publish();
  };

  ShapeLab.prototype.stop = function () {
    this.running = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this._autoSaveTimer !== null) {
      clearInterval(this._autoSaveTimer);
      this._autoSaveTimer = null;
    }
  };

  ShapeLab.prototype.scheduleAutoSave = function () {
    var self = this;
    if (this._autoSaveTimer !== null) {
      return;
    }
    this._autoSaveTimer = setInterval(function () {
      if (self.running) {
        self.saveLocal();
      }
    }, AUTO_SAVE_MS);
  };

  ShapeLab.prototype.bindUnloadSave = function () {
    var self = this;
    if (this._unloadBound || typeof global.addEventListener !== "function") {
      return;
    }
    this._unloadBound = true;
    function flush() {
      self.saveLocal();
    }
    global.addEventListener("pagehide", flush);
    global.addEventListener("beforeunload", flush);
    global.addEventListener("visibilitychange", function () {
      if (global.document && global.document.visibilityState === "hidden") {
        flush();
      }
    });
  };

  ShapeLab.prototype.pickPlayMove = function (botId, board, player, budgetSec) {
    var ms = (budgetSec || 0) * 1000;
    this._lastPlayDepth = 0;
    if (botId === "minimax" || botId === "mcts" || botId === "random") {
      var bases = this.ensurePlayBaselines();
      var base = bases[botId];
      if (!base || !base.agent) {
        return null;
      }
      if (botId === "minimax") {
        var mmMove = base.agent.chooseMove(board, player, ms);
        this._lastPlayDepth = base.agent.lastDepth || 0;
        return mmMove;
      }
      if (botId === "mcts") {
        return base.agent.chooseMove(board, player, false, ms);
      }
      return base.agent.chooseMove(board, player);
    }
    var cat = this.unionCatalog();
    var i;
    for (i = 0; i < cat.length; i += 1) {
      if (cat[i].id === botId) {
        var agent = cat[i].agent;
        if (!agent && cat[i].blob) {
          agent = APP.shapeAgentFromDict(cat[i].blob);
        }
        var learnedMove = APP.shapeAgentChoose(agent, cat[i].family, board, player, false, ms);
        this._lastPlayDepth = (agent && agent.lastDepth) || 0;
        return learnedMove;
      }
    }
    if (this.trainOne.trainee && this.trainOne.trainee.id === botId) {
      var trainee = this.trainOne.trainee;
      var traineeMove = APP.shapeAgentChoose(
        trainee.agent,
        trainee.family,
        board,
        player,
        false,
        ms
      );
      this._lastPlayDepth = (trainee.agent && trainee.agent.lastDepth) || 0;
      return traineeMove;
    }
    return null;
  };

  ShapeLab.prototype.buildCheckpoint = function () {
    var t = this.trainOne;
    var trainPayload = null;
    if (t.trainee) {
      trainPayload = {
        id: t.trainee.id,
        sourceId: t.trainee.imported ? null : (t.trainee.sourceId || null),
        imported: !!t.trainee.imported,
        family: t.trainee.family,
        layerSizes: cloneSizes(t.trainee.layerSizes),
        learningRate: t.trainee.learningRate,
        label: t.trainee.label,
        blob: APP.shapeAgentToDict(t.trainee.agent, t.trainee.family),
        mmLevel: t.mmLevel,
        ladder: t.ladder.slice(),
        T_tourn: t.T_tourn,
        trainSinceTourn: t.trainSinceTourn,
        totalTrainSec: t.totalTrainSec,
        userPaused: t.userPaused,
        lastTournament: t.lastTournament,
        warmupLeftMs: t.trainee.warmupLeftMs,
        trainSinceEvalMs: t.trainee.trainSinceEvalMs,
        trainBetweenTargetMs: t.trainee.trainBetweenTargetMs,
        trainBetweenLocked: !!t.trainee.trainBetweenLocked,
        evalSeconds: t.trainee.evalSeconds || 0,
        beatStreak: typeof t.trainee.beatStreak === "number" ? t.trainee.beatStreak : 0,
        peerBeatIds: Array.isArray(t.peerBeatIds) ? t.peerBeatIds.slice() : [],
        peerPlayCounts: t.trainee.peerPlayCounts || {}
      };
    }
    return {
      format: CHECKPOINT_FORMAT,
      savedAt: new Date().toISOString(),
      activeTab: this.activeTab,
      nextId: nextId,
      neatSerialByFamily: {
        neat: neatSerialByFamily.neat || 0,
        neat_value: neatSerialByFamily.neat_value || 0
      },
      search: {
        round: this.search.round,
        mmLevel: this.search.mmLevel,
        botSearchOffset: this.search.botSearchOffset,
        bestBeatSec: this.search.bestBeatSec,
        userPaused: this.search.userPaused,
        history: this.search.history,
        lastTournament: this.search.lastTournament,
        tournamentSnapshots: this.search.tournamentSnapshots,
        population: this.search.population.map(serializeIndividual)
      },
      trainOne: trainPayload
    };
  };

  ShapeLab.prototype.applyCheckpoint = function (data) {
    if (!data || !data.search) {
      return false;
    }
    if (typeof data.nextId === "number" && data.nextId > nextId) {
      nextId = data.nextId;
    }
    if (data.neatSerialByFamily) {
      neatSerialByFamily.neat = Math.max(
        neatSerialByFamily.neat || 0,
        data.neatSerialByFamily.neat || 0
      );
      neatSerialByFamily.neat_value = Math.max(
        neatSerialByFamily.neat_value || 0,
        data.neatSerialByFamily.neat_value || 0
      );
    }
    var s = data.search;
    this.search.round = s.round || 1;
    this.search.mmLevel = typeof s.mmLevel === "number" ? s.mmLevel : 0;
    this.setBotSearchOffset(
      typeof s.botSearchOffset === "number" ? s.botSearchOffset : 0
    );
    this.search.bestBeatSec = s.bestBeatSec === undefined ? null : s.bestBeatSec;
    this.search.userPaused = !!s.userPaused;
    this.search.paused = this.search.userPaused;
    this.search.history = s.history || [];
    this.search.lastTournament = s.lastTournament || null;
    this.search.tournamentSnapshots = s.tournamentSnapshots || [];
    if (s.population && s.population.length) {
      this.search.population = s.population.map(deserializeIndividual);
    }
    this.search.statusLine = "Restored round " + this.search.round + " " +
      formatMmTag(this.search.mmLevel);

    if (this.search.maybeStartTournament()) {
      this.search.statusLine = "Restored fill-complete round - running tournament";
    }

    if (data.trainOne && data.trainOne.blob) {
      var tp = data.trainOne;
      bumpNextIdFromString(tp.id);
      var trainLr = typeof tp.learningRate === "number" ?
        tp.learningRate : APP.defaultLearningRate(tp.family);
      var trainAgent = APP.shapeAgentFromDict(tp.blob);
      if (typeof trainLr === "number") {
        APP.applyShapeLearningRate(trainAgent, tp.family, trainLr);
      }
      this.trainOne.trainee = {
        id: tp.id,
        sourceId: tp.imported ? null : tp.sourceId,
        imported: !!tp.imported,
        family: tp.family,
        layerSizes: cloneSizes(tp.layerSizes),
        learningRate: trainLr,
        label: tp.label,
        agent: trainAgent,
        warmupLeftMs: typeof tp.warmupLeftMs === "number" ? tp.warmupLeftMs : WARMUP_MS,
        trainSinceEvalMs: typeof tp.trainSinceEvalMs === "number" ? tp.trainSinceEvalMs : 0,
        trainBetweenTargetMs: typeof tp.trainBetweenTargetMs === "number" ?
          tp.trainBetweenTargetMs : TRAIN_BETWEEN_EVAL_MS,
        trainBetweenLocked: !!tp.trainBetweenLocked,
        evalSeconds: tp.evalSeconds || 0,
        beatStreak: typeof tp.beatStreak === "number" ? tp.beatStreak : 0,
        beaten: false,
        peerPlayCounts: tp.peerPlayCounts || {},
        nemesisSeeds: []
      };
      this.trainOne.mmLevel = typeof tp.mmLevel === "number" ? tp.mmLevel : 0;
      this.trainOne.ladder = tp.ladder || [];
      this.trainOne.T_tourn = tp.T_tourn === undefined ? null : tp.T_tourn;
      this.trainOne.trainSinceTourn = tp.trainSinceTourn || 0;
      this.trainOne.totalTrainSec = tp.totalTrainSec || 0;
      this.trainOne.userPaused = !!tp.userPaused;
      this.trainOne.paused = this.trainOne.userPaused;
      this.trainOne.lastTournament = tp.lastTournament || null;
      this.trainOne.peerBeatIds = Array.isArray(tp.peerBeatIds) ? tp.peerBeatIds.slice() : [];
      if (!this.trainOne.peerBeatIds.length && tp.lastTournament) {
        this.trainOne.peerBeatIds = peersWhoBeatTrainee(tp.lastTournament, tp.id);
      }
      this.trainOne.statusLine = "Restored " + tp.label;
    }

    if (data.activeTab) {
      this.activeTab = data.activeTab;
      if (this.activeTab !== "search") {
        this.search.setPaused(true, false);
      } else {
        this.search.resumeIfAllowed();
      }
      if (this.activeTab !== "train") {
        this.trainOne.setPaused(true, false);
      } else {
        this.trainOne.resumeIfAllowed();
      }
    }
    return true;
  };

  ShapeLab.prototype.saveLocal = function (done) {
    var self = this;
    var payload;
    var json;
    try {
      payload = this.buildCheckpoint();
      json = JSON.stringify(payload);
    } catch (buildErr) {
      var failBuild = {
        ok: false,
        error: String(buildErr && buildErr.message ? buildErr.message : buildErr)
      };
      this.lastSaveStatus = failBuild;
      if (done) {
        done(failBuild);
      }
      return failBuild;
    }

    var lsOk = false;
    var lsError = null;
    try {
      global.localStorage.setItem(STORAGE_KEY, json);
      lsOk = true;
    } catch (lsErr) {
      lsError = String(lsErr && lsErr.message ? lsErr.message : lsErr);
    }

    function finish(ok, error, backend) {
      var status = {
        ok: ok,
        error: error || null,
        backend: backend || null,
        bytes: json.length,
        pending: false
      };
      self.lastSaveStatus = status;
      if (ok) {
        self.lastSaveAt = performance.now();
      }
      if (done) {
        done(status);
      }
      return status;
    }

    if (idbSupported()) {
      this.lastSaveStatus = { ok: true, pending: true, bytes: json.length };
      idbSetShape(json, function (err) {
        if (!err) {
          finish(true, null, "indexedDB");
          return;
        }
        if (lsOk) {
          finish(true, null, "localStorage");
          return;
        }
        finish(false, String(err && err.message ? err.message : err) || lsError, null);
      });
      return { ok: true, pending: true, bytes: json.length };
    }

    if (lsOk) {
      return finish(true, null, "localStorage");
    }
    return finish(false, lsError || "save failed", null);
  };

  ShapeLab.prototype.loadLocal = function (done) {
    var self = this;
    function applyRaw(raw, backend) {
      if (!raw) {
        if (done) {
          done(null, null);
        }
        return;
      }
      try {
        var data = typeof raw === "string" ? JSON.parse(raw) : raw;
        self.applyCheckpoint(data);
        self.lastSaveStatus = {
          ok: true,
          backend: backend,
          bytes: typeof raw === "string" ? raw.length : JSON.stringify(raw).length,
          pending: false
        };
        self.lastSaveAt = performance.now();
        if (done) {
          done(null, data);
        }
      } catch (err) {
        if (done) {
          done(err, null);
        }
      }
    }

    if (idbSupported()) {
      idbGetShape(function (err, raw) {
        if (!err && raw) {
          applyRaw(raw, "indexedDB");
          return;
        }
        try {
          applyRaw(global.localStorage.getItem(STORAGE_KEY), "localStorage");
        } catch (lsErr) {
          if (done) {
            done(lsErr, null);
          }
        }
      });
      return;
    }
    try {
      applyRaw(global.localStorage.getItem(STORAGE_KEY), "localStorage");
    } catch (lsErr) {
      if (done) {
        done(lsErr, null);
      }
    }
  };

  ShapeLab.prototype.clearLocal = function (done) {
    try {
      global.localStorage.removeItem(STORAGE_KEY);
    } catch (ignore) {}
    idbDeleteShape(function () {
      if (done) {
        done(null);
      }
    });
  };

  ShapeLab.prototype.resetAll = function () {
    nextId = 1;
    neatSerialByFamily = { neat: 0, neat_value: 0 };
    this.search = new ShapeSearch();
    this.trainOne = new TrainOne();
    this.activeTab = "search";
    this.clearLocal();
    this.saveLocal();
    this.publish();
  };

  ShapeLab.prototype.exportTrainee = function () {
    if (!this.trainOne.trainee) {
      return { ok: false, error: "No trainee loaded. Start / load a bot first." };
    }
    var t = this.trainOne;
    var safeName = String(t.trainee.label || t.trainee.family || "bot")
      .replace(/[^\w\-]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 48) || "bot";
    var payload = {
      format: EXPORT_FORMAT,
      exportedAt: new Date().toISOString(),
      id: t.trainee.id,
      sourceId: t.trainee.imported ? null : (t.trainee.sourceId || null),
      imported: !!t.trainee.imported,
      family: t.trainee.family,
      layerSizes: cloneSizes(t.trainee.layerSizes),
      learningRate: t.trainee.learningRate,
      label: t.trainee.label,
      mmLevel: t.mmLevel,
      ladder: t.ladder.slice(),
      totalTrainSec: t.totalTrainSec,
      blob: APP.shapeAgentToDict(t.trainee.agent, t.trainee.family)
    };
    return {
      ok: true,
      filename: "c4-" + safeName + ".json",
      payload: payload,
      json: JSON.stringify(payload, null, 2)
    };
  };

  /**
   * Load an exported Train One bot as the current trainee.
   * The caller supplies the display name. The bot stays paused until Resume.
   * Save to Search adds it as a new population member (no original on this device).
   */
  ShapeLab.prototype.importTrainee = function (payload, name) {
    if (!payload || payload.format !== EXPORT_FORMAT) {
      return { ok: false, error: "Not a Train One bot export." };
    }
    if (!payload.family || !payload.blob) {
      return { ok: false, error: "Export is missing bot weights." };
    }
    var agent;
    try {
      agent = APP.shapeAgentFromDict(payload.blob);
    } catch (ignore) {
      agent = null;
    }
    if (!agent) {
      return { ok: false, error: "Could not read bot weights." };
    }
    var label = String(name == null ? "" : name).trim().replace(/\s+/g, " ").slice(0, 80);
    if (!label) {
      label = String(payload.label || "").trim().replace(/\s+/g, " ").slice(0, 80);
    }
    if (!label) {
      label = familyLabel(payload.family, payload.layerSizes);
    }
    var lr = typeof payload.learningRate === "number" ?
      payload.learningRate : APP.defaultLearningRate(payload.family);
    APP.applyShapeLearningRate(agent, payload.family, lr);
    var t = this.trainOne;
    t.trainee = {
      id: uid("train"),
      sourceId: null,
      imported: true,
      family: payload.family,
      layerSizes: cloneSizes(payload.layerSizes),
      learningRate: lr,
      label: label,
      agent: agent,
      warmupLeftMs: WARMUP_MS,
      evalSeconds: 0,
      beatStreak: 0,
      beaten: false,
      nemesisSeeds: [],
      peerPlayCounts: {},
      lastEvalWinRate: null,
      bestWinRate: null,
      bestBlob: null,
      evalGamesTarget: EVAL_GAMES
    };
    initTrainSchedule(t.trainee);
    t.mmLevel = typeof payload.mmLevel === "number" ? payload.mmLevel : 0;
    t.ladder = Array.isArray(payload.ladder) ? payload.ladder.slice() : [];
    t.T_tourn = null;
    t.trainSinceTourn = 0;
    t.lastTournament = null;
    t.peerBeatIds = [];
    t.totalTrainSec = typeof payload.totalTrainSec === "number" ? payload.totalTrainSec : 0;
    t.startedAt = performance.now();
    t.setPaused(true, true);
    t.statusLine = "Imported " + label;
    this.saveLocal();
    this.publish();
    return { ok: true, label: label };
  };

  /**
   * Copy trainee weights into Shape Search.
   * A bot loaded from this session overwrites its original.
   * An imported bot is added under its name; a later save overwrites that copy.
   */
  ShapeLab.prototype.promoteTraineeToSearch = function () {
    var t = this.trainOne && this.trainOne.trainee;
    if (!t) {
      return { ok: false, error: "No trainee loaded. Start / load a bot first." };
    }
    if (t.imported) {
      return this.addImportedTraineeToSearch(t);
    }
    var sourceId = t.sourceId;
    if (!sourceId) {
      return { ok: false, error: "Trainee has no source bot id." };
    }
    var blob = APP.shapeAgentToDict(t.agent, t.family);
    var lr = typeof t.learningRate === "number" ?
      t.learningRate : APP.defaultLearningRate(t.family);
    var updatedLive = false;
    var updatedSnap = false;
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      var ind = this.search.population[i];
      if (ind.id !== sourceId) {
        continue;
      }
      if (ind.family !== t.family) {
        return { ok: false, error: "Trainee family no longer matches Search bot." };
      }
      var agent = APP.shapeAgentFromDict(blob);
      if (typeof lr === "number") {
        APP.applyShapeLearningRate(agent, ind.family, lr);
      }
      ind.agent = agent;
      ind.learningRate = lr;
      ind.layerSizes = cloneSizes(t.layerSizes);
      ind.bestBlob = JSON.parse(JSON.stringify(blob));
      if (typeof t.bestWinRate === "number") {
        ind.bestWinRate = t.bestWinRate;
      }
      /* New weights: clear beat/eval so Search re-grades. */
      ind.beaten = false;
      ind.beatStreak = 0;
      ind.secondsToBeat = null;
      ind.lastEvalWins = null;
      ind.lastEvalDraws = null;
      ind.lastEvalGames = null;
      ind.lastEvalWinRate = null;
      updatedLive = true;
      break;
    }
    for (i = 0; i < this.search.tournamentSnapshots.length; i += 1) {
      var snap = this.search.tournamentSnapshots[i];
      if (snap.id !== sourceId) {
        continue;
      }
      snap.blob = JSON.parse(JSON.stringify(blob));
      snap.learningRate = lr;
      snap.layerSizes = cloneSizes(t.layerSizes);
      snap.family = t.family;
      updatedSnap = true;
      break;
    }
    if (!updatedLive && !updatedSnap) {
      return {
        ok: false,
        error: "Original bot is not in Shape Search (gone from population and snapshots)."
      };
    }
    this.trainOne.statusLine = (t.label || "trainee") + " saved over Search original";
    this.saveLocal();
    this.publish();
    return {
      ok: true,
      sourceId: sourceId,
      updatedLive: updatedLive,
      updatedSnap: updatedSnap
    };
  };

  /** Insert an imported trainee as a new Shape Search bot, then link later saves to it. */
  ShapeLab.prototype.addImportedTraineeToSearch = function (t) {
    var blob = APP.shapeAgentToDict(t.agent, t.family);
    var lr = typeof t.learningRate === "number" ?
      t.learningRate : APP.defaultLearningRate(t.family);
    var agent = APP.shapeAgentFromDict(blob);
    if (!agent) {
      return { ok: false, error: "Could not copy trainee weights." };
    }
    if (typeof lr === "number") {
      APP.applyShapeLearningRate(agent, t.family, lr);
    }
    var ind = createIndividual({
      family: t.family,
      layerSizes: cloneSizes(t.layerSizes),
      learningRate: lr,
      agent: agent
    });
    /* User name wins over the family or NEAT serial label. */
    ind.label = t.label;
    ind.bestBlob = JSON.parse(JSON.stringify(blob));
    if (typeof t.bestWinRate === "number") {
      ind.bestWinRate = t.bestWinRate;
    }
    this.search.population.push(ind);
    t.sourceId = ind.id;
    t.imported = false;
    this.trainOne.statusLine = (t.label || "trainee") + " added to Shape Search";
    this.saveLocal();
    this.publish();
    return {
      ok: true,
      added: true,
      sourceId: ind.id,
      updatedLive: true,
      updatedSnap: false
    };
  };

  /* Pure helpers exported for tests */
  global.C4_SHAPE_LAB = {
    STORAGE_KEY: STORAGE_KEY,
    CHECKPOINT_FORMAT: CHECKPOINT_FORMAT,
    EXPORT_FORMAT: EXPORT_FORMAT,
    AUTO_SAVE_MS: AUTO_SAVE_MS,
    SLICE_MS: SLICE_MS,
    MIN_DWELL_SEC: MIN_DWELL_SEC,
    nextTrainSliceTarget: nextTrainSliceTarget,
    botsBehindLeader: botsBehindLeader,
    EVAL_GAMES: EVAL_GAMES,
    EVAL_GAMES_LONG: EVAL_GAMES_LONG,
    pickShapeTrainKind: pickShapeTrainKind,
    peersWhoBeatTrainee: peersWhoBeatTrainee,
    addNemesisSeed: addNemesisSeed,
    NEMESIS_MAX: NEMESIS_MAX,
    formatMmTag: formatMmTag,
    isRandomMmLevel: isRandomMmLevel,
    wrBeats: wrBeats,
    pointsWinRate: pointsWinRate,
    BEAT_WR: BEAT_WR,
    BEAT_STREAK_NEED: BEAT_STREAK_NEED,
    noteBeatStreak: noteBeatStreak,
    streakComplete: streakComplete,
    BEST_WR_DIP: BEST_WR_DIP,
    saveBestWeights: saveBestWeights,
    restoreBestWeights: restoreBestWeights,
    evalSavedBlob: evalSavedBlob,
    applyKeepBestAfterEval: applyKeepBestAfterEval,
    runBotEval: runBotEval,
    WARMUP_MS: WARMUP_MS,
    TRAIN_BETWEEN_EVAL_MS: TRAIN_BETWEEN_EVAL_MS,
    TRAIN_TO_EVAL_RATIO: TRAIN_TO_EVAL_RATIO,
    adaptTrainBetweenTarget: adaptTrainBetweenTarget,
    initTrainSchedule: initTrainSchedule,
    MAX_SURVIVORS: MAX_SURVIVORS,
    MAX_MUTANT_PARENTS: MAX_MUTANT_PARENTS,
    MAX_SURVIVORS_FIRST: MAX_SURVIVORS_FIRST,
    MAX_SURVIVORS_LATER: MAX_SURVIVORS_LATER,
    HISTORY_LEARNED_LIMIT: HISTORY_LEARNED_LIMIT,
    HISTORY_BASELINE_TOP_N: HISTORY_BASELINE_TOP_N,
    MAX_NEAT_COPIES: MAX_NEAT_COPIES,
    survivorLimitForRound: survivorLimitForRound,
    pickMutantParents: pickMutantParents,
    MIN_TRAIN_SEC: MIN_TRAIN_SEC,
    MIN_TRAIN_SEC_RANDOM: MIN_TRAIN_SEC_RANDOM,
    FILL_AFTER_BEAT_SEC: FILL_AFTER_BEAT_SEC,
    gen0Specs: gen0Specs,
    mutateWidth: mutateWidth,
    mutateDepth: mutateDepth,
    mutateFrom: mutateFrom,
    clampLr: clampLr,
    flipHeadFamily: flipHeadFamily,
    sizesForHeadFlip: sizesForHeadFlip,
    copyOverlapNet: copyOverlapNet,
    cloneAgentFromParent: cloneAgentFromParent,
    padNeatClones: cloneNeatSurvivors,
    cloneNeatSurvivors: cloneNeatSurvivors,
    assignNeatIdentity: assignNeatIdentity,
    neatDisplayName: neatDisplayName,
    isNeatFamily: isNeatFamily,
    sizeSignature: sizeSignature,
    layersNearDuplicate: layersNearDuplicate,
    isNearDuplicateShape: isNearDuplicateShape,
    tryAddUniqueShape: tryAddUniqueShape,
    idAge: idAge,
    NEAR_WIDTH_DUP: NEAR_WIDTH_DUP,
    familyLabel: familyLabel,
    isMorphable: isMorphable,
    createIndividual: createIndividual,
    resetIndividualProgress: resetIndividualProgress,
    clampBotSearchOffset: clampBotSearchOffset,
    runRoundRobin: runRoundRobin,
    createRoundRobinSession: createRoundRobinSession,
    stepRoundRobinSession: stepRoundRobinSession,
    roundRobinStandings: roundRobinStandings,
    applyMatchResult: applyMatchResult,
    formatBeatMinimaxSuffix: formatBeatMinimaxSuffix,
    isMinimaxEntry: isMinimaxEntry,
    topLearnedRanks: topLearnedRanks,
    historyTopRanks: historyTopRanks,
    isBaselineRank: isBaselineRank,
    TOURN_UI_MIN_MS: TOURN_UI_MIN_MS,
    makeBaselineEntries: makeBaselineEntries,
    ShapeSearch: ShapeSearch,
    TrainOne: TrainOne,
    ShapeLab: ShapeLab,
    init: function (options) {
      var lab = new ShapeLab(options);
      lab.loadLocal(function () {
        lab.start();
      });
      /* If load is sync (no IDB), start may already be needed immediately.
         loadLocal always invokes done; start() is idempotent-guarded. */
      return lab;
    }
  };
})(typeof window !== "undefined" ? window : global);
