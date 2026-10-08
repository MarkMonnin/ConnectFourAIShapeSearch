/**
 * Shape-lab unit tests: mutations, train/eval helpers, search/train-one mechanics.
 */
var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

global.window = global;
global.localStorage = {
  _data: {},
  getItem: function (k) { return this._data[k] || null; },
  setItem: function (k, v) { this._data[k] = String(v); },
  removeItem: function (k) { delete this._data[k]; }
};
global.performance = { now: function () { return Date.now(); } };
global.requestAnimationFrame = function () { return 0; };
global.cancelAnimationFrame = function () {};
global.setInterval = function () { return 0; };
global.clearInterval = function () {};
global.addEventListener = function () {};

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-network.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-opponents.js"), "utf8"));
require(path.join(root, "c4_js/tfjs-agent.js"));
require(path.join(root, "c4_js/engine.js"));
require(path.join(root, "c4_js/shape-lab.js"));

var APP = global.C4_APP;
var LAB = global.C4_SHAPE_LAB;
var G = global.C4_GAME;
var C = global.C4_CONSTANTS;

var failures = [];

function fail(msg) {
  failures.push(msg);
}

function assert(cond, msg) {
  if (!cond) {
    fail(msg);
  }
}

function assertClose(a, b, eps, msg) {
  if (Math.abs(a - b) > (eps || 1e-6)) {
    fail(msg + " got " + a + " expected ~" + b);
  }
}

/* --- Engine shape helpers --- */
(function testCreateAgents() {
  var families = Object.keys(APP.SHAPE_FAMILIES);
  var i;
  for (i = 0; i < families.length; i += 1) {
    var f = families[i];
    try {
      var agent = APP.createShapeAgent(f, APP.defaultLayerSizes(f));
      assert(!!agent, "createShapeAgent " + f);
    } catch (err) {
      fail("createShapeAgent " + f + ": " + err.message);
    }
  }
})();

(function testTrainRandomAndSelf() {
  var agent = APP.createShapeAgent("reinforce_value", APP.defaultLayerSizes("reinforce_value"));
  var r1 = APP.shapeTrainVsRandom(agent, "reinforce_value", true, true);
  assert(r1 === "win" || r1 === "loss" || r1 === "draw", "trainVsRandom result");
  APP.shapeTrainVsSelf(agent, "reinforce_value", true);
  assert(agent.trainSteps > 0, "self-play should learn (trainSteps>0)");
  var vsMm = APP.shapeTrainVsMinimax(agent, "reinforce_value", 1, true, true, 1, 99);
  assert(vsMm.result === "win" || vsMm.result === "draw" || vsMm.result === "loss", "train vs MM result");
  assert(vsMm.seed === 99, "train vs MM keeps seed");
})();

(function testEvalTimed() {
  var agent = APP.createShapeAgent("nn_value", APP.defaultLayerSizes("nn_value"));
  var ev = APP.shapeEvalVsMinimaxTimed(agent, "nn_value", 1, 6);
  assert(ev.wins + ev.draws + ev.losses === 6, "eval game count");
  assert(ev.games === 6, "eval.games set");
  assert(typeof ev.winRate === "number", "eval winRate");
  assert(ev.winRate >= 0 && ev.winRate <= 1, "winRate bounds");
  var evDepth = APP.shapeEvalVsMinimaxTimed(agent, "nn_value", 2, 2, 1);
  assert(evDepth.wins + evDepth.draws + evDepth.losses === 2, "depth-capped eval games");
})();

(function testEvalShortPassesExtendOptsAndSavesBest() {
  var origEval = APP.shapeEvalVsMinimaxTimed;
  var seenGames = null;
  var seenOpts = "unset";
  APP.shapeEvalVsMinimaxTimed = function (a, f, level, evalGames, maxDepth, opts) {
    seenGames = evalGames;
    seenOpts = opts;
    return { wins: 14, draws: 0, losses: 6, games: evalGames, winRate: 0.7, extended: false };
  };
  try {
    var ind = LAB.createIndividual({
      family: "nn_value",
      layerSizes: APP.defaultLayerSizes("nn_value")
    });
    var packed = LAB.runBotEval(ind, 1, 1);
    assert(seenGames === LAB.EVAL_GAMES, "eval starts at short game count");
    assert(seenOpts && seenOpts.extendTo === LAB.EVAL_GAMES_LONG, "passes 20->100 extend opts");
    assert(seenOpts.extendIfWr === LAB.BEAT_WR, "extend if WR > 50%");
    assert(packed.ev.games === LAB.EVAL_GAMES, "mock short result stays short");
    assert(ind.bestWinRate === 0.7, "saved best WR on short eval");
    assert(!!ind.bestBlob, "bestBlob saved");
  } finally {
    APP.shapeEvalVsMinimaxTimed = origEval;
  }
})();

(function testKeepBestSavesOnTieAfterPlayoffKeepsCurrent() {
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  ind.evalGamesTarget = LAB.EVAL_GAMES_LONG;
  ind.agent.net.weights[0][0][0] = 1.25;
  LAB.saveBestWeights(ind, 0.80);
  ind.agent.net.weights[0][0][0] = 9.99;
  var orig = APP.shapeCompareKeepBest;
  APP.shapeCompareKeepBest = function () { return "current"; };
  try {
    var note = LAB.applyKeepBestAfterEval(ind, {
      wins: 80, draws: 0, losses: 20, games: 100, winRate: 0.80
    }, 1, 1);
    assert(note === "saved tie", "tie playoff keep current saves later weights");
    var restored = APP.shapeAgentFromDict(ind.bestBlob);
    assert(Math.abs(restored.net.weights[0][0][0] - 9.99) < 1e-9, "tie saved later weights");
  } finally {
    APP.shapeCompareKeepBest = orig;
  }
})();

(function testKeepBestPlayoffRestoresIfSavedWins() {
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  ind.agent.net.weights[0][0][0] = 1.25;
  LAB.saveBestWeights(ind, 0.80);
  ind.agent.net.weights[0][0][0] = 9.99;
  var origEval = APP.shapeEvalVsMinimaxTimed;
  var origCmp = APP.shapeCompareKeepBest;
  APP.shapeEvalVsMinimaxTimed = function () {
    return { wins: 60, draws: 0, losses: 40, games: 100, winRate: 0.60, extended: false };
  };
  APP.shapeCompareKeepBest = function () { return "saved"; };
  try {
    ind.evalGamesTarget = LAB.EVAL_GAMES_LONG;
    var packed = LAB.runBotEval(ind, 1, 1);
    assert(packed.weightNote === "restored best", "playoff: saved bot");
    assert(Math.abs(ind.agent.net.weights[0][0][0] - 1.25) < 1e-9, "restored winner");
    assert(ind.bestWinRate === 0.80, "claimed best WR kept on restore");
    assert(ind.lastEvalWins === 60, "mmWins stays live dip");
  } finally {
    APP.shapeEvalVsMinimaxTimed = origEval;
    APP.shapeCompareKeepBest = origCmp;
  }
})();

(function testKeepBestPlayoffKeepsCurrentOnDip() {
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  ind.agent.net.weights[0][0][0] = 1.25;
  LAB.saveBestWeights(ind, 0.80);
  ind.agent.net.weights[0][0][0] = 9.99;
  var origEval = APP.shapeEvalVsMinimaxTimed;
  var origCmp = APP.shapeCompareKeepBest;
  APP.shapeEvalVsMinimaxTimed = function () {
    return { wins: 60, draws: 0, losses: 40, games: 100, winRate: 0.60, extended: false };
  };
  APP.shapeCompareKeepBest = function () { return "current"; };
  try {
    ind.evalGamesTarget = LAB.EVAL_GAMES_LONG;
    var packed = LAB.runBotEval(ind, 1, 1);
    assert(packed.weightNote === "kept current", "playoff: current bot");
    assert(Math.abs(ind.agent.net.weights[0][0][0] - 9.99) < 1e-9, "kept later brain");
    assert(ind.bestWinRate === 0.60, "best WR is live dip after keep current");
  } finally {
    APP.shapeEvalVsMinimaxTimed = origEval;
    APP.shapeCompareKeepBest = origCmp;
  }
})();

(function testShapeCompareKeepBestLadder() {
  var dummy = {};
  var origH = APP.shapeHeadToHeadTwoGames;
  var origR = APP.shapeEvalVsRandom;
  var origM = APP.shapeEvalVsMinimaxTimed;
  try {
    APP.shapeHeadToHeadTwoGames = function () {
      return { liveWins: 2, savedWins: 0, draws: 0 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", dummy) === "current", "H2H sweep current");

    APP.shapeHeadToHeadTwoGames = function () {
      return { liveWins: 0, savedWins: 2, draws: 0 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", dummy) === "saved", "H2H sweep saved");

    var randomCalls = 0;
    APP.shapeHeadToHeadTwoGames = function () {
      return { liveWins: 1, savedWins: 1, draws: 0 };
    };
    APP.shapeEvalVsRandom = function (agent) {
      randomCalls += 1;
      return agent === dummy ? { wins: 40, draws: 20, losses: 40 } : { wins: 10, draws: 0, losses: 90 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", { other: true }) === "current", "random points current");
    assert(randomCalls === 2, "random eval both bots");

    APP.shapeEvalVsRandom = function () {
      return { wins: 20, draws: 10, losses: 70 };
    };
    APP.shapeEvalVsMinimaxTimed = function (agent, family, level) {
      if (level !== 1) {
        throw new Error("should not reach MM2");
      }
      return agent === dummy ?
        { wins: 5, draws: 0, losses: 95 } :
        { wins: 0, draws: 0, losses: 100 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", { other: true }) === "current", "MM1 points current");

    APP.shapeEvalVsMinimaxTimed = function (agent, family, level) {
      if (level === 1) {
        return { wins: 0, draws: 0, losses: 100 };
      }
      throw new Error("MM1 0-0 should not MM2");
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", dummy) === "current", "MM1 tie at 0 keeps newer");

    APP.shapeEvalVsMinimaxTimed = function (agent, family, level) {
      if (level === 1) {
        return { wins: 10, draws: 0, losses: 90 };
      }
      return agent === dummy ?
        { wins: 1, draws: 0, losses: 99 } :
        { wins: 4, draws: 0, losses: 96 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", { other: true }) === "saved", "MM2 points saved");

    APP.shapeEvalVsMinimaxTimed = function (agent, family, level) {
      if (level === 1) {
        return { wins: 10, draws: 2, losses: 88 };
      }
      return { wins: 3, draws: 1, losses: 96 };
    };
    assert(APP.shapeCompareKeepBest(dummy, "nn_value", dummy) === "current", "MM2 tie keeps newer");
  } finally {
    APP.shapeHeadToHeadTwoGames = origH;
    APP.shapeEvalVsRandom = origR;
    APP.shapeEvalVsMinimaxTimed = origM;
  }
})();

(function testKeepBestPlayoffSkippedOnShortEval() {
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  ind.agent.net.weights[0][0][0] = 1.25;
  LAB.saveBestWeights(ind, 0.80);
  ind.agent.net.weights[0][0][0] = 9.99;
  ind.evalGamesTarget = LAB.EVAL_GAMES;
  var note = LAB.applyKeepBestAfterEval(ind, {
    wins: 10, draws: 0, losses: 10, games: 20, winRate: 0.50
  });
  assert(note === null, "no playoff on short eval when not a new best");
  assert(Math.abs(ind.agent.net.weights[0][0][0] - 9.99) < 1e-9, "weights untouched");
})();

(function testKeepBestWorksBelow50In100Mode() {
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  ind.evalGamesTarget = LAB.EVAL_GAMES_LONG;
  ind.beaten = false;
  var note = LAB.applyKeepBestAfterEval(ind, {
    wins: 45, draws: 0, losses: 55, games: 100, winRate: 0.45
  });
  assert(note === "saved best", "save best even under 50% in 100 mode");
  assert(ind.bestWinRate === 0.45, "best WR recorded");
})();

(function testBotSearchDepthForLevel() {
  assert(APP.botSearchDepthForLevel(5, 0) === 5, "Match MM5 -> depth 5");
  assert(APP.botSearchDepthForLevel(5, -2) === 3, "MM5 -2 -> depth 3");
  assert(APP.botSearchDepthForLevel(1, -10) === 1, "depth floor at 1");
  assert(APP.botSearchDepthForLevel(0.3, 0) === 1, "sub-1 level Match -> 1");
  assert(LAB.clampBotSearchOffset(-3) === -3, "clamp keeps -3");
  assert(LAB.clampBotSearchOffset(-99) === -10, "clamp floor -10");
  assert(LAB.clampBotSearchOffset(2) === 0, "clamp rejects positive");
  var search = new LAB.ShapeSearch();
  search.mmLevel = 4;
  search.setBotSearchOffset(-1);
  assert(search.botMaxDepth() === 3, "search botMaxDepth MM4-1");
})();

(function testCalibrateMm() {
  var ms = APP.calibrateMinimaxMoveMs(1, 3);
  assert(ms >= 0.5, "calibrate mm ms positive");
})();

(function testSerializeRoundTrip() {
  var agent = APP.createShapeAgent("nn_policy", APP.defaultLayerSizes("nn_policy"));
  APP.shapeTrainVsRandom(agent, "nn_policy", true, true);
  var blob = APP.shapeAgentToDict(agent, "nn_policy");
  assert(blob.family === "nn_policy", "blob family");
  var restored = APP.shapeAgentFromDict(blob);
  assert(!!restored, "restore agent");
  var move = APP.shapeAgentChoose(restored, "nn_policy", G.emptyBoard(), C.X, false, 5);
  assert(typeof move === "number", "restored can move");
})();

/* --- Mutations --- */
(function testMutateWidthDepth() {
  var base = [77, 64, 77, 1];
  var wider = LAB.mutateWidth(base, 1.5);
  assert(wider[1] > base[1], "wider hidden");
  assert(wider[0] === 77 && wider[3] === 1, "width keeps IO");
  var narrow = LAB.mutateWidth(base, 2 / 3);
  assert(narrow[1] < base[1], "narrower hidden");
  var deeper = LAB.mutateDepth(base, 1, "nn_value");
  assert(deeper.length === base.length + 1, "deeper adds layer");
  var shallower = LAB.mutateDepth(deeper, -1, "nn_value");
  assert(shallower.length === base.length, "shallower removes layer");
  assert(LAB.mutateDepth([77, 64, 1], -1, "nn_value") === null, "cannot go below 1 hidden");
})();

(function testMutateFromMorphable() {
  var ind = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value"),
    learningRate: 0.01
  });
  var kids = LAB.mutateFrom(ind);
  assert(kids.length >= 5, "morphable yields width/depth/LR kids");
  var lrs = kids.map(function (k) { return k.learningRate; });
  assert(lrs.indexOf(0.02) >= 0 || lrs.indexOf(0.005) >= 0, "LR mutation present");
  var heads = kids.map(function (k) { return k.family; });
  assert(heads.every(function (f) { return f === "reinforce_value"; }), "no policy/value head swap");
  var tab = LAB.createIndividual({ family: "qtable", layerSizes: null });
  assert(LAB.mutateFrom(tab).length === 0, "tabular no mutate");
})();

(function testNearDuplicateWithin3() {
  assert(LAB.layersNearDuplicate([78, 64, 78, 7], [78, 67, 75, 7]), "example within 3");
  assert(LAB.layersNearDuplicate([78, 64, 78, 7], [78, 64, 78, 7]), "exact match");
  assert(!LAB.layersNearDuplicate([78, 64, 78, 7], [78, 68, 78, 7]), "diff 4 is not near");
  assert(!LAB.layersNearDuplicate([78, 64, 78, 7], [78, 64, 78, 65, 7]), "different depth");
  assert(!LAB.isNearDuplicateShape(
    "reinforce_policy", [78, 64, 78, 7],
    "nn_policy", [78, 64, 78, 7]
  ), "different family");
  assert(LAB.isNearDuplicateShape(
    "reinforce_policy", [78, 64, 78, 7],
    "reinforce_policy", [78, 67, 75, 7]
  ), "same family near");
  assert(!LAB.isNearDuplicateShape("neat", null, "neat", null),
    "NEAT seats are not near-dups");
})();

(function testTryAddKeepsOldest() {
  var pop = [];
  var older = LAB.createIndividual({
    family: "reinforce_policy",
    layerSizes: [78, 64, 78, 7]
  }, "bot_10");
  var newer = LAB.createIndividual({
    family: "reinforce_policy",
    layerSizes: [78, 67, 75, 7]
  }, "bot_99");
  assert(LAB.tryAddUniqueShape(pop, newer), "add newer first");
  assert(LAB.tryAddUniqueShape(pop, older) === true, "older replaces newer");
  assert(pop.length === 1, "still one shape");
  assert(pop[0].id === "bot_10", "kept oldest id");
  var otherNewer = LAB.createIndividual({
    family: "reinforce_policy",
    layerSizes: [78, 65, 76, 7]
  }, "bot_50");
  assert(LAB.tryAddUniqueShape(pop, otherNewer) === false, "younger near-dup dropped");
  assert(pop[0].id === "bot_10", "oldest remains");
})();

(function testGen0() {
  var specs = LAB.gen0Specs();
  assert(specs.length === 28, "gen0 = 6 morph*3 + 4 tabular + 3 neat + 3 neat_value");
  var morphCounts = {};
  var neat = 0;
  var neatVal = 0;
  specs.forEach(function (s) {
    if (LAB.isMorphable(s.family)) {
      morphCounts[s.family] = (morphCounts[s.family] || 0) + 1;
    }
    if (s.family === "neat") {
      neat += 1;
    }
    if (s.family === "neat_value") {
      neatVal += 1;
    }
  });
  Object.keys(morphCounts).forEach(function (f) {
    assert(morphCounts[f] === 3, f + " has 3 width seeds");
  });
  assert(neat === LAB.MAX_NEAT_COPIES, "gen0 NEAT copies");
  assert(neatVal === LAB.MAX_NEAT_COPIES, "gen0 NEAT value copies");
  var families = specs.map(function (s) { return s.family; });
  assert(families.indexOf("reinforce_value") >= 0, "has reinforce_value");
  assert(families.indexOf("qtable") >= 0, "has qtable");
})();

(function testNeatNamedAndCloned() {
  assert(LAB.MAX_NEAT_COPIES === 3, "up to 3 NEAT copies");
  assert(LAB.MAX_SURVIVORS === 10, "keep top 10");
  assert(LAB.MAX_MUTANT_PARENTS === 3, "mutants from top 3");
  assert(LAB.survivorLimitForRound(1) === 10, "R1 keeps 10");
  assert(LAB.survivorLimitForRound(2) === 10, "later rounds keep 10");
  var n1 = LAB.createIndividual({ family: "neat", layerSizes: null });
  assert(n1.label === "NEAT 1" || /^NEAT \d+$/.test(n1.label), "named NEAT serial");
  var n2 = LAB.createIndividual({ family: "neat", layerSizes: null });
  assert(n1.neatSerial !== n2.neatSerial, "distinct neat serials");
  assert(n1.label !== n2.label, "distinct neat labels");

  var search = new LAB.ShapeSearch();
  var neat = LAB.createIndividual({ family: "neat", layerSizes: null });
  var other = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  search.population = [neat, other];
  var neatId = neat.id;
  var otherId = other.id;
  search.applyRoundAdvance({
    ranked: [
      { id: neatId, family: "neat", label: neat.label, points: 10 },
      { id: otherId, family: "reinforce_value", label: other.label, points: 8 }
    ]
  });
  var neats = search.population.filter(function (p) { return p.family === "neat"; });
  assert(neats.length === LAB.MAX_NEAT_COPIES, "NEAT top-3 parent -> self + clones");
  assert(neats.some(function (p) { return p.id === neatId; }), "survivor NEAT kept");
  var morph = search.population.filter(function (p) {
    return p.family === "reinforce_value";
  });
  assert(morph.some(function (p) { return p.id === otherId; }), "morph survivor kept");
  assert(morph.length > 1, "morph top-3 parent gets mutants");
  var morphSameSize = morph.filter(function (p) {
    return p.layerSizes && p.layerSizes.join(",") === other.layerSizes.join(",");
  });
  assert(morphSameSize.length === 1, "morph has no same-size weight clones");
  var labels = neats.map(function (p) { return p.label; });
  assert(labels.every(function (L) { return /^NEAT \d+$/.test(L); }), "NEAT N labels");
  assert(new Set(labels).size === labels.length, "unique NEAT labels");
})();

(function testNeatClonesOnlyForTopMutantParents() {
  var search = new LAB.ShapeSearch();
  var neatA = LAB.createIndividual({ family: "neat", layerSizes: null });
  var neatB = LAB.createIndividual({ family: "neat", layerSizes: null });
  var neatC = LAB.createIndividual({ family: "neat", layerSizes: null });
  var morph = [];
  var i;
  for (i = 0; i < 7; i += 1) {
    morph.push(LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: [77, 40 + i * 12, 39 + i * 12, 1]
    }));
  }
  search.population = [neatA, neatB, neatC].concat(morph);
  var ranked = [
    { id: neatA.id, family: "neat", points: 30 },
    { id: neatB.id, family: "neat", points: 28 },
    { id: morph[0].id, family: "reinforce_value", points: 26 },
    { id: neatC.id, family: "neat", points: 24 }
  ].concat(morph.slice(1).map(function (p, idx) {
    return { id: p.id, family: p.family, points: 20 - idx };
  }));
  search.applyRoundAdvance({ ranked: ranked });
  var neats = search.population.filter(function (p) { return p.family === "neat"; });
  /* A+B in top 3 -> clones; C kept in top 10 but no clones. */
  assert(neats.some(function (p) { return p.id === neatA.id; }), "kept A");
  assert(neats.some(function (p) { return p.id === neatB.id; }), "kept B");
  assert(neats.some(function (p) { return p.id === neatC.id; }), "kept C without clones");
  assert(neats.length === 2 * LAB.MAX_NEAT_COPIES + 1, "clones only for top-3 NEAT parents");
})();

(function testKeepTenMutateTopThree() {
  var search = new LAB.ShapeSearch();
  search.round = 2;
  search.mmLevel = 2;
  var bots = [];
  var widths = [40, 52, 64, 80, 96, 112, 128, 144, 160, 176, 192];
  var i;
  for (i = 0; i < widths.length; i += 1) {
    bots.push(LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: [77, widths[i], Math.max(8, widths[i] - 1), 1]
    }));
  }
  search.population = bots.slice();
  var ranked = bots.map(function (p, idx) {
    return { id: p.id, family: p.family, points: 100 - idx };
  });
  search.applyRoundAdvance({ ranked: ranked });
  var originals = search.population.filter(function (p) {
    return bots.some(function (b) { return b.id === p.id; });
  });
  assert(originals.length === LAB.MAX_SURVIVORS, "keeps top 10 survivors");
  var top3Ids = {};
  for (i = 0; i < LAB.MAX_MUTANT_PARENTS; i += 1) {
    top3Ids[bots[i].id] = true;
  }
  var children = search.population.filter(function (p) {
    return !bots.some(function (b) { return b.id === p.id; });
  });
  assert(children.length > 0, "top 3 spawn mutants");
  /* Rank 4 survivor should not produce a wider mutant unique to its width. */
  var rank4Width = bots[3].layerSizes[1];
  var widerFromRank4 = Math.max(8, Math.min(512, Math.round(rank4Width * 1.5)));
  var hasRank4Wider = children.some(function (p) {
    return p.layerSizes && p.layerSizes[1] === widerFromRank4 &&
      p.layerSizes[2] === Math.max(8, Math.round((bots[3].layerSizes[2]) * 1.5));
  });
  assert(!hasRank4Wider, "rank 4 does not spawn width mutants");
  var top1Wider = Math.max(8, Math.min(512, Math.round(bots[0].layerSizes[1] * 1.5)));
  assert(children.some(function (p) {
    return p.layerSizes && p.layerSizes[1] === top1Wider;
  }), "rank 1 spawns width mutant");
})();

(function testCalibrateTournamentBudget() {
  var ms = APP.calibrateTournamentMoveMs(1);
  assert(ms >= 1, "tournament budget at least 1ms");
  var plain = APP.calibrateMinimaxMoveMs(1, 1);
  assert(typeof plain === "number", "plain calibrate still works");
})();

(function testTrainBetweenEvals() {
  assert(LAB.TRAIN_BETWEEN_EVAL_MS === 50, "starts at 50ms");
  assert(LAB.TRAIN_TO_EVAL_RATIO === 4, "targets 4x eval");
  var ind = { trainBetweenTargetMs: 50, trainBetweenLocked: false };
  LAB.adaptTrainBetweenTarget(ind, 100);
  assert(ind.trainBetweenTargetMs === 100, "50 doubles toward 4*100=400");
  assert(!ind.trainBetweenLocked, "not locked yet");
  LAB.adaptTrainBetweenTarget(ind, 100);
  assert(ind.trainBetweenTargetMs === 200, "doubles again");
  LAB.adaptTrainBetweenTarget(ind, 100);
  assert(ind.trainBetweenTargetMs === 400, "reaches 4x and caps");
  assert(ind.trainBetweenLocked, "locked at 4x");
  LAB.adaptTrainBetweenTarget(ind, 80);
  assert(ind.trainBetweenTargetMs === 320, "locked tracks 4x latest eval");

  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    })
  ];
  search.population[0].warmupLeftMs = 0;
  search.population[0].trainSinceEvalMs = 0;
  search.population[0].trainBetweenTargetMs = 50;
  var evalCalls = 0;
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    evalCalls += 1;
    return { winRate: 0, wins: 0, draws: 0, losses: 1 };
  };
  try {
    search.tick(40);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  assert(evalCalls === 0, "no eval until train target banked");
  assert(search.population[0].secondsUsed > 0, "trained while waiting to eval");
})();

(function testWarmupBlocksEval() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    })
  ];
  search.population[0].warmupLeftMs = LAB.WARMUP_MS;
  var calls = 0;
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    calls += 1;
    return { winRate: 0, wins: 0, draws: 0, losses: 1 };
  };
  try {
    search.tick(40);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  assert(calls === 0, "warmup prevents eval");
  assert(search.population[0].secondsUsed > 0, "warmup still trains");
})();

(function testTrainTimeOnlyClock() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    })
  ];
  search.population[0].warmupLeftMs = 0;
  search.population[0].trainSinceEvalMs = 50;
  search.population[0].trainBetweenTargetMs = 50;
  var evalMs = 0;
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    var t0 = performance.now();
    while (performance.now() - t0 < 15) { /* burn wall */ }
    evalMs += performance.now() - t0;
    return { winRate: 0, wins: 0, draws: 0, losses: 1 };
  };
  try {
    search.tick(80);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  var ind = search.population[0];
  assert(ind.evalSeconds > 0, "eval seconds tracked separately");
  assert(ind.secondsUsed > 0, "train clock advanced");
  assert(Math.abs(ind.evalSeconds - evalMs / 1000) < 0.05, "evalSeconds matches eval wall");
  /* Train clock should not include the forced eval burn: evalSeconds is large from burn,
     train secondsUsed comes only from doTrain. */
  assert(ind.evalSeconds >= evalMs / 1000 - 0.02, "eval burn landed in evalSeconds");
})();

(function testBeatRequiresStreakOfTen() {
  assert(LAB.BEAT_WR === 0.5, "beat bar is 50%");
  assert(LAB.BEAT_STREAK_NEED === 10, "need 10 in a row");
  assert(!LAB.wrBeats(0.5), "exactly 50% is not a beat");
  assert(LAB.wrBeats(0.505), "over 50% beats");
  assert(Math.abs(LAB.pointsWinRate({
    wins: 8, draws: 5, losses: 7, games: 20
  }) - 0.525) < 1e-9, "draws count as 0.5");
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    })
  ];
  var ind = search.population[0];
  ind.warmupLeftMs = 0;
  ind.secondsUsed = 3.5;
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function (a, f, level, games, depth, opts) {
    if (games === LAB.EVAL_GAMES && opts && opts.extendTo === LAB.EVAL_GAMES_LONG) {
      return { wins: 55, draws: 0, losses: 45, games: 100, winRate: 0.55, extended: true };
    }
    return { wins: 55, draws: 0, losses: 45, games: 100, winRate: 0.55, extended: false };
  };
  try {
    var i;
    for (i = 0; i < 9; i += 1) {
      search.runIndividualEval(ind, 1, true);
      assert(ind.beatStreak === i + 1, "streak grows");
      assert(!ind.beaten, "not beaten before 10");
      assert(ind.evalGamesTarget === LAB.EVAL_GAMES_LONG, "graduated to 100-game evals");
    }
    assert(typeof ind.secondsToBeat === "number", "toBeat_s set on first 100-game WR>50%");
    assert(Math.abs(ind.secondsToBeat - 3.5) < 1e-9, "toBeat_s uses train_s at first clear");
    var firstToBeat = ind.secondsToBeat;
    ind.secondsUsed = 1.0;
    search.runIndividualEval(ind, 1, true);
    assert(ind.beatStreak === 10, "streak hits need");
    assert(ind.beaten, "10 in a row sets beaten");
    assert(ind.secondsToBeat === firstToBeat, "toBeat_s not lowered later in the round");

    APP.shapeEvalVsMinimaxTimed = function () {
      return { wins: 50, draws: 0, losses: 50, games: 100, winRate: 0.5, extended: false };
    };
    search.runIndividualEval(ind, 1, true);
    assert(ind.beatStreak === 0, "exact 50% resets streak");
    assert(!ind.beaten, "failed grade clears beaten");
    assert(ind.secondsToBeat === firstToBeat, "toBeat_s survives streak reset");
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
})();

(function testToBeatRequiresHundredGamesAndResetsEachRound() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 0;
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [ind];
  ind.warmupLeftMs = 0;
  ind.secondsUsed = 2;
  var origR = APP.shapeEvalVsRandom;
  APP.shapeEvalVsRandom = function () {
    return { wins: 12, draws: 0, losses: 8, games: 20, winRate: 0.6, extended: false };
  };
  try {
    search.runIndividualEval(ind, 0, true);
    assert(ind.secondsToBeat === null, "20-game clear without extend does not set toBeat_s");
    assert(ind.evalGamesTarget === LAB.EVAL_GAMES, "stay on 20 if mock did not extend");
  } finally {
    APP.shapeEvalVsRandom = origR;
  }

  APP.shapeEvalVsRandom = function (a, f, games, depth, opts) {
    if (opts && opts.extendTo === LAB.EVAL_GAMES_LONG) {
      return { wins: 40, draws: 0, losses: 60, games: 100, winRate: 0.4, extended: true };
    }
    return { wins: 40, draws: 0, losses: 60, games: 100, winRate: 0.4, extended: false };
  };
  try {
    search.runIndividualEval(ind, 0, true);
    assert(ind.evalGamesTarget === LAB.EVAL_GAMES_LONG, "failed extend still graduates to 100");
    assert(ind.secondsToBeat === null, "failed 100-game does not set toBeat_s");
  } finally {
    APP.shapeEvalVsRandom = origR;
  }

  APP.shapeEvalVsRandom = function () {
    return { wins: 60, draws: 0, losses: 40, games: 100, winRate: 0.6, extended: false };
  };
  try {
    search.runIndividualEval(ind, 0, true);
    assert(typeof ind.secondsToBeat === "number", "100-game clear sets toBeat_s");
    LAB.resetIndividualProgress(ind);
    assert(ind.secondsToBeat === null, "round reset clears toBeat_s");
    assert(ind.evalGamesTarget === LAB.EVAL_GAMES, "round reset returns to 20-game grades");
  } finally {
    APP.shapeEvalVsRandom = origR;
  }
})();

(function testFinishRoundKeepsWeightsAndTournamentSurvivors() {
  var search = new LAB.ShapeSearch();
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: [77, 64, 77, 1]
  });
  var b = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: [77, 64, 77, 1]
  });
  var c = LAB.createIndividual({ family: "qtable", layerSizes: null });
  search.population = [a, b, c];
  a.beaten = true;
  a.secondsToBeat = 1.5;
  b.beaten = true;
  b.secondsToBeat = 2.0;
  c.beaten = true;
  c.secondsToBeat = 3.0;
  search.bestBeatSec = 1.5;
  /* Mark a weight so we can detect keep vs reset. */
  a.agent.net.weights[0][0][0] = 42.5;
  var oldRound = search.round;
  search.finishRound();
  search.drainLiveTournament();
  assert(search.round === oldRound + 1, "round increments");
  assert(search.mmLevel === 1, "MM0 advances to MM1");
  assert(search.population.length >= 3, "next pop has survivors+mutants");
  assert(search.population.every(function (p) {
    return p.beaten === false && p.secondsUsed === 0 && p.warmupLeftMs === LAB.WARMUP_MS;
  }), "progress reset for new round");
  var survivorA = search.population.filter(function (p) { return p.id === a.id; })[0];
  assert(!!survivorA, "tournament survivor a kept by id");
  assert(survivorA.agent.net.weights[0][0][0] === 42.5, "survivor keeps trained weights");
  assert(search.tournamentSnapshots.length >= 1, "stored tournament snaps");
})();

(function testAdvanceDedupsNearDuplicateSurvivors() {
  var search = new LAB.ShapeSearch();
  var older = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: [77, 64, 77, 1]
  }, "bot_10");
  var newer = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: [77, 67, 75, 1]
  }, "bot_99");
  var filler = LAB.createIndividual({
    family: "nn_value",
    layerSizes: [77, 96, 116, 1]
  });
  search.population = [newer, older, filler];
  search.population.forEach(function (p, idx) {
    p.beaten = true;
    p.secondsToBeat = idx + 1;
  });
  search.bestBeatSec = 1;
  search.finishRound();
  search.drainLiveTournament();
  var sameFamily = search.population.filter(function (p) {
    return p.family === "reinforce_value";
  });
  var survivorIds = sameFamily.filter(function (p) {
    return p.id === "bot_10" || p.id === "bot_99";
  });
  assert(survivorIds.length === 1, "near-dup survivors collapsed to one seat");
  assert(survivorIds[0].id === "bot_10", "kept oldest survivor id");
  /* Morph: no intentional same-size clones; only the survivor at default width. */
  var defaults = sameFamily.filter(function (p) {
    return p.layerSizes && p.layerSizes.join(",") === "77,64,77,1";
  });
  assert(defaults.length === 1, "morph has one default-width seat (mutants differ)");
})();

(function testCloneMutantKeepsOverlapWeights() {
  var parent = LAB.createIndividual({
    family: "nn_value",
    layerSizes: [77, 64, 77, 1],
    learningRate: 0.03
  });
  parent.agent.net.weights[0][0][0] = 7.25;
  var wider = LAB.mutateWidth(parent.layerSizes, 1.5);
  var childAgent = LAB.cloneAgentFromParent(parent, parent.family, wider, 0.03);
  assert(childAgent.net.weights[0][0][0] === 7.25, "width mutant inherits overlapping weights");
})();

/* --- Round robin --- */
(function testRoundRobin() {
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  var baselines = LAB.makeBaselineEntries(1, 3);
  var result = LAB.runRoundRobin([
    { id: a.id, family: a.family, label: a.label, agent: a.agent },
    { id: b.id, family: b.family, label: b.label, agent: b.agent }
  ].concat(baselines), 3);
  assert(result.ranked.length === 4, "rr ranked length");
  assert(result.durationMs > 0, "rr duration");
  var pts = 0;
  result.ranked.forEach(function (r) { pts += r.points; });
  assertClose(pts, 12, 0.01, "4 players * 3 games * 1pt each pairing pair=2 games * 6 pairings? wait");
  /* 4 players => C(4,2)=6 pairings * 2 games = 12 games * 1 point distributed = 12 */
  assertClose(pts, 12, 0.01, "total points 12");
})();

/* --- Mid-tournament sort by win% --- */
(function testMidTournWinPctSort() {
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  var session = LAB.createRoundRobinSession([
    { id: a.id, family: a.family, label: a.label, agent: a.agent },
    { id: b.id, family: b.family, label: b.label, agent: b.agent }
  ].concat(LAB.makeBaselineEntries(1, 1)), 2);
  /* Fabricate uneven mid-run record: high WR / low points beats low WR / high points. */
  session.wins[a.id] = 2;
  session.draws[a.id] = 0;
  session.losses[a.id] = 0;
  session.points[a.id] = 2;
  session.wins[b.id] = 1;
  session.draws[b.id] = 2;
  session.losses[b.id] = 1;
  session.points[b.id] = 4;
  session.gamesDone = 6;
  session.done = false;
  var mid = LAB.roundRobinStandings(session);
  assert(mid.inProgress, "mid inProgress");
  assert(mid.ranked[0].id === a.id, "mid sort by WR first");
  assertClose(mid.ranked[0].winPct, 1, 1e-9, "mid leader WR 100%");
  session.done = true;
  var fin = LAB.roundRobinStandings(session);
  assert(!fin.inProgress, "final not inProgress");
  assert(fin.ranked[0].id === b.id, "final sort by points");
})();

/* --- Tournament wins vs Minimax suffix --- */
(function testWinsVsMinimax() {
  var bot = { id: "bot", family: "nn_value", label: "Bot" };
  var mm = { id: "minimax", family: "minimax", label: "Minimax" };
  var session = LAB.createRoundRobinSession([bot, mm], 1);
  LAB.applyMatchResult(session, bot, mm, "X", "X");
  LAB.applyMatchResult(session, bot, mm, "O", "X");
  assert(session.winsVsMinimax[bot.id] === 1, "one win vs MM (home)");
  assert(session.winsVsMinimax[mm.id] === 0, "MM counter unused");
  var st = LAB.roundRobinStandings(session);
  var botRow = st.ranked.filter(function (r) { return r.id === "bot"; })[0];
  var mmRow = st.ranked.filter(function (r) { return r.id === "minimax"; })[0];
  assert(botRow.winsVsMinimax === 1, "standings carry winsVsMinimax");
  assert(LAB.formatBeatMinimaxSuffix(botRow) === " | beat Minimax in 1 game", "1-game suffix");
  botRow.winsVsMinimax = 2;
  assert(LAB.formatBeatMinimaxSuffix(botRow) === " | beat Minimax in 2 games", "2-game suffix");
  botRow.winsVsMinimax = 0;
  assert(LAB.formatBeatMinimaxSuffix(botRow) === "", "no suffix at 0");
  assert(LAB.formatBeatMinimaxSuffix(mmRow) === "", "no suffix on Minimax");
})();

/* --- Tournament history: top-10 learned + baselines only if overall top-5 --- */
(function testHistoryTopLearned() {
  var ranked = [
    { id: "minimax", family: "minimax", label: "Minimax", points: 9, wins: 9, draws: 0, losses: 0 },
    { id: "a", family: "nn_value", label: "A", points: 8, wins: 7, draws: 2, losses: 1 },
    { id: "mcts", family: "mcts", label: "MCTS", points: 7, wins: 6, draws: 2, losses: 2 },
    { id: "b", family: "reinforce_value", label: "B", points: 6, wins: 5, draws: 2, losses: 3 }
  ];
  var top = LAB.topLearnedRanks(ranked, 10);
  assert(top.length === 2, "baselines excluded from survivors");
  assert(top[0].id === "a" && top[1].id === "b", "learned order kept");

  /* MCTS place 3 -> in top 5, so history includes both baselines. */
  var histBoth = LAB.historyTopRanks(ranked, 10);
  assert(histBoth.length === 4, "history includes top-5 baselines");
  assert(histBoth[0].id === "minimax" && histBoth[2].id === "mcts", "baseline order");
  assert(histBoth[1].points === 8 && histBoth[1].wins === 7, "history keeps pts/WDL");

  /* Minimax #1, MCTS #10 -> only Minimax extra (5 learned + MM = 6). */
  var ranked2 = [
    { id: "minimax", family: "minimax", label: "Minimax", points: 10, wins: 10, draws: 0, losses: 0 }
  ];
  var i;
  for (i = 0; i < 8; i += 1) {
    ranked2.push({
      id: "bot" + i,
      family: "tfjs_value",
      label: "B" + i,
      points: 9 - i,
      wins: 9 - i,
      draws: 0,
      losses: i
    });
  }
  ranked2.push({ id: "mcts", family: "mcts", label: "MCTS", points: 1, wins: 1, draws: 0, losses: 9 });
  assert(ranked2[0].id === "minimax" && ranked2[9].id === "mcts", "fixture places 1 and 10");
  var histMmOnly = LAB.historyTopRanks(ranked2, 5);
  assert(histMmOnly.length === 6, "top 5 learned + Minimax");
  assert(histMmOnly[0].id === "minimax", "Minimax kept when #1");
  assert(histMmOnly.every(function (r) { return r.id !== "mcts"; }), "MCTS #10 omitted");
  assert(LAB.topLearnedRanks(ranked2, 5).every(function (r) {
    return r.id !== "minimax" && r.id !== "mcts";
  }), "survivors still exclude baselines");
})();

/* --- ShapeSearch mechanics --- */
(function testSearchPauseAndBudget() {
  var search = new LAB.ShapeSearch();
  assert(search.population.length >= 10, "search pop");
  search.setPaused(true, true);
  assert(search.paused && search.userPaused, "user pause");
  search.setPaused(false, false);
  assert(search.paused === false, "setPaused false");
  /* userPaused still true until resume from user */
  search.setPaused(false, true);
  assert(!search.userPaused, "user unpause");

  search.population[0].beaten = true;
  search.population[0].secondsToBeat = 2;
  search.bestBeatSec = 2;
  assert(LAB.MIN_TRAIN_SEC === 10, "MM min train floor is 10s");
  assert(LAB.MIN_TRAIN_SEC_RANDOM === 3, "R1 Random min train floor is 3s");
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC_RANDOM, "R1 maxTrainSec is 3s");
  search.mmLevel = 1;
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC, "MM1 maxTrainSec is 10s");
  search.mmLevel = 0;
  var needing = search.botsNeedingTrain();
  needing.forEach(function (ind) {
    assert(ind.secondsUsed < search.minTrainSec(), "under min train initially");
  });
  /* Past min with a beater: no longer needs train */
  var victim = needing[0];
  victim.secondsUsed = 100;
  var needing2 = search.botsNeedingTrain();
  assert(needing2.every(function (x) { return x.id !== victim.id; }), "past-min excluded when beater exists");
})();

(function testFilledBotStillEvalsWhenTrainSkipped() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  search.population = [a, b];
  a.beaten = true;
  a.secondsToBeat = 5;
  a.secondsUsed = 15;
  a.warmupLeftMs = 0;
  a.trainSinceEvalMs = 50;
  a.trainBetweenTargetMs = 50;
  a.lastEvalWinRate = 0.6;
  b.secondsUsed = 0;
  b.warmupLeftMs = 0;
  b.trainSinceEvalMs = 0;
  b.trainBetweenTargetMs = 1e9;
  search.bestBeatSec = 5;
  var evalAgents = 0;
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    evalAgents += 1;
    return { wins: 51, draws: 0, losses: 49, games: 100, winRate: 0.51, extended: false };
  };
  try {
    search.tick(80);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  assert(evalAgents >= 1, "filled bot still evals");
  assert(a.lastEvalWins === 51, "skip-train pass wrote mmWR");
  assert(a.trainSinceEvalMs === 0, "eval reset train-since-eval");
})();

(function testPeerTrainEvalsPeerIfDue() {
  var search = new LAB.ShapeSearch();
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [a, b];
  search.mmLevel = 1;
  a.lastEvalWinRate = 0.8;
  b.lastEvalWinRate = 0.8;
  a.warmupLeftMs = 0;
  b.warmupLeftMs = 0;
  b.secondsUsed = 10;
  b.trainSinceEvalMs = 50;
  b.trainBetweenTargetMs = 50;
  a.trainSinceEvalMs = 0;
  a.trainBetweenTargetMs = 1e9;
  var orig = APP.shapeEvalVsMinimaxTimed;
  var origRand = Math.random;
  var origPeer = APP.shapeTrainVsPeer;
  var evalCount = 0;
  APP.shapeEvalVsMinimaxTimed = function () {
    evalCount += 1;
    return { wins: 60, draws: 0, losses: 40, games: 100, winRate: 0.6, extended: false };
  };
  APP.shapeTrainVsPeer = function () { return null; };
  Math.random = function () { return 0.10; };
  try {
    search.trainIndividual(a, false);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
    APP.shapeTrainVsPeer = origPeer;
    Math.random = origRand;
  }
  assert(evalCount >= 1, "peer due for eval is evaluated");
  assert(b.lastEvalWins === 60, "peer mmWR updated");
})();

(function testPeerTrainCreditsBothClocks() {
  var search = new LAB.ShapeSearch();
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [a, b];
  search.mmLevel = 1;
  a.lastEvalWinRate = 0.8;
  b.lastEvalWinRate = 0.8;
  a.warmupLeftMs = 0;
  b.warmupLeftMs = 0;
  a.secondsUsed = 5;
  b.secondsUsed = 20;
  a.trainSinceEvalMs = 0;
  a.trainBetweenTargetMs = 1e9;
  b.trainSinceEvalMs = 0;
  b.trainBetweenTargetMs = 1e9;
  var origPeer = APP.shapeTrainVsPeer;
  var origRand = Math.random;
  var origNow = performance.now;
  var fakeNow = 1000;
  performance.now = function () {
    fakeNow += 40;
    return fakeNow;
  };
  APP.shapeTrainVsPeer = function () { return null; };
  Math.random = function () { return 0.10; };
  try {
    search.trainIndividual(a, false);
    assert(a.secondsUsed > 5, "initiator train clock advances");
    assert(b.secondsUsed > 20, "peer train clock also advances");
    assert(b.trainSinceEvalMs > 0, "peer gets eval cadence credit");
  } finally {
    APP.shapeTrainVsPeer = origPeer;
    Math.random = origRand;
    performance.now = origNow;
  }
})();

(function testPickPeerPrefersBehindTrainClock() {
  var search = new LAB.ShapeSearch();
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var ahead = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  var behind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [a, ahead, behind];
  a.lastEvalWinRate = 0.8;
  ahead.lastEvalWinRate = 0.8;
  behind.lastEvalWinRate = 0.8;
  ahead.secondsUsed = 30;
  behind.secondsUsed = 4;
  var peer = search.pickPeerFor(a);
  assert(peer && peer.id === behind.id, "equal RR prefers lower train_s peer");
})();

(function testSearchTickProgress() {
  var search = new LAB.ShapeSearch();
  /* Keep only a tiny value net for speed */
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    })
  ];
  var before = search.population[0].secondsUsed;
  search.tick(80);
  assert(search.population[0].secondsUsed >= before, "tick accumulates time");
})();

(function testSweepTopToBottomHalfSeconds() {
  assert(Math.abs(LAB.nextTrainSliceTarget(0, null) - 0.5) < 1e-9, "0 -> 0.5");
  assert(Math.abs(LAB.nextTrainSliceTarget(0.5, null) - 1.0) < 1e-9, "0.5 -> 1.0");

  var search = new LAB.ShapeSearch();
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    }),
    LAB.createIndividual({
      family: "nn_value",
      layerSizes: APP.defaultLayerSizes("nn_value")
    }),
    LAB.createIndividual({
      family: "qtable",
      layerSizes: null
    })
  ];
  search.population.forEach(function (p) {
    p.secondsUsed = 0;
    p.warmupLeftMs = 0;
    p.trainSinceEvalMs = 0;
    p.trainBetweenTargetMs = 1e9;
  });

  /* Synced at 0: push first bot toward 0.5. */
  search.tick(80);
  assert(search._currentId === search.population[0].id, "leader trains first when synced");
  assert(search._sliceTarget === 0.5, "leader target is next 0.5s");
  assert(search.population[1].secondsUsed === 0, "others wait");

  /* First bot ahead at 3.2: catch others up to 3.2 (not a fresh 0.5 wave). */
  search.population[0].secondsUsed = 3.2;
  search.population[1].secondsUsed = 1.0;
  search.population[2].secondsUsed = 0.5;
  search._dwellId = null;
  search._sweepIndex = -1;
  search.tick(80);
  assert(search._sliceTarget === 3.2, "catch-up target is first bot's train_s");
  assert(search._currentId === search.population[1].id ||
    search._currentId === search.population[2].id, "trains a bot behind the leader");
  assert(search._currentId !== search.population[0].id, "leader waits during catch-up");

  /* All caught up to leader: push leader all the way to next half-second (sticky). */
  search.population[0].secondsUsed = 3.2;
  search.population[1].secondsUsed = 3.2;
  search.population[2].secondsUsed = 3.2;
  search._dwellId = null;
  search._leaderAdvanceTarget = null;
  search._sweepIndex = -1;
  search.tick(40);
  assert(search._currentId === search.population[0].id, "synced -> leader advances");
  assert(Math.abs(search._leaderAdvanceTarget - 3.5) < 1e-9, "sticky advance target 3.5");
  assert(Math.abs(search._sliceTarget - 3.5) < 1e-9, "slice target 3.5");
  /* Partial progress must not retarget to 3.3 catch-up. */
  search.population[0].secondsUsed = 3.3;
  search.tick(40);
  assert(search._currentId === search.population[0].id, "keeps advancing leader");
  assert(Math.abs(search._leaderAdvanceTarget - 3.5) < 1e-9, "still sticky to 3.5");
  assert(search.population[1].secondsUsed === 3.2, "others wait until leader hits 3.5");
})();

(function testUnionPrefersLiveOverSnap() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  lab.search.population = [
    LAB.createIndividual({
      family: "nn_value",
      layerSizes: [77, 64, 77, 1]
    }, "bot_live")
  ];
  lab.search.tournamentSnapshots = [{
    id: "bot_live",
    family: "nn_value",
    layerSizes: [77, 64, 77, 1],
    label: "dup",
    blob: APP.shapeAgentToDict(lab.search.population[0].agent, "nn_value")
  }, {
    id: "bot_snap_only",
    family: "reinforce_value",
    layerSizes: [77, 64, 77, 1],
    label: "snap",
    blob: APP.shapeAgentToDict(
      APP.createShapeAgent("reinforce_value", [77, 64, 77, 1]),
      "reinforce_value"
    )
  }];
  var cat = lab.unionCatalog();
  var ids = cat.map(function (c) { return c.id; });
  assert(ids.indexOf("bot_live") >= 0, "live present");
  assert(ids.indexOf("bot_snap_only") >= 0, "snap-only present");
  assert(ids.filter(function (id) { return id === "bot_live"; }).length === 1, "no dup live");
})();

(function testTabAutoPause() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  lab.search.setPaused(false, true);
  lab.setActiveTab("train");
  assert(lab.search.paused === true, "search auto-paused on leave");
  assert(lab.search.userPaused === false, "auto-pause is not user pause");
  lab.setActiveTab("search");
  assert(lab.search.paused === false, "search resumes when not user-paused");
  lab.search.setPaused(true, true);
  lab.setActiveTab("observe");
  lab.setActiveTab("search");
  assert(lab.search.paused === true, "user pause sticks across tabs");
})();

/* --- Train One --- */
(function testTrainOneLadderSchedule() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  t1.startFromSnapshotOrLive(source);
  assert(t1.trainee, "has trainee");
  assert(t1.mmLevel === 0, "starts MM0");
  t1.T_tourn = 2;
  t1.trainSinceTourn = 11;
  assert(t1.trainSinceTourn >= 5 * t1.T_tourn, "5x tourn trigger ready");
  t1.resetTrainee();
  assert(t1.mmLevel === 0 && t1.ladder.length === 0, "reset clears ladder");
})();

(function testTrainOneRestartLadderKeepsWeights() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  t1.startFromSnapshotOrLive(source);
  t1.trainee.agent.net.weights[0][0][0] = 7.25;
  LAB.saveBestWeights(t1.trainee, 0.72);
  t1.trainee.lastEvalWinRate = 0.72;
  t1.mmLevel = 3;
  t1.ladder = [{ mmLevel: 0, seconds: 1, winRate: 0.6 }];
  t1.totalTrainSec = 40;
  t1.T_tourn = 2;
  t1.peerBeatIds = ["other_1"];
  t1.restartLadder();
  assert(t1.mmLevel === 0, "ladder back to Random");
  assert(t1.ladder.length === 0, "ladder entries cleared");
  assert(t1.totalTrainSec === 0, "train clock cleared");
  assert(t1.T_tourn === null, "tourn schedule cleared");
  assert(t1.peerBeatIds.length === 0, "peer beaters cleared");
  assert(Math.abs(t1.trainee.agent.net.weights[0][0][0] - 7.25) < 1e-9, "weights kept");
  assert(t1.trainee.bestWinRate === 0.72, "keep-best kept");
  assert(!!t1.trainee.bestBlob, "bestBlob kept");
  assert(t1.trainee.lastEvalWinRate === 0.72, "last eval kept");
})();

(function testTrainOneTick() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  t1.startFromSnapshotOrLive({
    id: source.id,
    family: source.family,
    layerSizes: source.layerSizes,
    label: source.label,
    agent: source.agent
  });
  var before = t1.totalTrainSec;
  t1.tick(60, []);
  assert(t1.totalTrainSec > before, "train one accumulates time");
})();

(function testPickPlayMove() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var id = lab.search.population[0].id;
  var board = G.emptyBoard();
  var move = lab.pickPlayMove(id, board, C.X, 0.05);
  assert(typeof move === "number", "pickPlayMove returns col");
  assert(G.legalMoves(board).indexOf(move) >= 0, "legal move");
})();

(function testPlayCatalogIncludesBaselines() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var cat = lab.getPlayCatalog();
  assert(cat.length >= 4, "catalog has baselines + learned");
  assert(cat[0].id === "random" && cat[0].name === "Random", "Random first");
  assert(cat[1].id === "minimax" && cat[1].name === "Minimax", "Minimax second");
  assert(cat[2].id === "mcts" && cat[2].name === "MCTS", "MCTS third");
  var trainIds = lab.unionCatalog().map(function (c) { return c.id; });
  assert(trainIds.indexOf("random") < 0, "Train One catalog excludes Random");
  assert(trainIds.indexOf("minimax") < 0, "Train One catalog excludes Minimax");
  assert(trainIds.indexOf("mcts") < 0, "Train One catalog excludes MCTS");
  var board = G.emptyBoard();
  var rnd = lab.pickPlayMove("random", board, C.X, 0);
  var mm = lab.pickPlayMove("minimax", board, C.X, 0.02);
  var mcts = lab.pickPlayMove("mcts", board, C.X, 0.02);
  assert(G.legalMoves(board).indexOf(rnd) >= 0, "random play legal");
  assert(G.legalMoves(board).indexOf(mm) >= 0, "minimax play legal");
  assert(G.legalMoves(board).indexOf(mcts) >= 0, "mcts play legal");
})();

(function testPlayCatalogIncludesTrainee() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var src = lab.search.population[0];
  lab.trainOne.startFromSnapshotOrLive(src);
  var cat = lab.getPlayCatalog();
  var ids = cat.map(function (c) { return c.id; });
  assert(ids.indexOf(lab.trainOne.trainee.id) === 3, "trainee after baselines");
  assert(cat[3].name.indexOf("(trainee)") >= 0, "trainee labeled");
  var board = G.emptyBoard();
  var mv = lab.pickPlayMove(lab.trainOne.trainee.id, board, C.X, 0.05);
  assert(G.legalMoves(board).indexOf(mv) >= 0, "trainee play legal");
})();

(function testSaveLocal() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  lab.search.round = 2;
  lab.search.mmLevel = 2;
  lab.search.population[0].secondsUsed = 3.5;
  lab.search.population[0].beaten = true;
  lab.search.population[0].secondsToBeat = 3.5;
  var st = lab.saveLocal();
  assert(st.ok, "save ok");
  var raw = global.localStorage.getItem(LAB.STORAGE_KEY);
  assert(!!raw, "storage written");
  var parsed = JSON.parse(raw);
  assert(parsed.format === LAB.CHECKPOINT_FORMAT, "checkpoint format");
  assert(parsed.search.round === 2, "saved round");
  assert(parsed.search.population[0].blob, "weights blob saved");
  assert(parsed.search.population[0].secondsUsed === 3.5, "progress saved");
})();

(function testSaveLoadRoundtrip() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  lab.search.round = 3;
  lab.search.mmLevel = 3;
  lab.search.population[1].secondsUsed = 12;
  lab.search.population[1].beaten = true;
  lab.search.population[1].secondsToBeat = 11.5;
  lab.trainOne.startFromSnapshotOrLive(lab.search.population[1]);
  lab.trainOne.mmLevel = 2;
  lab.trainOne.totalTrainSec = 9.25;
  lab.trainOne.ladder = [{ mmLevel: 1, seconds: 4, winRate: 0.6 }];
  lab.saveLocal();

  var lab2 = new LAB.ShapeLab({ onSnapshot: null });
  lab2.stop();
  var loaded = null;
  lab2.loadLocal(function (err, data) {
    assert(!err, "load no err");
    loaded = data;
  });
  assert(!!loaded, "loaded sync via localStorage");
  assert(lab2.search.round === 3, "restored round");
  assert(lab2.search.mmLevel === 3, "restored mm");
  assert(lab2.search.population[1].beaten, "restored beater");
  assert(Math.abs(lab2.search.population[1].secondsUsed - 12) < 1e-9, "restored used");
  assert(!!lab2.trainOne.trainee, "restored trainee");
  assert(lab2.trainOne.mmLevel === 2, "trainee mm");
  assert(Math.abs(lab2.trainOne.totalTrainSec - 9.25) < 1e-9, "trainee train sec");
  assert(lab2.trainOne.ladder.length === 1, "trainee ladder");
})();

(function testExportTrainee() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var empty = lab.exportTrainee();
  assert(!empty.ok, "export fails without trainee");
  lab.trainOne.startFromSnapshotOrLive(lab.search.population[0]);
  var exp = lab.exportTrainee();
  assert(exp.ok, "export ok");
  assert(exp.filename.indexOf(".json") > 0, "json filename");
  assert(exp.payload.format === LAB.EXPORT_FORMAT, "export format");
  assert(exp.payload.blob && exp.payload.family, "export has weights");
  var roundTrip = APP.shapeAgentFromDict(exp.payload.blob);
  assert(!!roundTrip, "exported blob reconstitutes");
})();

(function testImportTraineeNamesAndAddsToSearch() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var src = lab.search.population[0];
  lab.trainOne.startFromSnapshotOrLive(src);
  lab.trainOne.mmLevel = 2;
  lab.trainOne.totalTrainSec = 3.5;
  lab.trainOne.ladder = [{ mmLevel: 1, seconds: 1.2, winRate: 0.55 }];
  var fresh = APP.createShapeAgent(src.family, src.layerSizes, src.learningRate);
  lab.trainOne.trainee.agent = fresh;
  var exp = lab.exportTrainee();
  assert(exp.ok, "export for import");

  var other = new LAB.ShapeLab({ onSnapshot: null });
  other.stop();
  var before = other.search.population.length;
  var bad = other.importTrainee({ format: "nope" }, "Nope");
  assert(!bad.ok, "import rejects other formats");
  var imported = other.importTrainee(exp.payload, "  Pocket Ace  ");
  assert(imported.ok, "import ok");
  assert(imported.label === "Pocket Ace", "import trims the name");
  assert(other.trainOne.trainee.imported, "imported flag");
  assert(other.trainOne.trainee.sourceId === null, "import has no search source");
  assert(other.trainOne.userPaused, "import starts paused");
  assert(other.trainOne.mmLevel === 2, "import keeps ladder level");
  assert(other.trainOne.ladder.length === 1, "import keeps ladder");
  assert(Math.abs(other.trainOne.totalTrainSec - 3.5) < 1e-9, "import keeps train time");
  var trainBlob = JSON.stringify(APP.shapeAgentToDict(fresh, src.family));
  assert(
    JSON.stringify(APP.shapeAgentToDict(other.trainOne.trainee.agent, src.family)) === trainBlob,
    "import keeps weights"
  );

  var added = other.promoteTraineeToSearch();
  assert(added.ok && added.added, "save adds imported bot");
  assert(other.search.population.length === before + 1, "population grew by one");
  var seat = other.search.population[other.search.population.length - 1];
  assert(seat.label === "Pocket Ace", "search bot uses the import name");
  assert(
    JSON.stringify(APP.shapeAgentToDict(seat.agent, seat.family)) === trainBlob,
    "search bot has imported weights"
  );
  assert(!other.trainOne.trainee.imported, "later saves overwrite the new bot");
  assert(other.trainOne.trainee.sourceId === seat.id, "linked to the new bot");

  var again = APP.createShapeAgent(src.family, src.layerSizes, src.learningRate);
  other.trainOne.trainee.agent = again;
  var againBlob = JSON.stringify(APP.shapeAgentToDict(again, src.family));
  var second = other.promoteTraineeToSearch();
  assert(second.ok && !second.added, "second save overwrites");
  assert(other.search.population.length === before + 1, "second save does not add another bot");
  assert(
    JSON.stringify(APP.shapeAgentToDict(seat.agent, seat.family)) === againBlob,
    "overwrite replaced weights"
  );

  var neat = LAB.createIndividual({ family: "neat", layerSizes: null });
  var neatLab = new LAB.ShapeLab({ onSnapshot: null });
  neatLab.stop();
  neatLab.trainOne.startFromSnapshotOrLive(neat);
  var neatExp = neatLab.exportTrainee();
  var named = other.importTrainee(neatExp.payload, "Kitchen NEAT");
  assert(named.ok, "neat import ok");
  var neatAdd = other.promoteTraineeToSearch();
  assert(neatAdd.ok && neatAdd.added, "neat import added");
  var neatSeat = other.search.population[other.search.population.length - 1];
  assert(neatSeat.label === "Kitchen NEAT", "custom name beats NEAT serial label");
})();

(function testPromoteTraineeToSearch() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  assert(!lab.promoteTraineeToSearch().ok, "promote fails without trainee");
  var src = lab.search.population[0];
  src.beaten = true;
  src.beatStreak = 10;
  src.secondsToBeat = 4;
  src.lastEvalWinRate = 0.7;
  lab.trainOne.startFromSnapshotOrLive(src);
  var fresh = APP.createShapeAgent(src.family, src.layerSizes, src.learningRate);
  lab.trainOne.trainee.agent = fresh;
  var trainBlob = JSON.stringify(APP.shapeAgentToDict(fresh, src.family));
  var res = lab.promoteTraineeToSearch();
  assert(res.ok, "promote ok");
  assert(res.updatedLive, "updated live population");
  var after = JSON.stringify(
    APP.shapeAgentToDict(lab.search.population[0].agent, src.family)
  );
  assert(after === trainBlob, "Search bot has trainee weights");
  assert(!lab.search.population[0].beaten, "cleared beaten after promote");
  assert(lab.search.population[0].beatStreak === 0, "cleared streak after promote");
  assert(lab.search.population[0].lastEvalWinRate === null, "cleared last eval");
})();

(function testPromoteTraineeUpdatesSnapshot() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  var src = lab.search.population[0];
  var snapBlob = APP.shapeAgentToDict(src.agent, src.family);
  lab.search.tournamentSnapshots = [{
    id: "snap_only_1",
    family: src.family,
    layerSizes: src.layerSizes,
    learningRate: src.learningRate,
    label: src.label,
    blob: snapBlob
  }];
  lab.trainOne.startFromSnapshotOrLive({
    id: "snap_only_1",
    family: src.family,
    layerSizes: src.layerSizes,
    learningRate: src.learningRate,
    label: src.label,
    blob: snapBlob
  });
  var fresh = APP.createShapeAgent(src.family, src.layerSizes, src.learningRate);
  lab.trainOne.trainee.agent = fresh;
  var trainBlob = JSON.stringify(APP.shapeAgentToDict(fresh, src.family));
  var res = lab.promoteTraineeToSearch();
  assert(res.ok, "promote snap ok");
  assert(!res.updatedLive, "no live match");
  assert(res.updatedSnap, "updated tournament snap");
  assert(
    JSON.stringify(lab.search.tournamentSnapshots[0].blob) === trainBlob,
    "snap blob replaced"
  );
})();

/* --- playAlgo catalog path doesn't crash --- */
(function testTrainOneTournOnLevelPassForcesRefresh() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  t1.startFromSnapshotOrLive(source);
  /* Contender tournament includes trainee + baselines for current level. */
  t1.ladder.push({ mmLevel: 0, seconds: 1.2, winRate: 0.6 });
  t1.mmLevel = 2;
  t1.runContenderTournament([]);
  assert(t1.T_tourn !== null && t1.T_tourn > 0, "T_tourn set after contender tourn");
  assert(t1.lastTournament && t1.lastTournament.ranked.length >= 3, "trainee+mm+mcts ranked");
  var traineeRow = t1.lastTournament.ranked.filter(function (r) {
    return r.id === t1.trainee.id;
  });
  assert(traineeRow.length === 1, "trainee in tournament");
})();

(function testPeersWhoBeatTraineeByPoints() {
  var tourn = {
    ranked: [
      { id: "strong", family: "nn_value", points: 8 },
      { id: "tied", family: "nn_value", points: 5 },
      { id: "trainee", family: "reinforce_value", points: 5 },
      { id: "weak", family: "nn_value", points: 2 },
      { id: "minimax", family: "minimax", points: 9 }
    ]
  };
  var beaters = LAB.peersWhoBeatTrainee(tourn, "trainee");
  assert(beaters.indexOf("strong") >= 0, "higher points is peer beater");
  assert(beaters.indexOf("tied") < 0, "equal points not a beater");
  assert(beaters.indexOf("weak") < 0, "lower points not a beater");
  assert(beaters.indexOf("minimax") < 0, "baseline excluded from peer beaters");
})();

(function testTrainOnePeerPickOnlyPriorBeaters() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  t1.startFromSnapshotOrLive(source);
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var contenders = [
    { id: a.id, family: a.family, agent: a.agent, label: a.label },
    { id: b.id, family: b.family, agent: b.agent, label: b.label }
  ];
  t1.trainee.lastEvalWinRate = 0.8;
  assert(t1.pickPeerWhoBeatTrainee(contenders) === null, "no peer before tourn beaters");
  t1.peerBeatIds = [b.id];
  var peer = t1.pickPeerWhoBeatTrainee(contenders);
  assert(peer && peer.id === b.id, "only higher-point prior peer chosen");
})();

(function testTrainOneAdvancesOnlyWhenTournPointsBeatBaseline() {
  var t1 = new LAB.TrainOne();
  var source = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  t1.startFromSnapshotOrLive(source);
  assert(t1.mmLevel === 0, "starts Random");
  assert(t1.baselineIdForLevel() === "random", "MM0 baseline is Random");
  var fakeLow = {
    ranked: [
      { id: t1.trainee.id, points: 2 },
      { id: "random", family: "random", points: 4 },
      { id: "mcts", family: "mcts", points: 3 }
    ]
  };
  assert(!t1.traineeBeatsBaseline(fakeLow), "fewer points does not beat");
  t1.lastTournament = fakeLow;
  assert(t1.mmLevel === 0, "level unchanged without advance");

  var fakeWin = {
    ranked: [
      { id: t1.trainee.id, points: 5 },
      { id: "random", family: "random", points: 4 },
      { id: "mcts", family: "mcts", points: 3 }
    ]
  };
  assert(t1.traineeBeatsBaseline(fakeWin), "more points beats baseline");
  t1.trainee.lastEvalWinRate = 0.7;
  t1.advanceLevel(fakeWin);
  assert(t1.mmLevel === 1, "advanced to MM1");
  assert(t1.ladder.length === 1 && t1.ladder[0].mmLevel === 0, "ladder records Random");
  assert(t1.baselineIdForLevel() === "minimax", "MM1 baseline is Minimax");

  var tie = {
    ranked: [
      { id: t1.trainee.id, points: 4 },
      { id: "minimax", family: "minimax", points: 4 }
    ]
  };
  assert(!t1.traineeBeatsBaseline(tie), "tie does not advance");

  /* Eval streak alone must not advance Train One levels. */
  var levelBefore = t1.mmLevel;
  t1.trainee.beatStreak = LAB.BEAT_STREAK_NEED;
  t1.trainee.beaten = true;
  assert(t1.mmLevel === levelBefore, "streak does not change mmLevel");
})();

(function testManualTournamentIncludesBaselines() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  search.population = search.population.slice(0, 2);
  search.manualTournament();
  search.drainLiveTournament();
  var t = search.lastTournament;
  var ids = t.ranked.map(function (r) { return r.id; });
  assert(ids.indexOf("minimax") >= 0, "manual tourn has minimax");
  assert(ids.indexOf("mcts") >= 0, "manual tourn has mcts");
  assert(!t.inProgress, "manual tourn finished");
})();

(function testPausedFillCompleteStartsTournament() {
  var lab = new LAB.ShapeLab({ onSnapshot: null });
  lab.stop();
  lab.search.population = lab.search.population.slice(0, 3);
  lab.search.population.forEach(function (p, i) {
    p.beaten = i === 0;
    p.secondsToBeat = i === 0 ? 2 : null;
    p.secondsUsed = 12;
  });
  lab.search.bestBeatSec = 2;
  lab.search.userPaused = true;
  lab.search.paused = true;
  assert(lab.search.roundComplete(), "fill complete");
  lab.tick();
  assert(!!lab.search.liveTourn, "tournament starts even while Search paused");
  lab.search.drainLiveTournament();
  assert(lab.search.round >= 2, "round advanced after drained tourn");
})();

(function testActiveNotBeatenRequiresBeatersForComplete() {
  var search = new LAB.ShapeSearch();
  search.population = search.population.slice(0, 2);
  assert(search.minTrainSec() === LAB.MIN_TRAIN_SEC_RANDOM, "R1 uses 3s min");
  assert(!search.roundComplete(), "incomplete without beaters");
  search.population[0].beaten = true;
  search.population[0].secondsToBeat = 1;
  search.population[0].secondsUsed = 1;
  search.bestBeatSec = 1;
  search.population[1].secondsUsed = 1000;
  assert(!search.roundComplete(), "incomplete until every bot reaches min train sec");
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC_RANDOM, "max is R1 min floor");
  search.population[0].secondsUsed = LAB.MIN_TRAIN_SEC_RANDOM;
  search.population[1].secondsUsed = LAB.MIN_TRAIN_SEC_RANDOM;
  assert(search.roundComplete(), "complete when all bots at min and a beater exists");
  search.mmLevel = 1;
  assert(search.minTrainSec() === LAB.MIN_TRAIN_SEC, "MM1 uses 10s min");
})();

(function testTrainAtLeastMinNoFloatStall() {
  var search = new LAB.ShapeSearch();
  search.population = search.population.slice(0, 2);
  search.population[0].beaten = true;
  search.population[0].secondsToBeat = 1;
  search.bestBeatSec = 1;
  var minT = search.minTrainSec();
  search.population[0].secondsUsed = minT;
  /* Classic stall: just under min, where eps-based stop used to skip more train. */
  search.population[1].secondsUsed = minT - 2.98e-10;
  assert(!search.roundComplete(), "just-under min is not complete");
  assert(search.botsNeedingTrain().length === 1, "just-under still needs train");
  search.paused = false;
  var before = search.population[1].secondsUsed;
  var guard = 0;
  while (search.population[1].secondsUsed < minT && guard < 50) {
    search.tick(200);
    guard += 1;
  }
  assert(search.population[1].secondsUsed >= minT, "trains to at least minT");
  assert(search.population[1].secondsUsed > before, "actually trained past the float gap");
  assert(search.roundComplete(), "complete after at-least min fill");
})();

(function testKeepsTrainingPastMinUntilBeater() {
  var search = new LAB.ShapeSearch();
  search.population = search.population.slice(0, 2);
  search.population[0].beaten = false;
  search.population[0].secondsUsed = 20;
  search.population[1].beaten = false;
  search.population[1].secondsUsed = 20;
  assert(search.botsNeedingTrain().length === 2, "past min still trains without beater");
  assert(!search.roundComplete(), "no beater means incomplete");
  search.population[0].beaten = true;
  search.population[0].secondsToBeat = 12;
  search.bestBeatSec = 12;
  assert(search.botsNeedingTrain().length === 0, "with beater and past min, train done");
  assert(search.roundComplete(), "round complete on streak + min train");
})();

(function testFillTrainsToMinWhenBeaterExists() {
  var search = new LAB.ShapeSearch();
  search.population = [
    LAB.createIndividual({
      family: "reinforce_value",
      layerSizes: APP.defaultLayerSizes("reinforce_value")
    }),
    LAB.createIndividual({
      family: "nn_value",
      layerSizes: APP.defaultLayerSizes("nn_value")
    })
  ];
  search.population[0].beaten = true;
  search.population[0].secondsToBeat = 1;
  search.population[0].secondsUsed = 1;
  search.bestBeatSec = 1;
  search.population[1].secondsUsed = 1;
  var needing = search.botsNeedingTrain();
  assert(needing.length === 2, "both need min train");
  assert(needing.some(function (b) { return b.id === search.population[0].id; }), "includes beater");
  var before = search.population[0].secondsUsed;
  search.tick(80);
  assert(search.population[0].secondsUsed > before || search.population[1].secondsUsed > 1,
    "min-train phase advances a lagging bot");
  assert(search.population[0].secondsToBeat === 1, "toBeat stays frozen while filling");
})();

(function testDipClearsBeatAndMaxTrain() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  var a = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  var b = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [a, b];
  a.beaten = true;
  a.secondsToBeat = 6;
  a.secondsUsed = 12;
  a.warmupLeftMs = 0;
  a.trainSinceEvalMs = 50;
  a.trainBetweenTargetMs = 50;
  a.lastEvalWins = 15;
  b.beaten = true;
  b.secondsToBeat = 8;
  b.secondsUsed = 12;
  search.bestBeatSec = 6;
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC, "max is min floor");
  assert(search.roundComplete(), "two beaters past min is complete");

  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    return { winRate: 0.45, wins: 45, draws: 0, losses: 55, games: 100, extended: false };
  };
  try {
    search.tick(100);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  assert(a.lastEvalWins === 45, "mmWR source wins update on dip");
  assert(Math.abs(a.lastEvalWinRate - 0.45) < 1e-9, "mmWR is 45%");
  assert(!a.beaten, "dip clears beaten");
  assert(a.secondsToBeat === 6, "dip keeps toBeat_s");
  assert(search.bestBeatSec === 6, "bestBeatSec keeps lowest toBeat");
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC, "min floor unchanged");
  assert(search.roundComplete(), "remaining beater past min still completes");
})();

(function testDipClearsAllBeatersWaitsAgain() {
  var search = new LAB.ShapeSearch();
  search.mmLevel = 1;
  var a = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  search.population = [a];
  a.beaten = true;
  a.secondsToBeat = 1;
  a.secondsUsed = 12;
  a.warmupLeftMs = 0;
  a.trainSinceEvalMs = 50;
  a.trainBetweenTargetMs = 50;
  search.bestBeatSec = 1;
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC, "max is min floor");
  assert(search.roundComplete(), "complete with beater past min");
  var orig = APP.shapeEvalVsMinimaxTimed;
  APP.shapeEvalVsMinimaxTimed = function () {
    return { winRate: 0.5, wins: 50, draws: 0, losses: 50, games: 100, extended: false };
  };
  try {
    search.tick(80);
  } finally {
    APP.shapeEvalVsMinimaxTimed = orig;
  }
  assert(search.bestBeatSec === 1, "toBeat_s still counted after last beater dips");
  assert(a.secondsToBeat === 1, "solo dip keeps toBeat_s");
  assert(search.maxTrainSec() === LAB.MIN_TRAIN_SEC, "min floor still shown");
  assert(!search.roundComplete(), "must wait for a 10/10 beater again");
  assert(search.botsNeedingTrain().length === 1, "unbeaten needs train again");
})();

(function testPeerPickAmongHighWr() {
  var search = new LAB.ShapeSearch();
  var a = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  var b = LAB.createIndividual({
    family: "reinforce_value",
    layerSizes: APP.defaultLayerSizes("reinforce_value")
  });
  var c = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [a, b, c];
  a.lastEvalWinRate = 0.75;
  b.lastEvalWinRate = 0.72;
  c.lastEvalWinRate = 0.40;
  var peer = search.pickPeerFor(a);
  assert(peer && peer.id === b.id, "peer is other high-WR bot");
  assert(search.pickPeerFor(c) === null, "low WR has no peer play");
})();

(function testMm0IsRandomRound() {
  var search = new LAB.ShapeSearch();
  assert(search.mmLevel === 0, "search starts at MM0");
  assert(LAB.isRandomMmLevel(0), "0 is random level");
  assert(!LAB.isRandomMmLevel(1), "1 is minimax");
  assert(LAB.formatMmTag(0) === "Random", "level 0 labeled Random");
  assert(LAB.formatMmTag(1) === "MM1", "level 1 labeled MM1");
  var origR = APP.shapeEvalVsRandom;
  var origM = APP.shapeEvalVsMinimaxTimed;
  var usedRandom = false;
  var usedMm = false;
  APP.shapeEvalVsRandom = function () {
    usedRandom = true;
    return { wins: 12, draws: 0, losses: 8, games: 20, winRate: 0.6, extended: false };
  };
  APP.shapeEvalVsMinimaxTimed = function () {
    usedMm = true;
    return { wins: 0, draws: 0, losses: 20, games: 20, winRate: 0, extended: false };
  };
  try {
    LAB.runBotEval(search.population[0], 0, 1);
  } finally {
    APP.shapeEvalVsRandom = origR;
    APP.shapeEvalVsMinimaxTimed = origM;
  }
  assert(usedRandom, "MM0 eval vs random");
  assert(!usedMm, "MM0 does not eval vs minimax");
  var bases = search.buildTournamentEntries(search.population.slice(0, 1), 2);
  var ids = bases.map(function (e) { return e.id; });
  assert(ids.indexOf("random") >= 0, "MM0 tournament has Random");
  assert(ids.indexOf("minimax") < 0, "MM0 tournament has no Minimax");
})();

(function testPickShapeTrainKindCurriculum() {
  var unbeaten = { beatsMm: false, hasPeer: false, hasFrozen: true, hasNemesis: true, canSelf: true };
  assert(LAB.pickShapeTrainKind(unbeaten, 0.10) === "nemesis", "unbeaten nemesis slice");
  assert(LAB.pickShapeTrainKind(unbeaten, 0.40) === "mm", "unbeaten MM slice");
  assert(LAB.pickShapeTrainKind(unbeaten, 0.65) === "frozen", "unbeaten frozen slice");
  assert(LAB.pickShapeTrainKind(unbeaten, 0.90) === "random", "unbeaten random slice");
  var beaten = { beatsMm: true, hasPeer: true, hasFrozen: true, hasNemesis: true, canSelf: true };
  assert(LAB.pickShapeTrainKind(beaten, 0.10) === "peer", "beater peer slice");
  assert(LAB.pickShapeTrainKind(beaten, 0.45) === "self", "beater self slice");
  assert(LAB.pickShapeTrainKind(beaten, 0.70) === "frozen", "beater frozen slice");
})();

(function testTrainVsMmRecordsNemesis() {
  var search = new LAB.ShapeSearch();
  var ind = LAB.createIndividual({
    family: "nn_value",
    layerSizes: APP.defaultLayerSizes("nn_value")
  });
  search.population = [ind];
  search.mmLevel = 1;
  ind.lastEvalWinRate = 0.20;
  ind.warmupLeftMs = 0;
  var origMm = APP.shapeTrainVsMinimax;
  var origRand = Math.random;
  APP.shapeTrainVsMinimax = function () {
    return { result: "loss", seed: 4242 };
  };
  Math.random = function () { return 0.40; };
  try {
    search.trainIndividual(ind, false);
  } finally {
    APP.shapeTrainVsMinimax = origMm;
    Math.random = origRand;
  }
  assert(ind.nemesisSeeds.length === 1, "loss vs MM stored as nemesis");
  assert(ind.nemesisSeeds[0].seed === 4242, "nemesis seed kept");
})();

(function testTabularBoxCaps() {
  assert(C.Q_MAX_STATES === 10000, "Q/SARSA state cap 10000");
  assert(C.MENACE_MAX_BOXES === 10000, "MENACE box cap 10000");
})();

(function testSarsaGreedyUsesExploreFlag() {
  var agent = APP.createShapeAgent("sarsa");
  var board = G.emptyBoard();
  var key = G.boardKey(board) + "|" + C.X;
  agent.ensureQ(key);
  agent.qTable[key] = [0, 0, 0, 12, 0, 0, 0];
  agent.visits[key] = [1, 1, 1, 1, 1, 1, 1];
  var i;
  for (i = 0; i < 25; i += 1) {
    var move = agent.chooseMove(board, C.X, G.boardKey, G.legalMoves, false);
    assert(move === 3, "SARSA greedy respects explore=false");
  }
})();

(function testQGreedyAllowsUnvisitedOverTriedLoser() {
  var agent = APP.createShapeAgent("qtable");
  var board = G.emptyBoard();
  var key = G.boardKey(board) + "|" + C.X;
  agent.ensureQ(key);
  agent.qTable[key] = [-4, 0, 0, 0, 0, 0, 0];
  agent.visits[key] = [5, 0, 0, 0, 0, 0, 0];
  var counts = [0, 0, 0, 0, 0, 0, 0];
  var i;
  for (i = 0; i < 120; i += 1) {
    counts[agent.chooseMove(board, C.X, G.boardKey, G.legalMoves, false)] += 1;
  }
  assert(counts[0] === 0, "greedy must not stick to visited loser when Q=0 alternatives exist");
  assert(counts.slice(1).reduce(function (a, b) { return a + b; }, 0) === 120,
    "greedy spreads over unvisited / equal-Q moves");
})();

(function testTabularTakesImmediateWin() {
  var agent = APP.createShapeAgent("qtable");
  /* Three red in a row on bottom; col 3 (0-based) wins. */
  var board = G.emptyBoard();
  board = G.applyMove(board, C.X, 0);
  board = G.applyMove(board, C.O, 6);
  board = G.applyMove(board, C.X, 1);
  board = G.applyMove(board, C.O, 6);
  board = G.applyMove(board, C.X, 2);
  board = G.applyMove(board, C.O, 6);
  var move = agent.chooseMove(board, C.X, G.boardKey, G.legalMoves, false);
  assert(move === 3, "Q takes immediate win in column 4");
  var menace = APP.createShapeAgent("menace");
  assert(menace.chooseMove(board, C.X, G.boardKey, G.legalMoves, true) === 3,
    "MENACE takes immediate win even when exploring");
})();

(function testGeneticMenaceEvolvePlaysGames() {
  var agent = APP.createShapeAgent("genetic_menace");
  assert(agent.champion().boxCount() === 0, "genetic starts with empty boxes");
  APP.shapeEvolveOne(agent, "genetic_menace");
  assert(agent.generation === 1, "genetic evolve bumps generation");
  assert(agent.champion().boxCount() > 0, "fitness games fill MENACE boxes");
})();

if (failures.length) {
  console.error("FAILED " + failures.length + " assertion(s):");
  failures.forEach(function (f) { console.error(" - " + f); });
  process.exit(1);
}
console.log("test-shape-lab: ok (" + [
  "agents", "train", "eval", "mutate", "gen0", "warmup", "trainClock", "beat50",
  "rr", "search", "survivors", "union", "tabs", "trainOne", "play",
  "save", "load", "export", "import", "tabularCaps", "sarsaGreedy", "qUnvisited", "tabularTactics", "geneticFitness",
  "peerEvenSpread", "playBaselines", "traineeObserve", "promoteTrainee", "toBeat100"
].join(", ") + ")");
