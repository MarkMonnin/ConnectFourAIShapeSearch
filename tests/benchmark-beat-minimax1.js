/**
 * Benchmark: wall-clock time until a bot beats fixed FractionalMinimax (greedy).
 *
 * Protocol (all bots, including NEAT / genetic MENACE):
 *   1. Eval first: EVAL_GAMES greedy vs FractionalMinimax (explore off, no learn).
 *   2. If winRate > 0.5, stop (beat). Draws count as non-wins.
 *   3. Else train vs random (explore on) for the same wall-clock as that eval,
 *      then repeat from step 1.
 *   - Primary score: seconds to beat.
 *   - Order: PRIORITY_ALGOS first. After the best time among finishers is known,
 *     remaining bots get max(10x that time, --min-seconds default 10) then give up.
 *
 * Run:
 *   node tests/benchmark-beat-minimax1.js
 *   node tests/benchmark-beat-minimax1.js --eval-level=2 --eval-games=40
 *   node tests/benchmark-beat-minimax1.js --min-seconds=10 --out=tests/out.json
 */
var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

global.window = global;
global.localStorage = {
  getItem: function () { return null; },
  setItem: function () {},
  removeItem: function () {}
};
global.performance = { now: function () { return Date.now(); } };

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-network.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-opponents.js"), "utf8"));
require(path.join(root, "c4_js/tfjs-agent.js"));
require(path.join(root, "c4_js/engine.js"));

var APP = global.C4_APP;
var C = global.C4_CONSTANTS;

var DEFAULT_EVAL_LEVEL = 1.0;
var DEFAULT_EVAL_GAMES = 40;
var TIME_CAP_MULT = 10;
/** Floor on per-bot wall budget after a best time is known (avoids tiny 10x caps). */
var DEFAULT_MIN_SECONDS = 10;

/** Likely fast learners first (value / hybrid), then other NNs, then tabular/GA. */
var PRIORITY_ALGOS = [
  "nn_value2",
  "tfjs_value",
  "reinforce_value2",
  "tfjs",
  "nn2",
  "reinforce2",
  "nn_value3",
  "reinforce_value3",
  "tfjs_value5",
  "nn3",
  "reinforce3",
  "qtable",
  "sarsa",
  "menace",
  "neat_value",
  "neat",
  "genetic_menace"
];

function parseArgs(argv) {
  var opts = {
    algos: PRIORITY_ALGOS.slice(),
    evalGames: DEFAULT_EVAL_GAMES,
    timeCapMult: TIME_CAP_MULT,
    evalLevel: DEFAULT_EVAL_LEVEL,
    minSeconds: DEFAULT_MIN_SECONDS,
    maxSeconds: null,
    outPath: null
  };
  var i;
  for (i = 2; i < argv.length; i += 1) {
    var a = argv[i];
    if (a.indexOf("--eval-games=") === 0) {
      opts.evalGames = parseInt(a.split("=")[1], 10);
    } else if (a.indexOf("--time-cap-mult=") === 0) {
      opts.timeCapMult = parseFloat(a.split("=")[1]);
    } else if (a.indexOf("--eval-level=") === 0) {
      opts.evalLevel = parseFloat(a.split("=")[1]);
    } else if (a.indexOf("--min-seconds=") === 0) {
      opts.minSeconds = parseFloat(a.split("=")[1]);
    } else if (a.indexOf("--max-seconds=") === 0) {
      opts.maxSeconds = parseFloat(a.split("=")[1]);
    } else if (a.indexOf("--out=") === 0) {
      opts.outPath = a.split("=")[1];
    } else if (a.indexOf("--train-games=") === 0) {
      /* Ignored: training is wall-clock matched to the preceding eval. */
    } else if (a.indexOf("--") !== 0) {
      opts.algos = a.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
    }
  }
  return opts;
}

/**
 * Per-bot wall budget. null = uncapped (first bot when no --max-seconds).
 * After a best time exists: max(10x best, minSeconds), then optional maxSeconds ceiling.
 */
function resolveBudget(bestSeconds, opts) {
  var budget;
  if (bestSeconds === null) {
    budget = opts.maxSeconds;
  } else {
    budget = bestSeconds * opts.timeCapMult;
  }
  if (budget === null) {
    return null;
  }
  if (typeof opts.minSeconds === "number" && budget < opts.minSeconds) {
    budget = opts.minSeconds;
  }
  if (opts.maxSeconds !== null && budget > opts.maxSeconds) {
    budget = opts.maxSeconds;
  }
  return budget;
}

function trainOneStep(te, algoId, seed) {
  var state = te.registry.states[algoId];
  if (algoId === "neat") {
    te.neatOneGeneration(state);
    return;
  }
  if (algoId === "neat_value") {
    te.neatValueOneGeneration(state);
    return;
  }
  if (algoId === "genetic_menace") {
    te.geneticOneGeneration(state);
    return;
  }
  state.phase = "random";
  te.trainVsRandom(algoId, true, true, seed, false);
}

/**
 * Train until wall-clock deadline (inclusive of in-flight steps that overshoot).
 * Always runs at least one step so a slow GA generation still gets a turn.
 */
function trainForDuration(te, algoId, seedRef, durationMs, hardDeadlineMs) {
  var trainDeadline = Date.now() + Math.max(0, durationMs);
  var games = 0;
  var t0 = Date.now();
  do {
    if (hardDeadlineMs !== null && Date.now() >= hardDeadlineMs) {
      break;
    }
    trainOneStep(te, algoId, seedRef.value);
    seedRef.value += 1;
    games += 1;
  } while (Date.now() < trainDeadline &&
    (hardDeadlineMs === null || Date.now() < hardDeadlineMs));
  return {
    trainGames: games,
    trainSeconds: (Date.now() - t0) / 1000
  };
}

function evalVsMinimax(te, algoId, evalGames, seedBase, evalLevel) {
  var wins = 0;
  var draws = 0;
  var losses = 0;
  var i;
  var state = te.registry.states[algoId];
  state.phase = "minimax";
  state.minimaxDepth = evalLevel;
  for (i = 0; i < evalGames; i += 1) {
    var seed = seedBase + i;
    var result;
    if (algoId === "neat") {
      result = te.playNeatGenomeVsMinimax(
        te.registry.neat.bestGenomeForPlay(), evalLevel, false, seed
      );
    } else if (algoId === "neat_value") {
      result = te.playNeatValueGenomeVsMinimax(
        te.registry.neat_value.bestGenomeForPlay(), evalLevel, false, seed
      );
    } else if (algoId === "genetic_menace") {
      result = te.playMenaceVsMinimax(
        te.registry.genetic_menace.champion(), evalLevel, false, false, seed
      );
    } else {
      result = te.trainVsMinimax(algoId, evalLevel, false, false, seed, false);
    }
    if (result === "win") {
      wins += 1;
    } else if (result === "draw") {
      draws += 1;
    } else {
      losses += 1;
    }
  }
  state.phase = "random";
  return {
    wins: wins,
    draws: draws,
    losses: losses,
    winRate: wins / evalGames
  };
}

function overallDeadlineMs(t0, timeBudgetSec) {
  if (timeBudgetSec === null) {
    return null;
  }
  return t0 + timeBudgetSec * 1000;
}

/**
 * @param {number|null} timeBudgetSec null = no wall-clock cap (priority phase)
 */
function runAlgo(algoId, opts, timeBudgetSec) {
  var app = new APP.Application(true);
  var te = app.training;
  if (!app.registry.states[algoId]) {
    return { algo: algoId, error: "unknown algo" };
  }
  var name = C.ALGO_NAMES[algoId] || algoId;
  var seedRef = { value: 1 };
  var chunks = 0;
  var trainGames = 0;
  var t0 = Date.now();
  var hardDeadline = overallDeadlineMs(t0, timeBudgetSec);
  var history = [];
  var beat = false;
  var lastEval = null;
  var gaveUp = false;
  var giveUpReason = null;

  var budgetLabel = timeBudgetSec === null ? "uncapped" : (timeBudgetSec.toFixed(1) + "s cap");
  console.log("\n=== " + name + " (" + algoId + ") [" + budgetLabel + "] ===");

  while (true) {
    if (hardDeadline !== null && Date.now() >= hardDeadline) {
      gaveUp = true;
      giveUpReason = "time_cap";
      break;
    }
    chunks += 1;

    var evalT0 = Date.now();
    lastEval = evalVsMinimax(
      te, algoId, opts.evalGames, 1000000 + chunks * 1000, opts.evalLevel
    );
    var evalSeconds = (Date.now() - evalT0) / 1000;
    var elapsed = (Date.now() - t0) / 1000;

    history.push({
      chunk: chunks,
      phase: "eval",
      trainGames: trainGames,
      evalSeconds: Math.round(evalSeconds * 1000) / 1000,
      trainSeconds: 0,
      seconds: Math.round(elapsed * 100) / 100,
      winRate: lastEval.winRate,
      wins: lastEval.wins,
      draws: lastEval.draws,
      losses: lastEval.losses
    });
    console.log(
      "  t=" + elapsed.toFixed(1) + "s" +
      " eval=" + evalSeconds.toFixed(3) + "s" +
      " gamesTrain=" + trainGames +
      " W/D/L=" + lastEval.wins + "/" + lastEval.draws + "/" + lastEval.losses +
      " winRate=" + lastEval.winRate.toFixed(3)
    );

    if (lastEval.winRate > 0.5) {
      beat = true;
      break;
    }
    if (hardDeadline !== null && Date.now() >= hardDeadline) {
      gaveUp = true;
      giveUpReason = "time_cap";
      break;
    }

    var trainResult = trainForDuration(
      te, algoId, seedRef, evalSeconds * 1000, hardDeadline
    );
    trainGames += trainResult.trainGames;
    elapsed = (Date.now() - t0) / 1000;
    history.push({
      chunk: chunks,
      phase: "train",
      trainGames: trainGames,
      evalSeconds: Math.round(evalSeconds * 1000) / 1000,
      trainSeconds: Math.round(trainResult.trainSeconds * 1000) / 1000,
      seconds: Math.round(elapsed * 100) / 100,
      winRate: null,
      wins: null,
      draws: null,
      losses: null
    });
    console.log(
      "  t=" + elapsed.toFixed(1) + "s" +
      " train=" + trainResult.trainSeconds.toFixed(3) + "s" +
      " (+" + trainResult.trainGames + " games, total=" + trainGames + ")"
    );

    if (hardDeadline !== null && Date.now() >= hardDeadline) {
      /* Finish with another eval only if budget remains; otherwise stop. */
      if (Date.now() > hardDeadline) {
        gaveUp = true;
        giveUpReason = "time_cap";
        break;
      }
    }
  }

  var totalSec = (Date.now() - t0) / 1000;
  return {
    algo: algoId,
    name: name,
    beat: beat,
    gaveUp: gaveUp,
    giveUpReason: giveUpReason,
    secondsToBeat: beat ? Math.round(totalSec * 100) / 100 : null,
    secondsUsed: Math.round(totalSec * 100) / 100,
    trainGames: trainGames,
    chunks: chunks,
    finalWinRate: lastEval ? lastEval.winRate : null,
    timeBudgetSec: timeBudgetSec,
    protocol: "eval_then_train_matched",
    evalGames: opts.evalGames,
    history: history
  };
}

function main() {
  var opts = parseArgs(process.argv);
  var results = [];
  var bestSeconds = null;
  var i;
  var suiteT0 = Date.now();

  console.log(
    "Protocol: eval first, then train for that eval's wall time (all bots)."
  );
  console.log(
    "Eval level=" + opts.evalLevel +
    "; after first beat, others capped at max(" +
    opts.timeCapMult + "x best, " + opts.minSeconds + "s min)."
  );

  for (i = 0; i < opts.algos.length; i += 1) {
    var algoId = opts.algos[i];
    var budget = resolveBudget(bestSeconds, opts);
    var result = runAlgo(algoId, opts, budget);
    results.push(result);
    if (result.beat && result.secondsToBeat !== null) {
      if (bestSeconds === null || result.secondsToBeat < bestSeconds) {
        bestSeconds = result.secondsToBeat;
        console.log("  >> new best time: " + bestSeconds + "s (" + algoId + ")");
      }
    }
  }

  var ranked = results.filter(function (r) { return r.beat; }).slice().sort(function (a, b) {
    return a.secondsToBeat - b.secondsToBeat;
  });

  console.log("\n========== RANKING (seconds to beat minimax-" + opts.evalLevel + ") ==========");
  if (!ranked.length) {
    console.log("(none beat within budget)");
  } else {
    console.log(
      pad("#", 4) + pad("algo", 22) + pad("seconds", 10) + pad("games", 8) + "finalWR"
    );
    for (i = 0; i < ranked.length; i += 1) {
      var r = ranked[i];
      console.log(
        pad(String(i + 1), 4) +
        pad(r.algo, 22) +
        pad(r.secondsToBeat.toFixed(2), 10) +
        pad(String(r.trainGames), 8) +
        r.finalWinRate.toFixed(3)
      );
    }
  }

  console.log("\n========== ALL ==========");
  console.log(
    pad("algo", 22) + pad("result", 12) + pad("seconds", 10) + pad("games", 8) + "finalWR"
  );
  for (i = 0; i < results.length; i += 1) {
    var row = results[i];
    if (row.error) {
      console.log(pad(row.algo, 22) + row.error);
      continue;
    }
    var status = row.beat ? "BEAT" : (row.gaveUp ? "gave up" : "fail");
    console.log(
      pad(row.algo, 22) +
      pad(status, 12) +
      pad(row.beat ? row.secondsToBeat.toFixed(2) : row.secondsUsed.toFixed(2), 10) +
      pad(String(row.trainGames), 8) +
      (row.finalWinRate === null ? "-" : row.finalWinRate.toFixed(3))
    );
  }

  var suiteSec = Math.round((Date.now() - suiteT0) / 10) / 100;
  var outPath = opts.outPath ?
    path.resolve(root, opts.outPath) :
    path.join(root, "tests", "bench-beat-minimax1.json");
  fs.writeFileSync(outPath, JSON.stringify({
    evalLevel: opts.evalLevel,
    beatRule: "winRate > 0.5",
    primaryMetric: "secondsToBeat",
    protocol: "eval_then_train_matched",
    timeCapMult: opts.timeCapMult,
    bestSeconds: bestSeconds,
    suiteSeconds: suiteSec,
    trainOpponent: "random",
    opts: opts,
    ranking: ranked,
    results: results
  }, null, 2));
  console.log("\nwrote " + outPath);
  console.log("Suite wall-clock: " + suiteSec + "s");
  if (ranked.length) {
    console.log(
      "Fastest learner under this protocol: " +
      ranked[0].algo + " (" + ranked[0].secondsToBeat + "s)."
    );
  }
}

function pad(s, n) {
  s = String(s);
  while (s.length < n) {
    s += " ";
  }
  return s;
}

main();
