/* Regression: minimax nemesis replays must not re-enter handleExploreLoss. */
var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

function mockStorage() {
  var m = {};
  return {
    getItem: function (k) { return m[k] || null; },
    setItem: function (k, v) { m[k] = v; },
    removeItem: function (k) { delete m[k]; }
  };
}

global.window = global;
global.localStorage = mockStorage();
global.performance = { now: function () { return Date.now(); } };
global.requestAnimationFrame = function () { return 1; };
global.cancelAnimationFrame = function () {};

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-network.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-opponents.js"), "utf8"));
require(path.join(root, "c4_js/tfjs-agent.js"));
require(path.join(root, "c4_js/engine.js"));

var app = C4_APP.init({ onSnapshot: function () {} });
app.running = false;
var engine = app.training;
var state = app.registry.states.reinforce2;
state.phase = "minimax";
state.minimaxDepth = 1;

var lastTrackNemesis = null;
var origMinimax = engine.trainVsMinimax.bind(engine);
engine.trainVsMinimax = function (algoId, level, explore, train, seed, trackNemesis) {
  lastTrackNemesis = trackNemesis;
  return origMinimax.apply(this, arguments);
};

engine.playSeededEvalGame("reinforce2", 42, 1, true, true);
if (lastTrackNemesis !== false) {
  console.log("FAIL playSeededEvalGame should pass trackNemesis=false, got " + lastTrackNemesis);
  process.exit(1);
}

var maxDepth = 0;
var curDepth = 0;
var origHandle = engine.handleExploreLoss.bind(engine);
engine.handleExploreLoss = function () {
  curDepth += 1;
  if (curDepth > maxDepth) {
    maxDepth = curDepth;
  }
  if (curDepth > 3) {
    throw new Error("handleExploreLoss nested too deep: " + curDepth);
  }
  try {
    return origHandle.apply(this, arguments);
  } finally {
    curDepth -= 1;
  }
};

var lossSeed = null;
var s;
for (s = 1; s < 10000; s += 1) {
  if (engine.trainVsMinimax("reinforce2", 1, false, false, s, true) === "loss") {
    lossSeed = s;
    break;
  }
}
if (lossSeed === null) {
  console.log("SKIP: no greedy loss seed found");
  process.exit(0);
}

engine.handleExploreLoss("reinforce2", lossSeed, 1);
if (maxDepth > 1) {
  console.log("FAIL handleExploreLoss maxDepth=" + maxDepth);
  process.exit(1);
}

console.log("OK reinforce nemesis replay trackNemesis=false maxDepth=" + maxDepth);
process.exit(0);
