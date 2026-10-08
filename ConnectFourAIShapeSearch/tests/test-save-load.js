/* Round-trip checkpoint save/load (Node uses localStorage; browser uses IndexedDB). */
var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

var store = {};
global.window = global;
global.localStorage = {
  getItem: function (k) { return store[k] || null; },
  setItem: function (k, v) { store[k] = v; },
  removeItem: function (k) { delete store[k]; }
};
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

var app = new C4_APP.Application();
app.registry.states.reinforce2.gamesPlayed = 4242;
app.registry.states.reinforce2.phase = "minimax";
app.registry.states.reinforce2.minimaxDepth = 1.3;
app.latestResult = {
  points: { nn2: 5.5, qtable: 3.0, mcts: 4.0 },
  wins: { nn2: 5, qtable: 3, mcts: 4 },
  draws: { nn2: 1, qtable: 0, mcts: 0 },
  losses: { nn2: 0, qtable: 2, mcts: 1 },
  lossTo: { nn2: {}, qtable: { nn2: 2 }, mcts: { nn2: 1 } },
  gamesPlayed: 10
};
var saveStatus = app.saveNow();
if (!saveStatus.ok) {
  console.log("FAIL save:", saveStatus.error);
  process.exit(1);
}

var app2 = new C4_APP.Application();
if (app2.registry.states.reinforce2.gamesPlayed !== 4242) {
  console.log("FAIL gamesPlayed", app2.registry.states.reinforce2.gamesPlayed);
  process.exit(1);
}
if (!app2.latestResult || typeof app2.latestResult.ranked !== "function") {
  console.log("FAIL latestResult ranked()");
  process.exit(1);
}
var ranked = app2.latestResult.ranked();
if (!ranked.length || ranked[0][0] !== "nn2") {
  console.log("FAIL latestResult order", ranked);
  process.exit(1);
}
try {
  app2.getSnapshot();
} catch (e) {
  console.log("FAIL getSnapshot with loaded latestResult:", e.message);
  process.exit(1);
}
console.log("OK save/load roundtrip", saveStatus.bytes, "bytes", saveStatus.backend);
process.exit(0);
