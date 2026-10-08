/* Save must succeed when tabular agents are full. */
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

function fillTabular(brain, prefix, count) {
  var i;
  for (i = 0; i < count; i += 1) {
    var key = prefix + i;
    brain.qTable[key] = [0, 0, 0, 0, 0, 0, 0];
    brain.visits[key] = [1, 1, 1, 1, 1, 1, 1];
    brain.lastUsedEpisode[key] = i;
  }
  brain.episodeCounter = count;
}

function fillMenace(brain, prefix, count) {
  var i;
  for (i = 0; i < count; i += 1) {
    var key = prefix + i;
    brain.boxes[key] = [1, 1, 1, 1, 1, 1, 1];
    brain.lastUsedEpisode[key] = i;
  }
  brain.episodeCounter = count;
}

fillTabular(app.registry.qtable, "q", 12000);
fillTabular(app.registry.sarsa, "s", 12000);
fillMenace(app.registry.menace, "m", 12000);
var gi;
for (gi = 0; gi < app.registry.genetic_menace.population.length; gi += 1) {
  fillMenace(app.registry.genetic_menace.population[gi], "g" + gi + ":", 8000);
}
app.registry.states.qtable.gamesPlayed = 99999;

var status = app.saveNow();
if (!status.ok) {
  console.log("FAIL save with full tabular:", status.error);
  process.exit(1);
}
console.log("OK full-tabular save tier=" + status.tier + " bytes=" + status.bytes);

var app2 = new C4_APP.Application();
if (app2.registry.states.qtable.gamesPlayed !== 99999) {
  console.log("FAIL state restore gamesPlayed", app2.registry.states.qtable.gamesPlayed);
  process.exit(1);
}
console.log("OK quota-safe save/load");
process.exit(0);
