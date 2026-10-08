/* Timed search lastDepth plumbing (Node). */
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

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;
var APP = global.C4_APP;
var failures = [];

function assert(cond, msg) {
  if (!cond) {
    failures.push(msg);
  }
}

var b = G.emptyBoard();
var value = new APP.NnValueAgent();
var move = value.chooseMoveTimed(b, C.X, 40);
assert(typeof move === "number", "NnValueAgent timed returns move number");
assert(G.legalMoves(b).indexOf(move) >= 0, "NnValueAgent timed move legal");
assert(value.lastDepth > 0, "NnValueAgent lastDepth > 0 after timed search");

var timed = M.chooseMoveUntilDeadline(b, C.X, Date.now() + 40, Math.random.bind(Math));
assert(timed && timed.depth > 0, "chooseMoveUntilDeadline reports depth");

if (failures.length) {
  console.error("FAIL:\n" + failures.join("\n"));
  process.exit(1);
}
console.log("OK: timed depth nn=" + value.lastDepth + " mm=" + timed.depth);
