/* Manual one-off: max depth reached from empty board at given time budgets. */
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

global.window = global;
global.performance = { now: function () { return Date.now(); } };

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;

var BUDGETS_SEC = [1, 5, 30];

console.log("Empty board, red (X) to move, chooseMoveUntilDeadline (make/undo minimax)");
console.log("");

var i;
for (i = 0; i < BUDGETS_SEC.length; i += 1) {
  var budgetMs = BUDGETS_SEC[i] * 1000;
  var t0 = performance.now();
  var res = M.chooseMoveUntilDeadline(
    G.emptyBoard(), C.X, t0 + budgetMs, Math.random);
  var elapsed = performance.now() - t0;
  console.log(
    BUDGETS_SEC[i] + "s budget: depth " + res.depth +
    ", move col " + (res.move + 1) +
    ", elapsed " + elapsed.toFixed(0) + "ms"
  );
}
