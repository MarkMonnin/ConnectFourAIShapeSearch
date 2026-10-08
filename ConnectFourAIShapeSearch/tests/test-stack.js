var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

var sandbox = {
  window: {}, console: console, Math: Math, Date: Date, JSON: JSON,
  Object: Object, Array: Array, parseFloat: parseFloat, parseInt: parseInt,
  isNaN: isNaN, Infinity: Infinity, setTimeout: setTimeout, clearTimeout: clearTimeout,
  localStorage: { getItem: function () { return null; }, setItem: function () {}, removeItem: function () {} },
  performance: { now: function () { return Date.now(); } },
  requestAnimationFrame: function (fn) { return setTimeout(fn, 0); },
  cancelAnimationFrame: clearTimeout
};
sandbox.window = sandbox;
vm.createContext(sandbox);
["c4_js/constants.js", "c4_js/game.js", "c4_js/minimax.js", "c4-network.js", "c4-opponents.js", "c4_js/tfjs-agent.js", "c4_js/engine.js"].forEach(function (f) {
  vm.runInContext(fs.readFileSync(path.join(root, f), "utf8"), sandbox);
});
var errors = 0;
sandbox.C4_APP.init({
  onSnapshot: function () {},
  onError: function () { errors += 1; }
});
setTimeout(function () {
  if (errors) { console.log("FAIL errors=" + errors); process.exit(1); }
  console.log("OK no stack overflow");
  process.exit(0);
}, 3000);
