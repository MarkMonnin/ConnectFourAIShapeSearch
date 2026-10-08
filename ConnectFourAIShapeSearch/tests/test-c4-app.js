/**
 * Smoke: Shape Lab starts and publishes a snapshot (replaces old Training Application UI smoke).
 */
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
global.requestAnimationFrame = function (fn) {
  return setTimeout(fn, 0);
};
global.cancelAnimationFrame = function (id) {
  clearTimeout(id);
};
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

try {
  var snap = null;
  var lab = C4_SHAPE_LAB.init({
    onSnapshot: function (s) { snap = s; }
  });
  setTimeout(function () {
    lab.stop();
    if (!snap) {
      try {
        snap = lab.getSnapshot();
      } catch (e) {
        console.log("FAIL getSnapshot:", e.message);
        process.exit(1);
      }
    }
    if (!snap.search || typeof snap.search.statusLine !== "string") {
      console.log("FAIL: missing search.statusLine");
      process.exit(1);
    }
    if (!snap.catalog || !snap.catalog.length) {
      console.log("FAIL: empty catalog");
      process.exit(1);
    }
    console.log("OK:", snap.search.statusLine.slice(0, 100));
    process.exit(0);
  }, 500);
} catch (e) {
  console.log("init error:", e.stack);
  process.exit(1);
}
