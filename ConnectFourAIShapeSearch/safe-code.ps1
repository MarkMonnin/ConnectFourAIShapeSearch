# Safe: verify tabular visit-lock fix + shape-lab tests
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

node tests/test-shape-lab.js
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

node -e @"
global.window = global;
global.performance = { now: function () { return Date.now(); } };
var fs = require('fs');
var vm = require('vm');
require('./c4_js/constants.js');
require('./c4_js/game.js');
require('./c4_js/minimax.js');
vm.runInThisContext(fs.readFileSync('./c4-network.js','utf8'));
vm.runInThisContext(fs.readFileSync('./c4-opponents.js','utf8'));
require('./c4_js/tfjs-agent.js');
require('./c4_js/engine.js');
var APP = global.C4_APP;
function bench(family, sec) {
  var agent = APP.createShapeAgent(family);
  var t0 = performance.now();
  var games = 0;
  while ((performance.now() - t0) / 1000 < sec) {
    APP.shapeTrainVsRandom(agent, family, true, true);
    games += 1;
  }
  var ev = APP.shapeEvalVsRandom(agent, family, 200, 1);
  console.log(family + ' ' + sec + 's games=' + games + ' wr=' + ev.winRate.toFixed(3) +
    ' WDL=' + ev.wins + '/' + ev.draws + '/' + ev.losses);
  return ev.winRate;
}
var families = ['qtable','sarsa','menace'];
var f;
for (f = 0; f < families.length; f++) {
  var scores = [];
  var t;
  for (t = 0; t < 3; t++) scores.push(bench(families[f], 3));
  var avg = scores.reduce(function (a, b) { return a + b; }, 0) / scores.length;
  console.log(families[f] + ' avgWR=' + avg.toFixed(3));
  if (avg <= 0.65) {
    console.error('FAIL: ' + families[f] + ' avg WR ' + avg.toFixed(3) + ' <= 0.65');
    process.exit(1);
  }
}
console.log('OK: tabular beat Random after 3s');
"@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "OK: all"
