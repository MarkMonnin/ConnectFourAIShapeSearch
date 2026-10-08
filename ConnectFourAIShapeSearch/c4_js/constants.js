(function (global) {
  "use strict";

  var NN_POLICY_LAYER_SIZES = {
    2: [78, 64, 78, 7],
    3: [78, 64, 56, 48, 7]
  };
  var NN_VALUE_LAYER_SIZES = {
    2: [77, 64, 77, 1],
    3: [77, 64, 56, 48, 1]
  };
  /* Experiment: 5 hidden layers, wider than the 2L/3L value nets. */
  var TFJS_VALUE_5L_SIZES = [77, 256, 192, 160, 128, 96, 1];

  global.C4_CONSTANTS = {
    ROWS: 6,
    COLS: 7,
    CELLS: 42,
    ACTIONS: 7,
    EMPTY: "-",
    X: "X",
    O: "O",
    NN_POLICY_LAYER_SIZES: NN_POLICY_LAYER_SIZES,
    NN_VALUE_LAYER_SIZES: NN_VALUE_LAYER_SIZES,
    TFJS_VALUE_5L_SIZES: TFJS_VALUE_5L_SIZES,
    TRAINING_ALGO_IDS: [
      "nn2", "nn3",
      "nn_value2", "nn_value3",
      "neat", "neat_value",
      "reinforce2", "reinforce3",
      "reinforce_value2", "reinforce_value3",
      "tfjs", "tfjs_value", "tfjs_value5",
      "qtable", "sarsa", "menace", "genetic_menace"
    ],
    /** Per-game learners only; excludes population-evolution bots (NEAT, genetic MENACE). */
    NEMESIS_ALGO_IDS: [
      "nn2", "nn3",
      "nn_value2", "nn_value3",
      "reinforce2", "reinforce3",
      "reinforce_value2", "reinforce_value3",
      "tfjs", "tfjs_value", "tfjs_value5",
      "qtable", "sarsa", "menace"
    ],
    VALUE_AGENT_IDS: [
      "nn_value2", "nn_value3",
      "reinforce_value2", "reinforce_value3",
      "neat_value",
      "tfjs_value", "tfjs_value5"
    ],
    ALL_ALGO_IDS: null,
    ALGO_NAMES: {
      nn2: "Backprop NN (2L)",
      nn3: "Backprop NN (3L)",
      nn_value2: "Backprop NN value (2L)",
      nn_value3: "Backprop NN value (3L)",
      neat: "NEAT",
      neat_value: "NEAT (value)",
      reinforce2: "REINFORCE (2L)",
      reinforce3: "REINFORCE (3L)",
      reinforce_value2: "REINFORCE value (2L)",
      reinforce_value3: "REINFORCE value (3L)",
      tfjs: "TF.js hybrid",
      tfjs_value: "TF.js value 2L",
      tfjs_value5: "TF.js value 5L",
      qtable: "Q-Learning",
      sarsa: "SARSA",
      menace: "MENACE",
      genetic_menace: "Genetic MENACE",
      mcts: "MCTS",
      minimax: "Minimax"
    },
    ROTATION_SECONDS: 35,
    TOURNAMENT_INTERVAL: 30,
    DROP_FIRST_TOURNAMENT: 20,
    DROP_INTERVAL: 10,
    MIN_ACTIVE_TRAINING_BOTS: 3,
    ROLLING_TOURNAMENT_WINDOW: 5,
    UI_REFRESH_SECONDS: 0.3,
    GRADUATION_TARGET_WIN_RATE: 0.5,
    GREEDY_EVAL_GAMES: 100,
    /** Greedy win rate at or above this promotes by two level steps (e.g. 0.7 -> 0.9). */
    GREEDY_SKIP_LEVEL_WIN_RATE: 0.95,
    MAX_MINIMAX_LEVEL: 7.9,
    MINIMAX_LEVEL_STEP: 0.1,
    MINIMAX_START_LEVEL: 0.1,
    Q_MAX_STATES: 10000,
    MENACE_MAX_BOXES: 10000,
    /** Tabular bots stop training when games AND states/boxes both hit cap. */
    TABULAR_MAX_TRAINING_GAMES: 10000,
    TABULAR_ALGO_IDS: ["qtable", "sarsa", "menace", "genetic_menace"],
    /** Analysis features appended to NN inputs (threats, heights, win/block cols). */
    BOARD_FEATURE_DIM: 35,
    POLICY_INPUT_DIM: 78,
    VALUE_INPUT_DIM: 77,
    AUTO_SAVE_SECONDS: 180,
    SAVE_SIZE_LIMIT: 4000000,
    TOURNAMENT_HISTORY_LEN: 5,
    SCORE_WIN: 1,
    SCORE_DRAW: 0.5,
    SCORE_LOSS: 0,
    MCTS_SIMS: 100,
    NN_TRAIN_MINIMAX_DEPTH: 1,
    NN_TOURNAMENT_DEPTH: 1,
    /** Target wall-clock duration for a full tournament round. */
    TOURNAMENT_TARGET_MS: 10000,
    /** Conservative ply cap per game when estimating first move budget (full board). */
    TOURNAMENT_MAX_PLIES_PER_GAME: 42,
    TOURNAMENT_MOVE_BUDGET_MIN_MS: 0.25,
    TOURNAMENT_MOVE_BUDGET_MAX_MS: 200,
    /** 0 = frozen budget, 1 = jump to corrected budget; 0.5 = gradual. */
    /** Default depth when minimax has no time budget (play observe 0s, tournament estimate). */
    TOURNAMENT_MINIMAX_MAX_DEPTH: 7,
    ELIMINATION_ENABLED: true,
    NEAT_POP_SIZE: 20,
    GENETIC_MENACE_POP: 20,
    /** Fitness evals per genome/individual (NEAT + genetic MENACE). */
    NEAT_FITNESS_GAMES: 10,
    STORAGE_KEY: "learnC4_js_v1",
    /** Per-bot slice if session time were split evenly (actual slices use remaining budget). */
    sliceSeconds: function (n) {
      return C4_CONSTANTS.TOURNAMENT_INTERVAL / Math.max(n, 1);
    },
    TRAINING_SLICE_SCALE_MIN: 0.25,
    TRAINING_SLICE_SCALE_MAX: 4
  };
  C4_CONSTANTS.ALL_ALGO_IDS = C4_CONSTANTS.TRAINING_ALGO_IDS.concat(["mcts", "minimax"]);
})(window);
