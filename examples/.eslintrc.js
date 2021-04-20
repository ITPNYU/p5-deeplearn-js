const p5SoundGlobals = require("./p5SoundGlobals");

module.exports = {
  extends: ["p5js"],
  globals: {
    document: true,
    ml5: false,
    p5: false,
    ...p5SoundGlobals,
  },
  rules: {
    "no-use-before-define": [
      "error",
      {
        functions: false,
        classes: true,
        variables: true,
      },
    ],
    "no-unused-vars": ["error", { varsIgnorePattern: "^setup$|^draw$|^preload$" }],
    "new-cap": 0,
    "no-underscore-dangle": 0,
    "no-console": 0,
  },
  env: {
    browser: true,
    jasmine: true,
  },
};
