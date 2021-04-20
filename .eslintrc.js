module.exports = {
  extends: ["airbnb-base", "prettier"],
  globals: {
    fetch: false,
    document: true,
  },
  rules: {
    "no-console": 0,
  },
  env: {
    browser: true,
    jasmine: true,
  },
  overrides: [
    {
      files: ["**/**_test.js"],
      globals: {
        ml5: false,
      },
    },
  ],
};
