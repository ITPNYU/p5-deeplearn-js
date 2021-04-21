// Copyright (c) 2018 ml5
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import { join, resolve } from 'path';

const include = join(__dirname, 'src');

export const indexEntryWithBabel = ['babel-polyfill', './src/index.js'];
export const developmentPort = 8080;

export default {
  name: 'ml5',
  entry: indexEntryWithBabel,
  output: {
    path: resolve(__dirname, 'dist'),
    publicPath: '/',
    libraryTarget: 'umd',
    filename: 'ml5.js',
    library: 'ml5',
  },
  module: {
    rules: [
      {
        enforce: 'pre',
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          emitWarning: true
        },
      },
      {
        test: /\.js$/,
        loader: 'babel-loader',
        include,
      },
    ],
  },
  node: {
    fs: "empty"
  },
};
