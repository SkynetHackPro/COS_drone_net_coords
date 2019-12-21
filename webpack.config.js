const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const {VueLoaderPlugin} = require('vue-loader');

module.exports = {
  mode: 'development',
  entry: './index.js',
  devtool: 'eval-source-map',

  resolve: {
    extensions: ['*', '.js', '.vue', '.json', '.html'],
  },
  module: {
    rules: [
      {
        test: /\.vue$/,
        use: 'vue-loader'
      },
      {
        test: /\.(css|postcss)?$/,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: 'css-loader',
            options: {url: false},
          }
        ]
      }
    ]
  },
  plugins: [
    new VueLoaderPlugin(),
    new HtmlWebpackPlugin({
      title: 'COS example',
      inject: false,
      template: require('html-webpack-template'),
      appMountId: 'app',
    }),
  ]
}
;
