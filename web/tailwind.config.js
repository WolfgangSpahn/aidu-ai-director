/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx,html}",
    "../../aidu-frontend-dialog/src/**/*.{js,jsx,ts,tsx,html}",
  ],
  corePlugins: {
    preflight: false,
  },
};
