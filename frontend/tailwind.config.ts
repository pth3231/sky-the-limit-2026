import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        draft: {
          ground: "#080708",
          surface: "#151114",
          raised: "#21191b",
          line: "#493b34",
          text: "#f1ece5",
          muted: "#aaa099",
          gold: "#c5a36a",
          cloud9: "#74b9e5",
          opponent: "#d85c62",
          excluded: "#934951",
          warning: "#e2bf78",
        },
      },
      fontFamily: {
        display: ["Arial Narrow", "Liberation Sans Narrow", "Arial", "sans-serif"],
        body: ["Arial", "Helvetica", "sans-serif"],
        mono: ["IBM Plex Mono", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
