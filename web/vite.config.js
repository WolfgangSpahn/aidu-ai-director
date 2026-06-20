import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss(), solid()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: {
      allow: ["../.."],
    },
  },
  build: {
    target: "esnext",
  },
});
