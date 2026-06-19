# Dialog Demo

A small SolidJS demo that monitors dialog turns from the director SSE stream.

## Run

1. Start the director SSE source (for example from repo root):

   ```bash
   AIDU_DIRECTOR_SSE=1 AIDU_DIRECTOR_SSE_PORT=8100 make smoke
   ```

2. Start the demo:

   ```bash
   cd demo
   npm install
   npm run dev
   ```

3. Open `http://127.0.0.1:5173`.

## Notes

- Default SSE URL is `http://127.0.0.1:8100/events`.
- Edit `src/App.jsx` if you want a different stream URL.
