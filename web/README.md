# Dialog Demo

A small SolidJS demo that monitors dialog turns from the director SSE stream.

## Run

1. Start the director SSE source (for example from repo root):

   ```bash
   make smoke
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
- Director SSE defaults live in `src/aidu/ai/director/config.py`.
- Edit `src/App.jsx` if you want a different stream URL.
