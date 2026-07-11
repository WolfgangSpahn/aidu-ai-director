/*
 * Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
 *
 * MIT License — see LICENSE file for details.
 * If you use this software in academic work, citation of the original author is requested.
 */
import { createSignal } from "solid-js";
import Dialog from "../../../aidu-frontend-dialog/src/dialog";

const exampleDialogTurns = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is the capital of France?" },
  { role: "assistant", content: "The capital of France is Paris." },
];




export default function App() {
  const [visible] = createSignal(true);

  async function handleSend(turn) {
    try {
      const response = await fetch(
        "http://127.0.0.1:8100/input",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            role: turn.role,
            content: turn.content,
          }),
        },
      );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}`,
        );
      }

      const result = await response.json();

      console.log(
        "Director response:",
        result,
      );
    } catch (err) {
      console.error(
        "Failed to send message:",
        err,
      );
    }
  }
  return (
    <main class="app-shell">
      <section class="frame">
        <header class="hero">
          <p class="eyebrow">AIDU Director</p>
          <h1>Dialog Dev Sandbox</h1>
          <p class="subtitle">Monitoring live dialog turns from SSE.</p>
        </header>

        <section class="card">
          {visible() && (
            <Dialog
              initialTurns={exampleDialogTurns}
              sseUrl="http://127.0.0.1:8100/events"
              onSend={handleSend}
            />
          )}
        </section>
      </section>
    </main>
  );
}
