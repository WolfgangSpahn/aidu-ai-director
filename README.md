# AIDu AI Director

`aidu.ai.director` is the workflow orchestration layer of the AIDu ecosystem.

The director coordinates actors and executes educational or collaborative workflows.

Actors:

* receive tasks,
* perform local reasoning,
* produce artifacts,
* update workflow state.

The director:

* executes workflows,
* dispatches tasks,
* maintains global state,
* evaluates conditions,
* controls loops,
* decides when execution stops.

This separates local cognition from global workflow control.

---

# Execution Model

```text
Workflow
    ↓
 Director
    ↓
  Actor
    ↓
Artifacts + State Updates
    ↓
 Director
    ↓
Next Action
```

Workflows define the process.

The director executes actions and routes them to actors.

---

# Example

```text
Tutor: Present task
        ↓

while not solved

Student: Work on task
        ↓

Tutor: Evaluate answer
        ↓

Tutor: Summarize solution
```

Execution continues until the workflow completes or a stop condition is reached.

---

# Architecture

```text
Workflow
    ↓
Director
    ↓
Actor
    ↓
Controller
    ↓
Processor
```

Responsibilities:

* Workflow — defines the process
* Director — executes workflows
* Actor — represents a participant
* Controller — coordinates local reasoning
* Processor — performs specialized tasks

Examples:

* AI Tutor
* AI Student
* Human Student
* Human Teacher

---

# Workflow Example

```python
class TutoringWorkflow(Workflow):

    def run(self, director):

        yield Action(
            actor="ai_tutor",
            task="Present the exercise"
        )

        while not director.state.data.get("solved", False):

            yield Action(
                actor="human_student",
                task="Work on the exercise"
            )

            yield Action(
                actor="ai_tutor",
                task="Evaluate the answer"
            )
```

Workflows are implemented directly in Python, enabling loops, conditions, and reusable workflow libraries.

---

# Development

## Install Local Dependencies

```toml
[tool.uv.sources]
aidu-ai-actor = { path = "../aidu-ai-actor", editable = true }
```

## Run Example

```bash
python -m aidu.ai.director.main
```

## Run Smoke Tests

```bash
python -m aidu.ai.director.director
```

Smoke tests verify:

* workflow execution,
* actor dispatching,
* state updates,
* workflow loops.

---

# Design Goals

Future versions may support:

* distributed actors,
* curriculum integration,
* workflow persistence,
* parallel branches,
* workflow analytics.

---

# License

MIT License.

Copyright (c) 2026 Wolfgang Spahn, PHBern.
