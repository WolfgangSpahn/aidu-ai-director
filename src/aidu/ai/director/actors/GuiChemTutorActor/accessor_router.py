"""Run learner-state assessments alongside the AI tutor workflow.

This module starts four independent assessments whenever the learner sends a
message:

* ``LearningTargetAssessor`` looks for evidence about what the learner knows.
    - It selects matching evidence from the permitted target list; all targets it does not select receive no knowledge update for that turn.
* ``StudentBeliefAssessor`` estimates states such as confidence and confusion.
    - It estimates the student’s current learning state from the latest turn; each estimate can change by at most 0.15 per student turn before replacing the previous belief value.
* ``AiSupervisor`` evaluates the tutor response that preceded the message.
* ``AiLabelIntervention`` identifies that response's dominant intervention.

These are *side tasks*: they may run concurrently while ``GuiInputRouter``
continues the main workflow.  When a side task finishes, its callback updates
the joined turn context.  The actor later serializes that context, allowing the
backend to persist it and pass it to the next turn or exercise.

This file coordinates the assessors but does not implement their prompts or the
knowledge-scoring mathematics.  Those responsibilities live in the assessor
classes and in ``GuiChemTutorActor.helpers`` respectively.
"""

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from aidu.ai.actor.turn_scope import get_turn_side_tasks
from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.ai_label_intervention import AiLabelIntervention
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, TextArtifact
from aidu.ai.core.artifacts import AppletArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context
from aidu.ai.core.session import SessionContext
from aidu.ai.core.supervisor import SupervisorState
from aidu.ai.llm.agent import Agent, EndAgent, WorkflowAgent
from aidu.ai.llm.clients.google import GoogleClient

from .gui_input_router import GuiInputRouter
from .helpers import (
    apply_target_assessment,
    update_context_with_belief_assessment,
    update_context_with_intervention_label,
    update_context_with_supervision_assessment,
)

logger = logging.getLogger(__name__)


def learner_evidence_text(artifact: Artifact, context: Context) -> str:
    """Extract only text that the learner actually wrote.

    Args:
        artifact: The current workflow artifact.  It may contain ordinary text
            or structured applet data.
        context: The shared turn context, which stores the original student
            message separately from applet telemetry.

    Returns:
        The learner's message as plain text.  For an ``AppletArtifact``, the
        function deliberately reads ``CurrentStudentMessage`` from the context
        instead of converting machine-generated applet values to text.

    Keeping these sources separate prevents the knowledge assessor from
    treating a value produced by the software as something the learner knew or
    explained.
    """

    # Applet artifacts contain machine telemetry, so recover only the separately
    # stored learner-authored message rather than serializing the whole artifact.
    if isinstance(artifact, AppletArtifact):
        return str(context.state.data.get("CurrentStudentMessage") or "").strip()

    # Ordinary text artifacts already represent the learner's authored input.
    return artifact.to_text()


class AssessorRouter(WorkflowAgent):
    """Launch the assessment side tasks for one learner turn.

    The router is a workflow agent, but it does not create a learner-facing
    message.  Its job is coordination:

    1. Extract the learner-authored text.
    2. Give each assessor its own context snapshot.
    3. Start the assessor tasks.
    4. Register callbacks that update the shared context after joining.
    5. Recommend ``GuiInputRouter`` as the next step in the main workflow.

    Attributes:
        learning_target_assessor: Produces target-specific knowledge evidence.
        student_belief_assessor: Estimates the learner's current affective and
            learning state.
        ai_supervisor: Evaluates the quality of the preceding tutor response.
        ai_label_intervention: Labels its dominant pedagogical intervention,
            when configured.

    The separate assessor contexts are important.  Assessors can read the same
    starting state without racing to modify it.  Their results are applied to
    the joined context by ``on_result`` callbacks.
    """

    target = GuiInputRouter
    continuations = []

    @staticmethod
    def _run_assessment(
        *,
        agent: Agent,
        context: Context,
        prompt_params: dict[str, Any],
        instruction: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Run an assessor and validate the basic shape of its JSON response."""
        tutor_index = context.state.data.get("LastTutorTurnIndex")
        learner_index = context.state.data.get("OutcomeStudentTurnIndex")
        if learner_index is None and prompt_params.get("outcome_evidence_available") != "false":
            learner_index = max(0, context.state.data.get("TurnIndex", 1) - 1)
        turn_label = lambda index: f"#{index + 1}" if isinstance(index, int) else "none"
        assessment_label = (
            f"assessment_id={uuid4().hex} assessor={type(agent).__name__} "
            f"tutor_turn={turn_label(tutor_index)} learner_turn={turn_label(learner_index)}"
        )
        context.control.data["assessment_log_label"] = assessment_label
        logger.info("Assessment started %s", assessment_label)
        # Execute through the production agent path while constraining the
        # response to a small JSON payload.
        result, _ = agent.run(
            artifact=TextArtifact(
                producer="AssessorRouter",
                step=context.step,
                content=instruction,
            ),
            context=context,
            agents=[agent, EndAgent()],
            ask_params=prompt_params,
            ask_config=AskConfig(
                json_mode=True,
                max_tokens=max_tokens,
                vendor_config={
                    "reasoning": {"effort": "minimal"},
                    "verbosity": "low",
                },
            ),
        )

        logger.info("Assessment raw result %s\n%s", assessment_label, result.content())

        # Decode at this boundary so callbacks never receive model text.
        try:
            decoded = json.loads(result.content())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{agent.id} returned non-JSON assessment content."
            ) from exc

        # Specific assessment models validate fields later; this shared method
        # guarantees the common top-level object shape.
        if not isinstance(decoded, dict):
            raise ValueError(f"{agent.id} assessment must be a JSON object.")
        return decoded

    def __init__(
        self,
        assessors: tuple[
            LearningTargetAssessor,
            StudentBeliefAssessor,
            AiSupervisor,
        ] | tuple[
            LearningTargetAssessor,
            StudentBeliefAssessor,
            AiSupervisor,
            AiLabelIntervention,
        ],
    ):
        """Store the assessor instances supplied by ``GuiChemTutorActor``.

        Args:
            assessors: The knowledge, belief, supervisor, and optional
                intervention-label assessors, in that order. Dependency
                injection keeps model/client construction in the actor and
                makes this coordinator easier to test.
        """
        # Preserve the declared tuple order as the actor's three named roles.
        (
            self.learning_target_assessor,
            self.student_belief_assessor,
            self.ai_supervisor,
            *labelers,
        ) = assessors
        self.ai_label_intervention = labelers[0] if labelers else None

    def run(self, artifact: Artifact, context: Context, agents=None) -> tuple[AgentResult, Context]:
        """Start all assessments and route the main workflow onward.

        Args:
            artifact: Current input artifact for the learner turn.
            context: Mutable context shared by the complete actor turn.
            agents: Optional workflow-agent collection required by the common
                agent interface.  This router does not use it directly.

        Returns:
            A pair containing an ``AgentResult`` with the routing recommendation
            and the unchanged main context.  Assessment results arrive later
            through the side-task callbacks.

        The callbacks have distinct effects on the joined context:

        * ``apply_target_assessment`` adds weighted evidence to
          ``StudentKnowledgeProgress``.
        * ``update_context_with_belief_assessment`` updates ``StudentBelief``.
        * ``update_context_with_supervision_assessment`` updates
          ``SupervisorState``.

        To add another assessor, students should follow the same pattern:
        create an assessor-specific context, build its prompt arguments, call
        ``side.spawn``, and apply validated output in an ``on_result`` callback.
        """
        # Establish one learner-authored evidence string shared by all assessors.
        current_turn = learner_evidence_text(artifact, context)
        side = get_turn_side_tasks(context)

        # Find direct evidence for configured learning targets. The callback
        # applies validated evidence only after the side task joins the turn.
        target_context = context.for_assessor()
        target_params = self.learning_target_assessor.build_prompt_args(
            context=target_context,
            current_message=current_turn,
        )
        side.spawn(
            "learning_target_assessor",
            lambda: self._run_assessment(
                agent=self.learning_target_assessor,
                prompt_params=target_params,
                context=target_context,
                instruction="Assess the current chemistry learning evidence.",
                max_tokens=512,
            ),
            on_result=lambda result, joined: apply_target_assessment(
                assessment=result,
                context=joined,
                current_message=current_turn,
            ),
        )

        # Classify observable speech acts independently of the final belief
        # vector; deterministic code derives that vector in the callback.
        belief_context = context.for_assessor()
        belief_params = self.student_belief_assessor.build_prompt_args(
            context=belief_context,
            current_message=current_turn,
        )
        side.spawn(
            "student_belief_assessor",
            lambda: self._run_assessment(
                agent=self.student_belief_assessor,
                prompt_params=belief_params,
                context=belief_context,
                instruction="Classify observable learner speech acts in this turn.",
                max_tokens=512,
            ),
            on_result=lambda result, joined: update_context_with_belief_assessment(
                assessment=result,
                context=joined,
                current_message=current_turn,
            ),
        )

        # Evaluate the preceding tutor response against the learner outcome,
        # keeping this assessment isolated from the other side tasks.
        supervisor_context = context.for_assessor()
        supervisor_params = self.ai_supervisor.build_prompt_args(
            context=supervisor_context,
        )
        side.spawn(
            "ai_supervisor",
            lambda: self._run_assessment(
                agent=self.ai_supervisor,
                prompt_params=supervisor_params,
                context=supervisor_context,
                instruction="Assess the preceding AI tutor response.",
                max_tokens=1024,
            ),
            on_result=lambda result, joined: update_context_with_supervision_assessment(
                assessment=result,
                context=joined,
                assessed_tutor_turn_index=context.state.data.get("LastTutorTurnIndex"),
                outcome_student_turn_index=max(0, context.state.data["TurnIndex"] - 1),
            ),
        )

        # Label the same completed tutor response independently. The callback
        # merges the label into the canonical supervision state, regardless of
        # which of the two tutor assessments completes first.
        if self.ai_label_intervention is not None:
            label_context = context.for_assessor()
            label_params = self.ai_label_intervention.build_prompt_args(
                context=label_context,
                current_student_message=current_turn,
            )
            side.spawn(
                "ai_label_intervention",
                lambda: self._run_assessment(
                    agent=self.ai_label_intervention,
                    prompt_params=label_params,
                    context=label_context,
                    instruction="Label the intervention in the preceding AI tutor response.",
                    max_tokens=256,
                ),
                on_result=lambda result, joined: update_context_with_intervention_label(
                    assessment=result,
                    context=joined,
                ),
            )

        # Side tasks must not delay the learner-facing workflow, so immediately
        # hand the original turn to the GUI input router.
        recommendation = self.register_recommendation(
            "assess_and_route",
            target=GuiInputRouter,
            continuations=[],
            utility=1.0,
            rationale="Assessors are running; continue the GUI-originated route.",
        )
        return self.result(artifacts=[], recommendations=[recommendation]), context


def smoke_test(
    *,
    tutor_turn: str,
    student_turn: str,
    learning_targets: list[dict[str, str]],
    client: Any | None = None,
) -> dict[str, Any]:
    """Run one exchange through the production ``AssessorRouter`` pattern.

    Args:
        tutor_turn: The question or instruction previously sent by the tutor.
        student_turn: The learner's response to assess.
        learning_targets: Teacher-defined targets.  Each dictionary must contain
            an ``id`` used in saved state and a human-readable ``text``.
        client: Optional LLM client, primarily for tests.  When omitted, the
            smoke test creates a ``GoogleClient`` using ``GOOGLE_API_KEY`` and
            the model selected by ``AIDU_SMOKE_TEST_MODEL`` (default:
            ``gemini-3.5-flash-lite``).

    Returns:
        The joined knowledge, belief, and tutor-supervision assessments plus
        their derived states.

    Raises:
        ValueError: If either turn is blank, no targets are supplied, the API
            key is missing, or the assessor response violates its contract.

    The context is local to this function, so the complete production update
    path is exercised without modifying or persisting a real learner session.
    """
    import os

    # Reject incomplete fixtures before spending an LLM request.
    if not tutor_turn.strip():
        raise ValueError("tutor_turn must not be blank.")
    if not student_turn.strip():
        raise ValueError("student_turn must not be blank.")
    if not learning_targets:
        raise ValueError("At least one learning target is required.")

    # Use an injected client in tests; otherwise construct the standalone
    # smoke-test client from the developer's local environment.
    if client is None:
        load_dotenv(Path.home() / ".env")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GOOGLE_API_KEY in ~/.env before "
                "running the smoke test."
            )
        model = os.getenv("AIDU_SMOKE_TEST_MODEL", "gemini-3.5-flash-lite")
        client = GoogleClient(model=model, config={}, api_key=api_key)

    # Build the same assessor bundle used by GuiChemTutorActor.
    router = AssessorRouter((
        LearningTargetAssessor(client=client, target=EndAgent),
        StudentBeliefAssessor(client=client, target=EndAgent),
        AiSupervisor(client=client, target=EndAgent),
    ))

    # Seed the deployed state shape. The target list creates one initial
    # knowledge entry per configured target.
    session = SessionContext(on_air=True, domain_targets=learning_targets)
    initial_knowledge = session.initial_student_knowledge_progress()
    initial_belief = StudentBelief()
    initial_supervision = SupervisorState.prior()
    initial_state = {
        "knowledge": initial_knowledge.model_dump(mode="json"),
        "belief": initial_belief.model_dump(mode="json"),
        "supervision": initial_supervision.model_dump(mode="json"),
    }
    context = Context()
    context.trace.messages = [{"role": "assistant", "content": tutor_turn.strip()}]
    context.state.data.update({
        "SessionContext": session,
        "StudentKnowledgeProgress": initial_knowledge,
        "StudentBelief": initial_belief,
        "SupervisorState": initial_supervision,
        "AppletState": {},
        "TurnIndex": 2,
        "LastTutorTurnIndex": 0,
        "OutcomeStudentTurnIndex": 1,
    })

    # Let the router spawn all three assessors, then use the normal turn-scoped
    # join to apply their callbacks to the shared context.
    router.run(
        TextArtifact(producer="SmokeTestLearner", step=0, content=student_turn.strip()),
        context,
    )
    # Unlike the live workflow, a smoke test must surface assessor failures to
    # the caller instead of logging them and continuing with a partial result.
    get_turn_side_tasks(context).join(context, raise_errors=True)

    # Expose targets, the one-or-two knowledge evidence items allowed by the
    # production contract, and the complete initial-to-final state transition.
    return {
        "targets": learning_targets,
        "initial_state": initial_state,
        "assessments": {
            "knowledge": context.control.data["learning_target_assessor_evidence"],
            "belief": context.control.data["student_belief_assessment"],
            "supervision": context.control.data["ai_supervision_assessment"],
        },
        "final_state": {
            "knowledge": context.state.data["StudentKnowledgeProgress"].model_dump(mode="json"),
            "belief": context.state.data["StudentBelief"].model_dump(mode="json"),
            "supervision": context.state.data["SupervisorState"].model_dump(mode="json"),
        },
    }


def learning_targets_for_domain(
    domain: str,
    *,
    teacher: str | None = None,
    class_name: str | None = None,
    data_path: Path | None = None,
) -> list[dict[str, str]]:
    """Load the learning targets configured for a curriculum domain.

    Args:
        domain: Stable curriculum-domain value, such as ``atomic-structure``.
        teacher: Optional teacher username for class-specific targets.
        class_name: Optional class name.  Supply it together with ``teacher``.
        data_path: Optional AIDu data directory.  By default this uses
            ``AIDU_DATA_PATH`` or the monorepo's ``aidu-data`` directory.

    Returns:
        Ordered target dictionaries accepted by :func:`smoke_test`.  When a
        teacher and class have no overrides, the domain's default curriculum
        targets are returned, matching the production fallback behavior.

    Raises:
        ValueError: If arguments are missing or the domain has no targets.
        FileNotFoundError: If the AIDu database cannot be found.
    """
    import os

    # Normalize and validate selector arguments before opening the database.
    domain = domain.strip()
    if not domain:
        raise ValueError("domain must not be blank.")
    if bool(teacher) != bool(class_name):
        raise ValueError("Supply --teacher and --class-name together.")

    # Resolve the same data location used by the monorepo unless explicitly
    # overridden by the caller or environment.
    default_data_path = Path(__file__).resolve().parents[7] / "aidu-data"
    resolved_data_path = data_path or Path(
        os.getenv("AIDU_DATA_PATH", default_data_path)
    ).expanduser()
    database = resolved_data_path / "data.db"
    if not database.is_file():
        raise FileNotFoundError(f"AIDu database not found: {database}")

    # Prefer class-specific targets, then fall back to curriculum defaults so
    # the smoke test mirrors production target selection.
    with sqlite3.connect(database) as connection:
        if teacher and class_name:
            rows = connection.execute(
                """
                SELECT target_id, text
                FROM targets
                WHERE teacher_username = ? AND class_name = ? AND domain_value = ?
                ORDER BY sort_order
                """,
                (teacher, class_name, domain),
            ).fetchall()
        else:
            rows = []

        if not rows:
            rows = connection.execute(
                """
                SELECT target.target_id, target.text
                FROM curriculum_target AS target
                JOIN curriculum_domain AS domain ON domain.id = target.domain_id
                WHERE domain.value = ?
                ORDER BY target.sort_order
                """,
                (domain,),
            ).fetchall()

    # Expose a small stable shape that can be inserted directly into the prompt.
    targets = [{"id": target_id, "text": text} for target_id, text in rows]
    if not targets:
        raise ValueError(f"No learning targets found for domain {domain!r}.")
    return targets


def log_smoke_test_report(
    *,
    domain: str,
    tutor_turn: str,
    student_turn: str,
    learning_targets: list[dict[str, str]],
    report: dict[str, Any],
) -> None:
    """Log the important stages of a smoke-test assessment with Rich markup."""
    # Derive omitted targets locally; the assessor intentionally returns only
    # evidence it selected rather than explicit negative entries for all targets.
    selected = report["assessments"]["knowledge"]["evidence"]
    selected_ids = {item["target"] for item in selected}
    rejected = [
        target for target in learning_targets if target["id"] not in selected_ids
    ]

    # Present knowledge evidence first because it is the primary smoke-test
    # result and includes both selected and omitted targets.
    logger.info("[bold cyan]Knowledge-assessor smoke test[/bold cyan]")
    logger.info("Domain: [bold]%s[/bold]", domain)
    logger.info("Loaded [bold]%d[/bold] learning targets", len(learning_targets))
    for target in learning_targets:
        logger.info("  [cyan]%s[/cyan] — %s", target["id"], target["text"])

    logger.info("Tutor: [bold blue]%s[/bold blue]", tutor_turn)
    logger.info("Student: [bold green]%s[/bold green]", student_turn)
    logger.info("Selected [bold green]%d[/bold green] evidence item(s)", len(selected))
    for item in selected:
        logger.info(
            "  [green]✓ %s[/green] — %s, %s, confidence %.2f",
            item["target"],
            item["direction"],
            item["strength"],
            item["confidence"],
        )
    for target in rejected:
        logger.info(
            "  [dim]✗ %s — no direct evidence returned[/dim]",
            target["id"],
        )

    logger.info("[bold cyan]Knowledge state: initial → final[/bold cyan]")
    for target in learning_targets:
        target_id = target["id"]
        initial = report["initial_state"]["knowledge"][target_id]["mastery"]
        final = report["final_state"]["knowledge"][target_id]["mastery"]
        logger.info("  [green]%s[/green]: %.3f → %.3f", target_id, initial, final)

    # Show observable speech acts separately from the belief values derived
    # from them, making the evidence-to-state boundary visible in logs.
    logger.info("[bold cyan]Belief evidence[/bold cyan]")
    for item in report["assessments"]["belief"]["evidence"]:
        logger.info(
            "  [yellow]%s[/yellow] — %s, confidence %.2f; quote=%r",
            item["speech_act"],
            item["strength"],
            item["confidence"],
            item["quote"],
        )
    logger.info("[bold cyan]Belief derived from evidence[/bold cyan]")
    for name, value in report["final_state"]["belief"].items():
        logger.info("  [green]%s[/green]: %.2f", name, value)

    logger.info("[bold cyan]Tutor supervision[/bold cyan]")
    for dimension, value in report["assessments"]["supervision"].items():
        if isinstance(value, dict) and "fit" in value:
            logger.info("  [blue]%s[/blue]: %.2f — %s", dimension, value["fit"], value["reason"])


if __name__ == "__main__":
    import argparse

    # Configure readable diagnostic output for direct module execution.
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Collect one exchange plus the curriculum scope needed by the knowledge
    # assessor; optional class selectors reproduce teacher-specific setups.
    parser = argparse.ArgumentParser(
        description="Run all tutor assessors over one tutor/student exchange.",
    )
    parser.add_argument("--tutor-turn", required=True, help="Tutor question or instruction.")
    parser.add_argument("--student-turn", required=True, help="Student response to assess.")
    parser.add_argument("--domain", required=True, help="Curriculum domain whose targets should be assessed.")
    parser.add_argument("--teacher", help="Teacher username for class-specific targets.")
    parser.add_argument("--class-name", help="Class name for class-specific targets.")
    parser.add_argument("--data-path", type=Path, help="Directory containing the AIDu data.db file.")
    args = parser.parse_args()

    # Run both evidence extractors over the same exchange.
    targets = learning_targets_for_domain(
        args.domain,
        teacher=args.teacher,
        class_name=args.class_name,
        data_path=args.data_path,
    )
    report = smoke_test(
        tutor_turn=args.tutor_turn,
        student_turn=args.student_turn,
        learning_targets=targets,
    )
    log_smoke_test_report(
        domain=args.domain,
        tutor_turn=args.tutor_turn,
        student_turn=args.student_turn,
        learning_targets=targets,
        report=report,
    )
    console.rule("[bold cyan]Structured result")
    console.print_json(data=report)
