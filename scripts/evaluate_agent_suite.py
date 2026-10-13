import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATORS = [
    (
        "routing",
        "scripts/evaluate_agent_routing.py",
    ),
    (
        "tool_use",
        "scripts/evaluate_agent_tool_use.py",
    ),
    (
        "answer_grounding",
        "scripts/evaluate_answer_grounding.py",
    ),
]

CASE_PATTERN = re.compile(
    r"^([A-Z]\d+) \[(PASS|FAIL|ERROR)\]",
    re.MULTILINE,
)


def run_evaluator(
    name: str,
    script_path: str,
    environment: dict[str, str],
) -> dict:
    print(f"\n===== Running {name} evaluation =====", flush=True)

    started_at = perf_counter()
    full_script_path = PROJECT_ROOT / script_path

    try:
        process = subprocess.run(
            [
                sys.executable,
                str(full_script_path),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = process.stdout
        stderr = process.stderr
        exit_code = process.returncode

    except OSError as exc:
        print(f"Could not run {script_path}: {exc}")
        stdout = ""
        stderr = str(exc)
        exit_code = None

    duration_seconds = round(
        perf_counter() - started_at,
        2,
    )

    if stdout:
        print(stdout, end="")

    if stderr:
        print(stderr, file=sys.stderr, end="")

    case_results = [
        {
            "case_id": case_id,
            "status": status,
            "passed": status == "PASS",
        }
        for case_id, status in CASE_PATTERN.findall(stdout)
    ]

    total_cases = len(case_results)
    passed_cases = sum(
        case["passed"]
        for case in case_results
    )

    pass_rate = (
        passed_cases / total_cases
        if total_cases
        else 0.0
    )

    passed = (
        exit_code == 0
        and total_cases > 0
        and passed_cases == total_cases
    )

    return {
        "evaluator": name,
        "script": script_path,
        "status": "PASS" if passed else "FAIL",
        "exit_code": exit_code,
        "duration_seconds": duration_seconds,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": round(pass_rate, 4),
        "cases": case_results,
    }


def main() -> int:
    environment = os.environ.copy()

    source_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )

    started_at = perf_counter()
    evaluations = []

    for name, script_path in EVALUATORS:
        result = run_evaluator(
            name,
            script_path,
            environment,
        )
        evaluations.append(result)

    total_cases = sum(
        evaluation["total_cases"]
        for evaluation in evaluations
    )
    passed_cases = sum(
        evaluation["passed_cases"]
        for evaluation in evaluations
    )
    failed_cases = total_cases - passed_cases

    overall_passed = all(
        evaluation["status"] == "PASS"
        for evaluation in evaluations
    )

    report = {
        "report_version": 1,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "overall_status": (
            "PASS" if overall_passed else "FAIL"
        ),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round(
            passed_cases / total_cases,
            4,
        ) if total_cases else 0.0,
        "duration_seconds": round(
            perf_counter() - started_at,
            2,
        ),
        "evaluations": evaluations,
    }

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = report_dir / "agent_evaluation_latest.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
            ensure_ascii=False,
        )
        report_file.write("\n")

    print("\n===== CONSOLIDATED EVALUATION REPORT =====")

    for evaluation in evaluations:
        print(
            f"{evaluation['evaluator']}: "
            f"{evaluation['passed_cases']}/"
            f"{evaluation['total_cases']} passed | "
            f"{evaluation['pass_rate']:.1%} | "
            f"{evaluation['duration_seconds']:.2f}s | "
            f"{evaluation['status']}"
        )

    print("------------------------------------------")
    print(f"Overall status: {report['overall_status']}")
    print(f"Cases passed: {passed_cases}/{total_cases}")
    print(f"Cases failed: {failed_cases}")
    print(f"Total duration: {report['duration_seconds']:.2f}s")
    print(f"JSON report saved to: {report_path}")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
