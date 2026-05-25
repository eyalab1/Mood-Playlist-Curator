"""Inspect agent traces from the SQLite database.

Examples:
    python inspect_traces.py curator           # latest curator trace
    python inspect_traces.py mood_interpreter  # latest mood_interpreter trace
    python inspect_traces.py curator --list    # list all curator traces
    python inspect_traces.py --id 12           # specific trace by id
"""

import argparse
import json

from db import get_conn


def _print_row(row) -> None:
    print(f"--- trace #{row['id']} ---")
    print(f"agent:      {row['agent_name']}")
    print(f"created_at: {row['created_at']}")
    print(f"tokens:     in={row['tokens_in']}  out={row['tokens_out']}")
    print(f"latency:    {row['latency_ms']} ms")
    print(f"input_hash: {row['input_hash'][:16]}...")
    print()
    print("INPUT:")
    try:
        print(json.dumps(json.loads(row["input_json"]), indent=2, ensure_ascii=False))
    except Exception:
        print(row["input_json"])
    print()
    print("OUTPUT:")
    if row["output_json"] is None:
        print("(none -- agent failed all retries)")
    else:
        try:
            print(json.dumps(json.loads(row["output_json"]), indent=2, ensure_ascii=False))
        except Exception:
            print(row["output_json"])


def _list_rows(agent: str | None) -> None:
    sql = "SELECT id, agent_name, created_at, tokens_in, tokens_out, latency_ms FROM agent_traces"
    params: tuple = ()
    if agent:
        sql += " WHERE agent_name = ?"
        params = (agent,)
    sql += " ORDER BY id DESC LIMIT 50"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No traces found.")
        return
    print(f"{'id':>5}  {'agent':<20}  {'created_at':<20}  {'tok_in':>7}  {'tok_out':>7}  {'ms':>6}")
    for r in rows:
        print(
            f"{r['id']:>5}  {r['agent_name']:<20}  {r['created_at']:<20}  "
            f"{r['tokens_in'] or 0:>7}  {r['tokens_out'] or 0:>7}  {r['latency_ms'] or 0:>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect agent_traces rows.")
    parser.add_argument(
        "agent",
        nargs="?",
        help="Agent name (mood_interpreter, curator, critic, sequencer).",
    )
    parser.add_argument("--list", action="store_true", help="List traces instead of printing one.")
    parser.add_argument("--id", type=int, help="Show a specific trace id.")
    args = parser.parse_args()

    if args.list:
        _list_rows(args.agent)
        return

    with get_conn() as conn:
        if args.id is not None:
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE id = ?", (args.id,)
            ).fetchone()
        elif args.agent:
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE agent_name = ? "
                "ORDER BY id DESC LIMIT 1",
                (args.agent,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM agent_traces ORDER BY id DESC LIMIT 1"
            ).fetchone()

    if row is None:
        print("No matching trace found.")
        return
    _print_row(row)


if __name__ == "__main__":
    main()
