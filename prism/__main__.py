"""CLI.
  python -m prism build  --items 1000000 --workers 8 --out data/
  python -m prism ask    --out data/ "Find wildfire burn scars near Malibu in 2025"
  python -m prism serve  --out data/ --port 8080
"""
import argparse
import json
import os

from . import pipeline as P
from .encoders import PrismEncoder


def main():
    ap = argparse.ArgumentParser("prism")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--items", type=int, default=200_000)
    b.add_argument("--workers", type=int, default=os.cpu_count()); b.add_argument("--shards", type=int, default=4)
    b.add_argument("--batch", type=int, default=1024)
    for p in (b, a := sub.add_parser("ask"), s := sub.add_parser("serve")):
        p.add_argument("--out", default="data")
    for p in (a, s):
        p.add_argument("--today", help="override the planner's current date (YYYY-MM-DD); default: newest item in the index")
    a.add_argument("question"); s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 to expose; set --token)")
    s.add_argument("--token", default=os.environ.get("PRISM_TOKEN"), help="bearer token required for POST endpoints")
    args = ap.parse_args()
    model, stage, index = f"{args.out}/model.npz", f"{args.out}/stage", f"{args.out}/index"

    if args.cmd == "build":
        os.makedirs(args.out, exist_ok=True)
        if not os.path.exists(model):
            print("fitting modality encoders + alignment heads"); PrismEncoder().fit().save(model)
        descs = P.plan(args.items, args.batch)
        print(json.dumps(P.ingest(descs, model, stage, workers=args.workers), indent=1))
        P.build(stage, index, n_shards=args.shards); P.save_plan(index, descs)
        return
    from .engine import Engine
    eng = Engine(index, model, today=getattr(args, "today", None))
    if args.cmd == "ask":
        from .harness import Harness
        from .planners import default_planner
        r = default_planner().run(Harness(eng, trace_dir=f"{args.out}/traces"), args.question)
        print(r["answer"]); print(f"\n[grounded={r['grounded']} steps={r['steps']} {r['total_ms']}ms trace={r['trace']}]")
    elif args.cmd == "serve":
        from .server import serve
        serve(eng, host=args.host, port=args.port, trace_dir=f"{args.out}/traces", token=args.token)
    else:
        raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
