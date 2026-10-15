"""Command line, handy for bulk indexing and quick tests without the UI.

  python cli.py index --youtube "https://youtu.be/XXXXXXXXXXX" --minutes 12 --title "Gate camera"
  python cli.py index --file data/uploads/lecture.mp4
  python cli.py search "red bike kab aayi" [--video yt_XXXXXXXXXXX] [--no-verify]
  python cli.py list
  python cli.py sync           # make sure every saved video is in the vector store (no video re-watching)
"""
import argparse
import logging

from clipscout import agent, indexer, library
from clipscout.schemas import fmt_time
from clipscout.video import file_source, youtube_source


def main() -> None:
    p = argparse.ArgumentParser(prog="clipscout")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index")
    i.add_argument("--youtube")
    i.add_argument("--file")
    i.add_argument("--minutes", type=float, default=0.0)
    i.add_argument("--title", default="")
    i.add_argument("--force", action="store_true")

    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("--video", default=None)
    s.add_argument("--no-verify", action="store_true")

    sub.add_parser("list")
    sub.add_parser("sync")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.cmd == "index":
        if args.youtube:
            src = youtube_source(args.youtube, args.minutes, args.title)
        elif args.file:
            src = file_source(args.file, args.title, args.minutes)
        else:
            p.error("give --youtube or --file")
        done = indexer.index_video(src, progress=lambda f, m: print(f"[{f:4.0%}] {m}"), force=args.force)
        print(f"{done.video_id}: {len(done.moments)} moments")

    elif args.cmd == "search":
        r = agent.run(args.question, video_id=args.video, settings=agent.Settings(verify=not args.no_verify))
        print(f"\n{r.answer}\n")
        for c in r.clips:
            flag = "✓" if c.verified else "?" if c.verified is None else "✗"
            print(f"  {flag} {fmt_time(c.best_start)}  {c.title}  {c.reason or c.text[:100]}")
        print("\ntrace:\n  " + "\n  ".join(r.trace) + f"\n({r.latency_ms} ms)")

    elif args.cmd == "list":
        for v in library.all_videos(only_indexed=False):
            state = "ready" if v.is_indexed else f"partial ({v.progress:.0%})"
            print(f"{v.video_id:24} {fmt_time(v.duration):>8}  {len(v.moments):4} moments  {state:18} {v.title}")

    elif args.cmd == "sync":
        print(f"added {indexer.sync_vector_store()} moments to the vector store")


if __name__ == "__main__":
    main()
