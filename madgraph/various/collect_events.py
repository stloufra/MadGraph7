#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect events from multiple LHE files, randomize them, and write a new LHE file.

Behavior:
  - Events are copied verbatim. No weight manipulation is performed.
  - The output header is built from a separate template file.
  - The <initrwgt> and <scalesfunctionalform> blocks are extracted from the
    first input file only. They are taken from the region between the
    <LesHouchesEvents ...> opening tag and the <header> block, and then placed
    inside the final output <header> block.
  - The <init> block is taken directly from the first input event file.

Notes on the header template:
  - The template may be a full LHE file, a <header>...</header> block, or a raw
    header fragment.
  - Any surrounding <LesHouchesEvents> wrapper and any <init> block in the
    template are stripped before insertion into the output file.
  - Lines containing only <![CDATA[ or only ]]> are removed.
  - The line containing the iseed setting is rewritten to match the requested
    seed. If no seed is provided, 0 is written.

This module can be used in two ways:
  1. Imported and called via collect_events(...)
  2. Run as a CLI script
"""

from __future__ import annotations

import argparse
import mmap
import random
import re
import sys
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union


PATT_LHE_OPEN = re.compile(rb"<\s*LesHouchesEvents\b[^>]*>", re.I)
PATT_LHE_CLOSE = re.compile(rb"</\s*LesHouchesEvents\s*>", re.I)
PATT_HEADER_OPEN = re.compile(rb"<\s*header\b[^>]*>", re.I)
PATT_HEADER_FULL = re.compile(rb"<\s*header\b[^>]*>(.*?)</\s*header\s*>", re.I | re.S)
PATT_INIT_OPEN = re.compile(rb"<\s*init\b[^>]*>", re.I)
PATT_INIT_CLOSE = re.compile(rb"</\s*init\s*>", re.I)
PATT_EVENT_OPEN = re.compile(rb"<\s*event\b", re.I)
PATT_EVENT_CLOSE = re.compile(rb"</\s*event\s*>", re.I)
PATT_XML_DECL = re.compile(rb"^\s*<\?xml[^>]*>\s*", re.I | re.S)
PATT_CDATA_LINE = re.compile(rb"(?m)^\s*(<!\[CDATA\[|\]\]>)\s*(?:\r?\n)?")
PATT_ISEED_LINE = re.compile(rb"(?m)^(?P<indent>\s*).*=\s*iseed\b.*(?:\r?\n)?")


PathLike = Union[str, Path]


@dataclass(frozen=True)
class EventRef:
    file_idx: int
    start: int
    end: int


@dataclass
class IndexedLHE:
    path: Path
    open_tag: bytes
    special_header_blocks: bytes
    init_block: bytes
    events: List[EventRef]


# ============================
# Generic helpers
# ============================

def _ensure_trailing_newline(blob: bytes) -> bytes:
    return blob if blob.endswith(b"\n") else blob + b"\n"


def _indent_bytes(blob: bytes, prefix: bytes = b"  ") -> bytes:
    lines = blob.splitlines(keepends=True)
    if not lines:
        return b""
    return b"".join(prefix + line if line.strip() else line for line in lines)


def _extract_open_tag(blob: bytes) -> Optional[bytes]:
    m = PATT_LHE_OPEN.search(blob)
    if not m:
        return None
    return _ensure_trailing_newline(m.group(0))


def _extract_header_inner(blob: bytes) -> bytes:
    m = PATT_HEADER_FULL.search(blob)
    if not m:
        return b""
    inner = m.group(1).strip()
    return _ensure_trailing_newline(inner) if inner else b""


def _extract_tag_blocks(blob: bytes, tag: str) -> List[bytes]:
    tag_re = re.escape(tag.encode("utf-8"))
    full_re = re.compile(rb"<\s*" + tag_re + rb"\b[^>]*>.*?</\s*" + tag_re + rb"\s*>", re.I | re.S)
    self_re = re.compile(rb"<\s*" + tag_re + rb"\b[^>]*/\s*>", re.I)
    out = full_re.findall(blob)
    out.extend(self_re.findall(blob))
    return [_ensure_trailing_newline(x.strip()) for x in out if x.strip()]


def _extract_special_header_blocks(blob: bytes) -> bytes:
    """
    Extract only the special blocks that live between the opening
    <LesHouchesEvents ...> tag and the <header> block.
    """
    m_open = PATT_LHE_OPEN.search(blob)
    if not m_open:
        return b""

    m_header = PATT_HEADER_OPEN.search(blob, m_open.end())
    if not m_header:
        return b""

    region = bytes(blob[m_open.end():m_header.start()])
    pieces: List[bytes] = []
    for tag in ("initrwgt", "scalesfunctionalform", "MonteCarloMasses"):
        pieces.extend(_extract_tag_blocks(region, tag))
    return b"".join(pieces)


def _rewrite_iseed_line(blob: bytes, seed: Optional[int]) -> bytes:
    out_seed = 0 if seed is None else seed

    def repl(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return indent + f"{out_seed}    = iseed       ! rnd seed (0=assigned automatically=default))\n".encode("utf-8")

    return PATT_ISEED_LINE.sub(repl, blob)


def _sanitize_header_template(blob: bytes, seed: Optional[int]) -> Tuple[Optional[bytes], bytes]:
    """
    Accepts either a full LHE file, a <header>...</header> block, or a raw header
    fragment. Returns:
      - optional <LesHouchesEvents ...> opening tag from the template
      - the header *inner* content to place inside the output <header>
    """
    open_tag = _extract_open_tag(blob)

    header_inner = _extract_header_inner(blob)
    if not header_inner:
        cleaned = blob
        cleaned = PATT_XML_DECL.sub(b"", cleaned, count=1)
        cleaned = PATT_LHE_OPEN.sub(b"", cleaned, count=1)
        cleaned = PATT_LHE_CLOSE.sub(b"", cleaned, count=1)
        cleaned = re.sub(rb"<\s*init\b[^>]*>.*?</\s*init\s*>", b"", cleaned, count=1, flags=re.I | re.S)
        cleaned = re.sub(rb"<\s*header\b[^>]*>", b"", cleaned, count=1, flags=re.I)
        cleaned = re.sub(rb"</\s*header\s*>", b"", cleaned, count=1, flags=re.I)
        header_inner = cleaned

    header_inner = PATT_CDATA_LINE.sub(b"", header_inner)
    header_inner = _rewrite_iseed_line(header_inner, seed)
    header_inner = header_inner.strip()
    return open_tag, (_ensure_trailing_newline(header_inner) if header_inner else b"")


# ============================
# LHE indexing
# ============================

def index_lhe_file(path: Path, file_idx: int) -> IndexedLHE:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            m_lhe = PATT_LHE_OPEN.search(mm)
            if not m_lhe:
                raise RuntimeError(f"{path} is missing <LesHouchesEvents>")
            open_tag = _ensure_trailing_newline(m_lhe.group(0))

            special_header_blocks = _extract_special_header_blocks(mm)

            m_init_open = PATT_INIT_OPEN.search(mm)
            m_init_close = PATT_INIT_CLOSE.search(mm)
            if not m_init_open or not m_init_close or m_init_close.start() < m_init_open.start():
                raise RuntimeError(f"{path} is missing a valid <init>...</init> block")
            init_block = bytes(mm[m_init_open.start():m_init_close.end()])
            init_block = _ensure_trailing_newline(init_block)

            size = mm.size()
            pos = m_init_close.end()
            events: List[EventRef] = []
            while True:
                mo = PATT_EVENT_OPEN.search(mm, pos)
                if not mo:
                    break
                s = mo.start()
                mc = PATT_EVENT_CLOSE.search(mm, s)
                if not mc:
                    raise RuntimeError(f"{path} has an <event> block without a closing </event>")
                e = mc.end()
                if e < size and mm[e:e + 1] == b"\n":
                    e += 1
                events.append(EventRef(file_idx=file_idx, start=s, end=e))
                pos = e
        finally:
            mm.close()

    return IndexedLHE(
        path=path,
        open_tag=open_tag,
        special_header_blocks=special_header_blocks,
        init_block=init_block,
        events=events,
    )


# ============================
# Header construction
# ============================

def build_output_header(
    header_template: Path,
    first_indexed_file: IndexedLHE,
    seed: Optional[int],
) -> Tuple[bytes, bytes]:
    template_bytes = header_template.read_bytes()
    template_open_tag, template_header_inner = _sanitize_header_template(template_bytes, seed)

    open_tag = template_open_tag or first_indexed_file.open_tag or b'<LesHouchesEvents version="3.0">\n'

    header_parts: List[bytes] = []
    if template_header_inner:
        header_parts.append(_ensure_trailing_newline(template_header_inner))
    if first_indexed_file.special_header_blocks:
        header_parts.append(_ensure_trailing_newline(first_indexed_file.special_header_blocks))

    if header_parts:
        header_inner = b"".join(header_parts)
        header_block = b"<header>\n" + _indent_bytes(header_inner, prefix=b"  ") + b"</header>\n"
    else:
        header_block = b"<header/>\n"

    return open_tag, header_block


# ============================
# Event writing
# ============================

def _split_chunks(arr: List[EventRef], k: int) -> List[List[EventRef]]:
    k = max(1, k)
    n = len(arr)
    base = n // k
    rem = n % k
    chunks: List[List[EventRef]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        if size > 0:
            chunks.append(arr[start:end])
        start = end
    return chunks


def _copy_event_refs(input_paths: Sequence[str], refs: Sequence[EventRef], fout) -> None:
    handles = {}
    mmaps = {}
    try:
        for ref in refs:
            if ref.file_idx not in mmaps:
                fh = open(input_paths[ref.file_idx], "rb")
                handles[ref.file_idx] = fh
                mmaps[ref.file_idx] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            fout.write(mmaps[ref.file_idx][ref.start:ref.end])
    finally:
        for mm in mmaps.values():
            mm.close()
        for fh in handles.values():
            fh.close()


def _write_part(input_paths: Sequence[str], refs: Sequence[EventRef], part_path: str) -> None:
    with open(part_path, "wb") as fout:
        _copy_event_refs(input_paths, refs, fout)


def write_randomized_events(
    input_paths: Sequence[Path],
    refs: List[EventRef],
    output_path: Path,
    open_tag: bytes,
    header_block: bytes,
    init_block: bytes,
    seed: Optional[int],
    subset: Optional[int],
    workers: int,
) -> None:
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if subset is not None and subset < len(shuffled):
        shuffled = shuffled[:subset]

    str_paths = [str(p) for p in input_paths]

    if workers <= 1 or len(shuffled) == 0:
        with open(output_path, "wb") as fout:
            fout.write(open_tag)
            fout.write(header_block)
            fout.write(init_block)
            _copy_event_refs(str_paths, shuffled, fout)
            fout.write(b"</LesHouchesEvents>\n")
        return

    chunks = _split_chunks(shuffled, workers)
    part_paths = [output_path.with_suffix(output_path.suffix + f".part{i}") for i in range(len(chunks))]

    procs: List[Process] = []
    for chunk, part in zip(chunks, part_paths):
        p = Process(target=_write_part, args=(str_paths, chunk, str(part)))
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("A worker process failed while writing event chunks.")

    try:
        with open(output_path, "wb") as fout:
            fout.write(open_tag)
            fout.write(header_block)
            fout.write(init_block)
            for part in part_paths:
                with open(part, "rb") as fp:
                    while True:
                        buf = fp.read(1024 * 1024)
                        if not buf:
                            break
                        fout.write(buf)
            fout.write(b"</LesHouchesEvents>\n")
    finally:
        for part in part_paths:
            try:
                part.unlink()
            except Exception:
                pass


# ============================
# Public API
# ============================

def collect_events(
    output: PathLike,
    header_template: PathLike,
    input_files: Sequence[PathLike],
    seed: Optional[int] = None,
    subset: Optional[int] = None,
    workers: int = 1,
    verbose: bool = False,
) -> Path:
    """
    Collect LHE events from multiple files, shuffle them, and write a new LHE file.

    Parameters
    ----------
    output:
        Path to the output LHE file.
    header_template:
        Path to the separate template used to build the final <header> block.
    input_files:
        Sequence of input LHE files.
    seed:
        Shuffle seed. Also used to rewrite the iseed line in the template.
        If None, the template iseed line is rewritten to 0.
    subset:
        If given, keep only this many events after shuffling.
    workers:
        Number of parallel worker processes for event copying. Must be >= 1.
    verbose:
        If True, print progress messages.

    Returns
    -------
    Path
        The resolved path of the written output file.
    """
    output_path = Path(output).resolve()
    header_template_path = Path(header_template).resolve()
    input_paths = [Path(x).resolve() for x in input_files]
    workers = max(1, workers)

    if not header_template_path.is_file():
        raise RuntimeError(f"Header template file not found: {header_template_path}")
    if not input_paths:
        raise RuntimeError("No input files provided.")
    for path in input_paths:
        if not path.is_file():
            raise RuntimeError(f"Input file not found: {path}")

    if verbose:
        print(f"[1/3] Indexing {len(input_paths)} input file(s) ...")

    indexed_files: List[IndexedLHE] = []
    all_refs: List[EventRef] = []
    for i, path in enumerate(input_paths):
        item = index_lhe_file(path, i)
        indexed_files.append(item)
        all_refs.extend(item.events)
        if verbose:
            print(f"      {path.name}: {len(item.events)} event(s)")

    if verbose:
        print("[2/3] Building output header and init block ...")

    open_tag, header_block = build_output_header(
        header_template=header_template_path,
        first_indexed_file=indexed_files[0],
        seed=seed,
    )
    init_block = indexed_files[0].init_block

    n_target = len(all_refs) if subset is None else min(subset, len(all_refs))
    if verbose:
        print(f"[3/3] Writing {n_target} shuffled event(s) -> {output_path} (workers={workers}, seed={seed})")

    write_randomized_events(
        input_paths=input_paths,
        refs=all_refs,
        output_path=output_path,
        open_tag=open_tag,
        header_block=header_block,
        init_block=init_block,
        seed=seed,
        subset=subset,
        workers=workers,
    )

    if verbose:
        print("Done.")

    return output_path


# ============================
# CLI
# ============================

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect LHE events from multiple files, shuffle them, and write a new LHE file. Events are copied verbatim."
    )
    ap.add_argument("-o", "--output", required=True, help="Output LHE file.")
    ap.add_argument("--header-template", required=True, help="Separate file used to build the output header.")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed.")
    ap.add_argument("--subset", type=int, default=None, help="Keep only this many events after shuffling.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel writer processes (>=1).")
    ap.add_argument("inputs", nargs="+", help="Input LHE file(s).")
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = parse_args(argv)
    collect_events(
        output=ns.output,
        header_template=ns.header_template,
        input_files=ns.inputs,
        seed=ns.seed,
        subset=ns.subset,
        workers=ns.workers,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
