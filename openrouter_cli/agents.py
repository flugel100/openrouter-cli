"""Agent system: spesialisasi tugas multi-langkah dengan tool registry.

Agent mengikuti workflow terstruktur (plan → execute → verify → synthesize)
dan menggunakan model + tool untuk menyelesaikan tugas kompleks.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from .backends import Backend
from .tools import ToolRegistry, build_default_registry


# --------------------------------------------------------------------------- #
# Agent report
# --------------------------------------------------------------------------- #

@dataclass
class AgentFinding:
    """Satu temuan dari agent review."""

    id: str
    location: str
    claim: str
    severity: str  # P0, P1, P2, P3, P4
    evidence: str
    verdict: str = "CONFIRMED"  # CONFIRMED, PLAUSIBLE, REFUTED
    fix: str = ""


@dataclass
class AgentReport:
    """Hasil akhir dari agent run."""

    task: str
    scope: str
    findings: list[AgentFinding] = field(default_factory=list)
    refuted: list[AgentFinding] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    token_estimate: int = 0
    status: str = "done"  # done, capped, failed

    def summary(self) -> str:
        by_sev: dict[str, int] = {}
        for f in self.findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        parts = [f"Agent Ultracode — {self.status.upper()}"]
        parts.append(f"  Temuan: {len(self.findings)} dikonfirmasi, {len(self.refuted)} tertolak")
        for s in ["P0", "P1", "P2", "P3", "P4"]:
            if by_sev.get(s):
                parts.append(f"  {s}: {by_sev[s]}")
        if self.missed:
            parts.append(f"  Terlewat: {len(self.missed)} area")
        parts.append(f"  Estimasi token: ~{self.token_estimate}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Base Agent
# --------------------------------------------------------------------------- #

class Agent:
    """Agent otonom dengan akses tool + model."""

    name: str = "base"

    def __init__(
        self,
        backend: Backend,
        tools: Optional[ToolRegistry] = None,
        *,
        max_turns: int = 6,
        verbose: bool = False,
    ):
        self.backend = backend
        self.tools = tools or build_default_registry()
        self.max_turns = max_turns
        self.verbose = verbose

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        """Panggil model, kembalikan teks respons."""
        resp = self.backend.complete(messages)
        content = resp.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return str(content)

    def _step_with_tools(self, messages: list[dict[str, Any]]) -> str:
        """Satu putaran chat + eksekusi tool calls (jika ada)."""
        resp = self.backend.complete(messages, tools=self.tools.to_openai_specs())
        msg = dict(resp)
        messages.append(msg)

        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                tool_id = call.get("id", "")
                result = self.tools.execute_call(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })
            # Lanjutkan setelah hasil tool
            final = self.backend.complete(messages, tools=self.tools.to_openai_specs())
            messages.append(dict(final))
            return str(final.get("content", ""))
        return str(msg.get("content", ""))

    def run(self, task: str, scope: str = "") -> AgentReport:
        """Override di subclass. Kembalikan laporan."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Ultracode Agent — Code Review Exhaustive
# --------------------------------------------------------------------------- #

ULTRA_FINDER_PROMPT = """\
Anda adalah FINDER dalam workflow ultracode. Temukan SEMUA kandidat isu TANPA menilai.

TUGAS: {task}
SCOPE: {scope}

Pindai scope secara menyeluruh. Gunakan tool read_file, run_command, dan web_fetch
untuk memeriksa kode sumber, test, config, dan dokumentasi. Temukan SETIAP kandidat isu,
sekecil apapun. JANGAN filter, JANGAN verifikasi, JANGAN usulkan perbaikan.

Untuk setiap kandidat, laporkan:
- id (F001, F002, ...)
- lokasi (file:baris)
- klaim (satu kalimat)
- perkiraan severity (P0-P4)
- bukti (apa yang benar-benar Anda lihat di kode)
- ketidakpastian (tinggi/rendah, dan kenapa)

Cakup minimal: kebenaran logika, edge case, keamanan, konkurensi/urutan state,
kontrak antar-modul, perbedaan build/release, dan silent failure.
Jika ragu apakah sesuatu masalah, sertakan dengan catatan.

Laporan akhir harus dalam format JSON dengan struktur:
{{"findings": [{{"id": "F001", "location": "...", "claim": "...", "severity": "P2", "evidence": "...", "uncertainty": "low"}}], "unverified": ["area yang tidak bisa dicek"], "scope_touched": ["file yang dibaca"]}}
"""

ULTRA_VERIFIER_PROMPT = """\
Anda adalah ADVERSARIAL VERIFIER dalam workflow ultracode. Tugas Anda adalah MEMBANTAH
temuan Finder, bukan mengonfirmasi. Bias konfirmasi adalah musuh Anda.

TEMUAN:
{findings_json}

TUGAS KONTEKS: {task}

Untuk SETIAP temuan:
1. Baca ulang kode di lokasi persis (pakai tool read_file).
2. Lacak setiap jalur yang diklaim, termasuk pemanggil dan guard.
3. Cek secara spesifik: apakah klaim didasarkan pada asumsi yang salah?
   Apakah baris itu dead code? Apakah guard di atas sudah menanganinya?
   Apakah perilaku itu memang disengaja?
4. Berburu yang terlewat dari sisi penyerang: boundary values, empty/null/zero,
   item pertama/terakhir, state offline/failure, konkurensi/race, izin, silent failure.

Untuk SETIAP temuan, beri VERDIK:
- CONFIRMED: nyatakan pemicu minimal (input/state → hasil salah)
- PLAUSIBLE: tidak bisa dibantah, tapi tidak ada bukti langsung
- REFUTED: alasan, tunjuk kode yang membantahnya
Plus severity: P0/P1/P2/P3/P4

Laporan akhir harus dalam format JSON dengan struktur:
{{"verifications": [{{"id": "F001", "verdict": "CONFIRMED", "severity": "P2", "trigger": "...", "counter_case": "...", "evidence_path": "..."}}], "missed": ["hal baru yang ditemukan saat verifikasi"]}}
"""

ULTRA_SYNTHESIZER_PROMPT = """\
Anda adalah SYNTHESIZER dalam workflow ultracode. Finder sudah menemukan isu,
Verifier sudah memeriksa. Anda TIDAK memverifikasi apapun — Anda hanya menyusun hasil.

TUGAS: {task}
HASIL VERIFIKASI: {verified_json}
YANG TERTOLAK: {refuted_ids}

Tugas Anda:
1. Buang semua temuan yang diverdict REFUTED oleh mayoritas verifier.
2. Simpan CONFIRMED dan PLAUSIBLE; gabung duplikat.
3. Urutkan berdasarkan severity (P0 > P1 > P2 > P3 > P4), lalu blast radius.
4. Untuk setiap P0/P1, tulis rencana perbaikan dan pemicu minimal sebagai tes.
5. Hasilkan laporan akhir:
   - Daftar aksi terurut severity. Tiap isu: file:baris persis, kenapa penting,
     dan perbaikan konkret (diff, tanda tangan fungsi, atau keputusan). Jangan saran kosong.
   - Pertanyaan terbuka — hal yang benar-benar butuh keputusan manusia.
   - Pernyataan cakupan: apa yang diverifikasi, apa yang TIDAK, dan risiko residual.

Aturan:
- JANGAN verifikasi ulang. JANGAN tambah temuan baru. JANGAN buka kembali temuan.
- Frasa TERLARANG: "seharusnya aman", "mungkin berfungsi".
- Tanpa padding, tanpa pengulangan; setiap item bisa ditindaklanjuti.

Laporan akhir dalam teks bebas (Markdown), dimulai dengan ringkasan temuan berdasarkan severity.
"""


class UltracodeAgent(Agent):
    """Agent audit kode exhaustive — metodologi find→verify→synthesize.

    Mode: MANUAL (satu agen berganti peran berurutan).
    Tier: Lite / Medium / Deep (loop).
    """

    name = "ultracode"

    def __init__(
        self,
        backend: Backend,
        tools: Optional[ToolRegistry] = None,
        *,
        tier: str = "medium",
        max_turns: int = 12,
        verbose: bool = False,
    ):
        super().__init__(backend, tools, max_turns=max_turns, verbose=verbose)
        self.tier = tier  # lite, medium, deep
        self._scratch: dict[str, Any] = {}
        self._total_tokens = 0

    def run(self, task: str, scope: str = ".") -> AgentReport:
        """Jalankan workflow ultracode lengkap."""

        # Stage 0 — PLAN
        self._log(f"[ULTRA] Memulai audit tier={self.tier}")
        self._log(f"         Tugas: {task}")
        self._log(f"         Scope: {scope}")
        report = AgentReport(task=task, scope=scope)

        max_loops = {"lite": 1, "medium": 2, "deep": 3}.get(self.tier, 2)

        for loop in range(max_loops):
            self._log(f"\n[ULTRA] Loop {loop + 1}/{max_loops} — FINDER pass")

            # Stage 1 — FINDER
            findings_raw = self._run_finder(task, scope)
            if not findings_raw:
                self._log("[ULTRA] Finder tidak menemukan kandidat isu.")
                if loop > 0:
                    break
                report.status = "done"
                report.token_estimate = self._total_tokens
                return report

            self._log(f"[ULTRA] Finder: {len(findings_raw)} kandidat ditemukan")

            # Stage 2 — VERIFIER
            self._log("[ULTRA] VERIFIER pass — adversarial check")
            verified = self._run_verifier(task, findings_raw)
            confirmed = [f for f in verified if f["verdict"] == "CONFIRMED"]
            plausible = [f for f in verified if f["verdict"] == "PLAUSIBLE"]
            refuted = [f for f in verified if f["verdict"] == "REFUTED"]

            self._log(
                f"[ULTRA] Verifier: {len(confirmed)} dikonfirmasi, "
                f"{len(plausible)} plausible, {len(refuted)} tertolak"
            )

            # Simpan temuan yang lolos verifikasi
            for f in confirmed + plausible:
                report.findings.append(AgentFinding(
                    id=f["id"],
                    location=f.get("location", ""),
                    claim=f.get("claim", ""),
                    severity=f.get("severity", "P3"),
                    evidence=f.get("evidence_path", ""),
                    verdict=f["verdict"],
                ))
            for f in refuted:
                report.refuted.append(AgentFinding(
                    id=f["id"],
                    location="",
                    claim=f.get("reason", ""),
                    severity="P4",
                    evidence="",
                    verdict="REFUTED",
                ))

            # Jika setelah loop ini nol temuan baru, kita "dry"
            if loop > 0 and len(confirmed) == 0 and len(plausible) == 0:
                self._log("[ULTRA] Loop kering — tidak ada temuan baru.")
                break

        # Stage 3 — SYNTHESIZER
        if report.findings:
            self._log("[ULTRA] SYNTHESIZER pass — menyusun laporan akhir")
            synthesis = self._run_synthesizer(task, report.findings, report.refuted)
            report.missed = synthesis.get("missed", [])
        else:
            self._log("[ULTRA] Tidak ada temuan yang perlu disintesis.")

        report.token_estimate = self._total_tokens
        report.status = "done" if report.findings else "clean"
        self._log(f"\n[ULTRA] Selesai. {len(report.findings)} temuan total.")
        return report

    # --- Internal stages ---

    def _run_finder(self, task: str, scope: str) -> list[dict[str, Any]]:
        prompt = ULTRA_FINDER_PROMPT.format(task=task, scope=scope)
        messages = [
            {"role": "system", "content": "Anda adalah FINDER agent ultracode. Gunakan tool read_file dan run_command untuk memeriksa kode. Output HARUS JSON valid."},
            {"role": "user", "content": prompt},
        ]
        raw = self._step_with_tools(messages)
        self._total_tokens += len(raw) // 4  # estimasi kasar
        try:
            data = json.loads(self._extract_json(raw))
            return data.get("findings", [])
        except (json.JSONDecodeError, KeyError):
            self._log("[ULTRA] Gagal parse JSON finder — mencoba recovery")
            return []

    def _run_verifier(self, task: str, findings: list[dict]) -> list[dict[str, Any]]:
        # Bagi temuan menjadi batch agar tidak kelebihan konteks
        batch_size = 5
        all_verified: list[dict[str, Any]] = []

        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]
            prompt = ULTRA_VERIFIER_PROMPT.format(
                task=task,
                findings_json=json.dumps(batch, ensure_ascii=False, indent=2),
            )
            messages = [
                {"role": "system", "content": "Anda adalah ADVERSARIAL VERIFIER ultracode. Bersikap skeptis. Output HARUS JSON valid."},
                {"role": "user", "content": prompt},
            ]
            raw = self._step_with_tools(messages)
            self._total_tokens += len(raw) // 4
            try:
                data = json.loads(self._extract_json(raw))
                all_verified.extend(data.get("verifications", []))
            except (json.JSONDecodeError, KeyError):
                self._log("[ULTRA] Gagal parse JSON verifier batch — melanjutkan")

        return all_verified

    def _run_synthesizer(
        self,
        task: str,
        findings: list[AgentFinding],
        refuted: list[AgentFinding],
    ) -> dict[str, Any]:
        confirmed_json = json.dumps(
            [{"id": f.id, "location": f.location, "claim": f.claim, "severity": f.severity, "verdict": f.verdict} for f in findings],
            ensure_ascii=False, indent=2,
        )
        refuted_ids = json.dumps([f.id for f in refuted], ensure_ascii=False)

        prompt = ULTRA_SYNTHESIZER_PROMPT.format(
            task=task,
            verified_json=confirmed_json,
            refuted_ids=refuted_ids,
        )
        messages = [
            {"role": "system", "content": "Anda adalah SYNTHESIZER ultracode. Susun laporan akhir. Output teks Markdown."},
            {"role": "user", "content": prompt},
        ]
        raw = self._chat(messages)
        self._total_tokens += len(raw) // 4
        return {"synthesis_md": raw, "missed": []}

    def _extract_json(self, text: str) -> str:
        """Ekstrak blok JSON dari teks (mungkin dibungkus markdown)."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # Coba temukan { pertama dan } terakhir
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            return text[first:last + 1]
        return text

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def run_ultracode(
    backend: Backend,
    task: str,
    scope: str = ".",
    *,
    tier: str = "medium",
    verbose: bool = False,
) -> AgentReport:
    """Jalankan agent ultracode dan kembalikan laporan."""
    agent = UltracodeAgent(backend, tier=tier, verbose=verbose)
    return agent.run(task, scope)


# --------------------------------------------------------------------------- #
# CoderAgent — coding assistant seperti OpenCode / Claude Code
# --------------------------------------------------------------------------- #

CODER_SYSTEM_PROMPT = """Kamu adalah coding assistant otonom. Kamu bisa membaca, menulis, mengedit file, dan menjalankan perintah di terminal.

ATURAN:
1. BACA dulu — pahami struktur project sebelum menyentuh kode. Gunakan glob_files, grep_search, dan read_file.
2. TULIS/EDIT — setelah paham, gunakan write_file atau edit_file.
3. VERIFIKASI — setelah edit, jalankan test/build dengan run_command.
4. Iterasi sampai benar — kalau test gagal, baca error, perbaiki, ulangi.

TOOLS:
- read_file(path) — baca isi file
- write_file(path, content) — tulis file baru / overwrite
- edit_file(path, old_string, new_string, replace_all=false) — ganti teks persis
- glob_files(pattern) — cari file (contoh: **/*.py)
- grep_search(pattern, include) — cari teks di file
- run_command(command) — jalankan perintah shell
- web_fetch(url) — ambil konten web

OUTPUT: selalu jelaskan apa yang kamu lakukan, kenapa, dan hasilnya.
Jika buntu >3x, tanya user. Jangan berasumsi."""  # noqa: E501


class CoderAgent(Agent):
    """Agent coding otonom — baca → edit → verifikasi → iterasi."""

    name = "coder"

    def __init__(
        self,
        backend: Backend,
        tools: Optional[ToolRegistry] = None,
        *,
        max_turns: int = 10,
        verbose: bool = False,
    ):
        super().__init__(backend, tools, max_turns=max_turns, verbose=verbose)

    def run(self, task: str, scope: str = ".") -> dict[str, str]:
        self._log(f"[CODER] Tugas: {task}")

        # Auto-detect project context
        context = self._detect_context()
        if context:
            self._log(f"[CODER] Konteks project: {context[:100]}...")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
        ]

        if context:
            messages.append({"role": "user", "content": f"KONTEKS PROJECT:\n{context}"})

        messages.append({"role": "user", "content": f"TUGAS: {task}"})

        output_parts: list[str] = []
        for turn in range(self.max_turns):
            self._log(f"[CODER] Turn {turn + 1}/{self.max_turns}")
            resp = self._step_with_tools(messages)
            output_parts.append(resp)

            # Cek apakah tugas selesai (model bilang selesai)
            if any(kw in resp.lower() for kw in ["selesai", "done", "berhasil", "semua ok"]):
                if turn >= 1:  # jangan berhenti di turn pertama
                    self._log("[CODER] Selesai.")
                    break

        return {
            "output": "\n\n".join(output_parts),
            "turns": str(turn + 1),
        }

    def _detect_context(self) -> str:
        """Deteksi file konteks project (CLAUDE.md, package.json, dll)."""
        import os

        context_files = [
            "CLAUDE.md", "CLAUDE.local.md", "README.md",
            "package.json", "pyproject.toml", "Cargo.toml",
            "go.mod", "Makefile", "Dockerfile",
        ]
        parts: list[str] = []
        cwd = os.getcwd()

        for fname in context_files:
            path = os.path.join(cwd, fname)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read(2000)
                    parts.append(f"=== {fname} ===\n{content}")
                except OSError:
                    pass

        # Juga list struktur direktori
        try:
            items = sorted(os.listdir(cwd))[:30]
            parts.insert(0, "STRUKTUR DIR:\n" + "\n".join(f"  {i}" for i in items))
        except OSError:
            pass

        return "\n\n".join(parts)


def run_coder(
    backend: Backend,
    task: str,
    *,
    verbose: bool = False,
) -> dict[str, str]:
    """Jalankan agent coder."""
    agent = CoderAgent(backend, verbose=verbose)
    return agent.run(task)
