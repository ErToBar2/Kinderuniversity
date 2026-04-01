from __future__ import annotations
import cv2
import ctypes
import json
import os
import shutil
import subprocess
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import pygame


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

FFPLAY_CANDIDATES = (
    MODULE_DIR / "ffplay.exe",
    MODULE_DIR / "ffmpeg/bin/ffplay.exe",
    REPO_ROOT / "tools/ffmpeg/bin/ffplay.exe",
    Path(r"C:\Program Files\Shotcut\ffplay.exe"),
    Path(r"C:\Program Files\ffmpeg\bin\ffplay.exe"),
)


def _find_ffplay() -> str | None:
    for candidate in FFPLAY_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    system_ffplay = shutil.which("ffplay") or shutil.which("ffplay.exe")
    if system_ffplay:
        return system_ffplay
    return None


def get_monitor_rects() -> list[tuple[int, int, int, int]]:
    if os.name != "nt":
        return [(0, 0, 1920, 1080)]

    user32 = ctypes.windll.user32
    monitors: list[tuple[int, int, int, int]] = []

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def _callback(_monitor, _hdc, rect_ptr, _data):
        rect = rect_ptr.contents
        monitors.append(
            (
                int(rect.left),
                int(rect.top),
                int(rect.right - rect.left),
                int(rect.bottom - rect.top),
            )
        )
        return 1

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(_callback), 0)

    if not monitors:
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        monitors.append((0, 0, width, height))

    monitors.sort(key=lambda rect: (rect[1], rect[0]))
    return monitors


def get_dual_monitor_layout(primary_window: str = "game") -> dict[str, tuple[int, int, int, int]]:
    primary_window = primary_window.lower().strip()
    if primary_window not in {"game", "webcam"}:
        raise ValueError("primary_window must be 'game' or 'webcam'")

    monitors = get_monitor_rects()
    first_monitor = monitors[0]
    second_monitor = monitors[1] if len(monitors) > 1 else first_monitor

    if primary_window == "game":
        return {"game": first_monitor, "webcam": second_monitor}
    return {"game": second_monitor, "webcam": first_monitor}


def position_window_on_monitor(window_title: str, monitor_rect: tuple[int, int, int, int]) -> bool:
    if os.name != "nt":
        return False

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_title)
    if not hwnd:
        return False

    left, top, width, height = monitor_rect
    sw_restore = 9
    sw_maximize = 3
    flags = 0x0004 | 0x0200 | 0x0040

    user32.ShowWindow(hwnd, sw_restore)
    user32.SetWindowPos(hwnd, 0, int(left), int(top), int(width), int(height), flags)
    user32.ShowWindow(hwnd, sw_maximize)
    return True


def play_intro_video_on_all_monitors(
    video_path: str | Path,
    volume: int = 70,
    allow_skip: bool = False,
) -> bool:
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[intro] Missing video: {video_path}")
        return False

    ffplay = _find_ffplay()
    if ffplay is None:
        print("[intro] ffplay not found, skipping intro video.")
        return False

    processes = []
    for index, (left, top, width, height) in enumerate(get_monitor_rects()):
        command = [
            ffplay,
            "-loglevel",
            "quiet",
            "-autoexit",
            "-alwaysontop",
            "-noborder",
            "-left",
            str(left),
            "-top",
            str(top),
            "-x",
            str(width),
            "-y",
            str(height),
        ]
        if index == 0:
            command.extend(["-volume", str(volume)])
        else:
            command.append("-an")
        command.append(str(video_path))
        processes.append(subprocess.Popen(command))

    while True:
        active_processes = [process for process in processes if process.poll() is None]
        if not active_processes:
            break

        if allow_skip and pygame.get_init():
            for event in pygame.event.get((pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.JOYBUTTONDOWN)):
                if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.JOYBUTTONDOWN):
                    for process in active_processes:
                        process.terminate()
                    for process in active_processes:
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    return True

        pygame.time.wait(30)

    return True


def start_background_music(audio_path: str | Path, volume: float = 0.7) -> bool:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"[music] Missing background music: {audio_path}")
        return False

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)
        return True
    except pygame.error as exc:
        print(f"[music] Could not start background music: {exc}")
        return False


def stop_background_music() -> None:
    try:
        pygame.mixer.music.stop()
    except pygame.error:
        pass


def ensure_leaderboard_dir(directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_leaderboard_snapshot(
    surface_or_frame,
    directory: str | Path,
    prefix: str = "ritual_success",
) -> str:
    leaderboard_dir = ensure_leaderboard_dir(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = leaderboard_dir / f"{prefix}_{timestamp}.png"

    if isinstance(surface_or_frame, pygame.Surface):
        pygame.image.save(surface_or_frame, str(output_path))
    else:
        cv2.imwrite(str(output_path), surface_or_frame)

    return str(output_path)

def _leaderboard_json_path(directory: str | Path) -> Path:
    return ensure_leaderboard_dir(directory) / "leaderboard.json"


def load_leaderboard_entries(directory: str | Path) -> list[dict]:
    json_path = _leaderboard_json_path(directory)
    if not json_path.exists():
        return []

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_leaderboard_entries(directory: str | Path, entries: list[dict]) -> None:
    json_path = _leaderboard_json_path(directory)
    json_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def sort_leaderboard_entries(entries: list[dict]) -> list[dict]:
    def _entry_key(entry: dict) -> tuple[float, str]:
        return (float(entry.get("total_seconds", 999999999.0)), entry.get("completed_at", ""))

    return sorted(entries, key=_entry_key)


def get_candidate_rank(entries: list[dict], total_seconds: float, max_entries: int = 4) -> int | None:
    candidate = {
        "__candidate__": True,
        "total_seconds": float(total_seconds),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    ranked_entries = sort_leaderboard_entries(entries + [candidate])
    for index, entry in enumerate(ranked_entries, start=1):
        if entry.get("__candidate__"):
            return index if index <= max_entries else None
    return None


def format_duration(total_seconds: float | int | None) -> str:
    if total_seconds is None:
        return "--:--"

    total_seconds = max(0, int(round(float(total_seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _load_scaled_background(screen: pygame.Surface, background_image_path: str | Path | None) -> pygame.Surface:
    if background_image_path and Path(background_image_path).exists():
        background = pygame.image.load(str(background_image_path)).convert()
        return pygame.transform.smoothscale(background, screen.get_size())

    fallback = pygame.Surface(screen.get_size())
    fallback.fill((22, 14, 8))
    return fallback


def _draw_background(screen: pygame.Surface, background: pygame.Surface) -> None:
    screen.blit(background, (0, 0))
    panel = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    panel.fill((245, 238, 219, 210))
    screen.blit(panel, (0, 0))


def _load_profile_thumbnail(leaderboard_dir: Path, image_filename: str | None, size: tuple[int, int]) -> pygame.Surface | None:
    if not image_filename:
        return None

    image_path = leaderboard_dir / image_filename
    if not image_path.exists():
        return None

    try:
        image = pygame.image.load(str(image_path)).convert()
    except pygame.error:
        return None

    return pygame.transform.smoothscale(image, size)


def prompt_for_team_name(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    background_image_path: str | Path | None,
    total_seconds: float,
    rank: int,
) -> str | None:
    background = _load_scaled_background(screen, background_image_path)
    title_font = pygame.font.Font("freesansbold.ttf", 44)
    body_font = pygame.font.Font("freesansbold.ttf", 28)
    input_font = pygame.font.Font("freesansbold.ttf", 34)
    team_name = ""
    max_length = 18

    pygame.event.clear()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                return None
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                cleaned = team_name.strip()
                return cleaned if cleaned else None
            if event.key == pygame.K_BACKSPACE:
                team_name = team_name[:-1]
                continue

            if len(team_name) >= max_length:
                continue

            if event.unicode and (event.unicode.isalnum() or event.unicode in " -_"):
                team_name += event.unicode

        _draw_background(screen, background)

        panel_rect = pygame.Rect(220, 180, screen.get_width() - 440, 360)
        pygame.draw.rect(screen, (255, 250, 235), panel_rect, border_radius=28)
        pygame.draw.rect(screen, (103, 71, 39), panel_rect, width=4, border_radius=28)

        title_surface = title_font.render("Top 4! Enter your team name", True, (66, 38, 16))
        title_rect = title_surface.get_rect(center=(screen.get_width() // 2, panel_rect.y + 72))
        screen.blit(title_surface, title_rect)

        details = [
            f"Current rank: #{rank}",
            f"Time: {format_duration(total_seconds)}",
            "Press Enter to save or Esc to skip.",
        ]
        for index, line in enumerate(details):
            text_surface = body_font.render(line, True, (84, 54, 29))
            text_rect = text_surface.get_rect(center=(screen.get_width() // 2, panel_rect.y + 138 + index * 42))
            screen.blit(text_surface, text_rect)

        input_rect = pygame.Rect(panel_rect.x + 90, panel_rect.y + 232, panel_rect.width - 180, 72)
        pygame.draw.rect(screen, (255, 255, 255), input_rect, border_radius=18)
        pygame.draw.rect(screen, (150, 110, 68), input_rect, width=3, border_radius=18)

        cursor = "_" if (pygame.time.get_ticks() // 350) % 2 == 0 else ""
        rendered_name = input_font.render((team_name or "Team name") + cursor, True, (34, 34, 34))
        screen.blit(rendered_name, (input_rect.x + 24, input_rect.y + 20))

        pygame.display.flip()
        clock.tick(30)


def show_leaderboard(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    entries: list[dict],
    leaderboard_dir: str | Path,
    background_image_path: str | Path | None,
    highlight_completed_at: str | None = None,
    max_entries: int = 4,
) -> None:
    leaderboard_dir = ensure_leaderboard_dir(leaderboard_dir)
    top_entries = sort_leaderboard_entries(entries)[:max_entries]
    background = _load_scaled_background(screen, background_image_path)
    title_font = pygame.font.Font("freesansbold.ttf", 44)
    row_font = pygame.font.Font("freesansbold.ttf", 28)
    small_font = pygame.font.Font("freesansbold.ttf", 22)

    pygame.event.clear()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
                return

        _draw_background(screen, background)

        title_surface = title_font.render("Temple Leaderboard", True, (66, 38, 16))
        title_rect = title_surface.get_rect(center=(screen.get_width() // 2, 92))
        screen.blit(title_surface, title_rect)

        subtitle_surface = small_font.render("Fastest 4 teams from Level 1 to Level 7", True, (96, 64, 37))
        subtitle_rect = subtitle_surface.get_rect(center=(screen.get_width() // 2, 132))
        screen.blit(subtitle_surface, subtitle_rect)

        if not top_entries:
            message_surface = row_font.render("No full-run teams recorded yet.", True, (84, 54, 29))
            message_rect = message_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(message_surface, message_rect)
        else:
            card_count = min(4, len(top_entries))
            margin_x = max(70, screen.get_width() // 18)
            margin_bottom = max(42, screen.get_height() // 20)
            grid_top = 170
            gap_x = max(30, screen.get_width() // 40)
            gap_y = max(26, screen.get_height() // 36)
            available_width = screen.get_width() - (2 * margin_x)
            available_height = screen.get_height() - grid_top - margin_bottom
            card_width = max(260, int((available_width - gap_x) / 2))
            card_height = max(220, int((available_height - gap_y) / 2))

            for index in range(card_count):
                entry = top_entries[index]
                rank = index + 1
                row_index = index // 2
                col_index = index % 2
                card_x = margin_x + col_index * (card_width + gap_x)
                card_y = grid_top + row_index * (card_height + gap_y)
                card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
                completed_at = entry.get("completed_at", "")
                highlight = completed_at == highlight_completed_at

                fill_color = (255, 246, 220) if highlight else (255, 252, 242)
                border_color = (206, 153, 70) if highlight else (155, 121, 78)
                pygame.draw.rect(screen, fill_color, card_rect, border_radius=24)
                pygame.draw.rect(screen, border_color, card_rect, width=4 if highlight else 2, border_radius=24)

                header_surface = row_font.render(f"Place #{rank}", True, (54, 34, 16))
                header_rect = header_surface.get_rect(midtop=(card_rect.centerx, card_rect.y + 14))
                screen.blit(header_surface, header_rect)

                image_padding = 18
                image_rect = pygame.Rect(
                    card_rect.x + image_padding,
                    card_rect.y + 56,
                    card_rect.width - (2 * image_padding),
                    int(card_rect.height * 0.52),
                )
                pygame.draw.rect(screen, (226, 210, 186), image_rect, border_radius=18)
                thumb = _load_profile_thumbnail(
                    leaderboard_dir,
                    entry.get("image_filename"),
                    (image_rect.width, image_rect.height),
                )
                if thumb is not None:
                    screen.blit(thumb, image_rect)
                else:
                    placeholder = small_font.render("No image", True, (105, 85, 61))
                    placeholder_rect = placeholder.get_rect(center=image_rect.center)
                    screen.blit(placeholder, placeholder_rect)

                details_y = image_rect.bottom + 10
                team_surface = row_font.render(entry.get("team_name", "Team"), True, (54, 34, 16))
                screen.blit(team_surface, (card_rect.x + 20, details_y))

                time_surface = small_font.render(f"Time: {format_duration(entry.get('total_seconds'))}", True, (78, 52, 27))
                screen.blit(time_surface, (card_rect.x + 20, details_y + 34))

                date_surface = small_font.render(f"Date: {entry.get('date', '--')}", True, (92, 64, 42))
                screen.blit(date_surface, (card_rect.x + 20, details_y + 64))

        footer_surface = small_font.render("Press Space to close", True, (96, 64, 37))
        footer_rect = footer_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() - 42))
        screen.blit(footer_surface, footer_rect)

        pygame.display.flip()
        clock.tick(30)


def maybe_record_and_show_leaderboard(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    leaderboard_dir: str | Path,
    background_image_path: str | Path | None,
    total_seconds: float | None,
    allow_name_entry: bool,
    snapshot_path: str | None,
    max_entries: int = 4,
) -> None:
    if not allow_name_entry or total_seconds is None:
        return

    leaderboard_dir = ensure_leaderboard_dir(leaderboard_dir)
    entries = load_leaderboard_entries(leaderboard_dir)
    rank = get_candidate_rank(entries, total_seconds, max_entries=max_entries)
    if rank is None:
        return

    team_name = prompt_for_team_name(screen, clock, background_image_path, total_seconds, rank)
    if not team_name:
        return

    completed_at = datetime.now().isoformat(timespec="seconds")
    image_filename = Path(snapshot_path).name if snapshot_path else None
    entries.append(
        {
            "team_name": team_name,
            "total_seconds": round(float(total_seconds), 2),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "completed_at": completed_at,
            "image_filename": image_filename,
        }
    )
    entries = sort_leaderboard_entries(entries)
    save_leaderboard_entries(leaderboard_dir, entries)

    show_leaderboard(
        screen,
        clock,
        entries,
        leaderboard_dir,
        background_image_path,
        highlight_completed_at=completed_at,
        max_entries=max_entries,
    )
