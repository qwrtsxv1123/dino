import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright


# ============================================================
#  НАСТРОЙКИ ИГРЫ И ОБУЧЕНИЯ — меняй их под себя
# ============================================================

GAME_URL = "https://wayou.github.io/t-rex-runner/"

EPISODES = 5000                 # сколько жизней за один запуск программы
TARGET_SCORE = 50000            # цель: минимум 15000 очков
STOP_ON_TARGET = False          # остановить, когда цель достигнута
MAX_STEPS = 1000000             # максимум шагов за одну жизнь
STEP_DELAY = 0.016              # пауза между действиями, сек (~кадр)
START_DELAY = 1.2               # ожидание после запуска игры
RESTART_DELAY = 0.8             # ожидание после перезапуска
LOAD_DELAY = 2.0                # ожидание загрузки страницы

# --- Когда прыгать / приседать ---
# ttc = расстояние_до_препятствия / (скорость_игры + speedOffset_птицы).
# Так скорость игры учитывается автоматически.
JUMP_TTC_MIN = 3.2              # раньше — лишний/опасный прыжок
JUMP_TTC_MAX = 10.5             # позже — уже поздно
JUMP_EMERGENCY_TTC = 2.0        # аварийный прыжок, если что-то пошло не так
DUCK_TTC_MAX = 20.0             # начинаем приседать под средней/высокой птицей заранее
DUCK_RELEASE_MARGIN = 6.0       # держим приседание, пока птица не вышла из зоны динозавра

# --- Q-learning ---
# По умолчанию EPSILON = 0: динозавр сразу действует по безопасной эвристике,
# а не прыгает случайно. Таблица Q при этом всё равно обновляется.
ALPHA = 0.20                    # скорость обучения
GAMMA = 0.90                    # важность будущей награды
EPSILON_START = 0.00            # вероятность случайного БЕЗОПАСНОГО действия
EPSILON_END = 0.00
EPSILON_DECAY = 1.0

REWARD_STEP = 0.05              # награда за выживание
REWARD_PER_POINT = 0.02         # награда за очки
REWARD_CLEARED = 35.0           # награда за успешно пройденное препятствие
REWARD_MILESTONE = 3.0          # бонус за каждую новую 1000 очков
PENALTY_DEATH = -500.0          # штраф за смерть
PENALTY_USELESS_ACTION = -1.0   # штраф за бесполезное действие

USE_HEURISTIC_PRIOR = True      # начальные подсказки для новых состояний
PRIOR_VALUE = 8.0
USE_SAFETY_FILTER = True        # не разрешать очевидно опасные действия

# --- Файл и режимы ---
MODEL_FILE = "dino_q_table_v2.npy"
BEST_SCORE_FILE = "dino_best_score.txt"
TRAIN_MODE = True               # True = обновлять таблицу, False = только играть
LOAD_MODEL = True               # загружать таблицу при старте
SAVE_MODEL = True               # сохранять таблицу
SAVE_EVERY_EPISODES = 5         # сохранять каждые N эпизодов
RESET_MODEL_ON_START = False    # удалить старую таблицу при старте и начать с нуля
PRINT_Q_LIMIT = 60              # сколько строк таблицы печатать в конце

# --- Браузер ---
BROWSER_CHANNEL = "chrome"      # "chrome" = системный Chrome; None = chromium Playwright
HEADLESS = False                # False = видно окно браузера
BROWSER_ARGS = ["--disable-gpu"]
VIEWPORT = {"width": 900, "height": 500}
WAIT_FOR_ENTER = False          # ждать Enter перед закрытием браузера

# геометрия динозавра (из исходников игры)
DINO_WIDTH = 44
DINO_WIDTH_DUCK = 59

# ============================================================
#  КОНСТАНТЫ
# ============================================================

ACTION_RUN = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
ACTIONS = (ACTION_RUN, ACTION_JUMP, ACTION_DUCK)
ACTION_NAMES = ("БЕЖАТЬ", "ПРЫГАТЬ", "ПРИСЕДАТЬ")

OBSTACLE_NONE = 0
CACTUS_SMALL = 1
CACTUS_LARGE = 2
BIRD_LOW = 3         # yPos ~100 — низко: прыгаем
BIRD_MID = 4         # yPos ~75  — средняя высота: приседаем
BIRD_HIGH = 5        # yPos ~50  — высоко: приседаем

PHASE_RUNNING = 0
PHASE_JUMPING = 1
PHASE_DUCKING = 2

JUMP_KINDS = (CACTUS_SMALL, CACTUS_LARGE, BIRD_LOW)
DUCK_KINDS = (BIRD_MID, BIRD_HIGH)

q_table = {}
epsilon = EPSILON_START
best_score_all_time = 0


# ============================================================
#  Q-ТАБЛИЦА
# ============================================================

def new_row(state):
    """Создаёт строку Q-таблицы. Новые состояния появляются здесь автоматически."""
    row = [0.0, 0.0, 0.0]

    if not USE_HEURISTIC_PRIOR:
        return row

    kind, size, ttc_bucket, speed_bucket, phase = state

    if phase == PHASE_JUMPING:
        row[ACTION_RUN] = PRIOR_VALUE
    elif phase == PHASE_DUCKING and kind in DUCK_KINDS:
        row[ACTION_DUCK] = PRIOR_VALUE
    elif kind in JUMP_KINDS and 2 <= ttc_bucket <= 5:
        row[ACTION_JUMP] = PRIOR_VALUE
    elif kind in DUCK_KINDS and ttc_bucket <= 7:
        row[ACTION_DUCK] = PRIOR_VALUE
    else:
        row[ACTION_RUN] = PRIOR_VALUE * 0.6

    return row


def get_q(state):
    if state not in q_table:
        q_table[state] = new_row(state)
    return q_table[state]


def load_best_score():
    global best_score_all_time
    if os.path.exists(BEST_SCORE_FILE):
        try:
            with open(BEST_SCORE_FILE, "r", encoding="utf-8") as f:
                best_score_all_time = int(float(f.read().strip() or 0))
        except Exception:
            best_score_all_time = 0
    return best_score_all_time


def save_best_score():
    try:
        with open(BEST_SCORE_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(best_score_all_time)))
    except Exception as error:
        print(f"Не удалось сохранить лучший счёт: {error}")


def load_q_table():
    global q_table

    if RESET_MODEL_ON_START:
        for path in (MODEL_FILE, "dino_q_table.npy", "q_table.npy"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Удалена старая таблица: {path}")
                except Exception as error:
                    print(f"Не удалось удалить {path}: {error}")
        q_table = {}
        print("Q-таблица создана с нуля.")
        return

    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        try:
            loaded = np.load(MODEL_FILE, allow_pickle=True).item()
            q_table = {}
            for state, values in loaded.items():
                values = list(values)
                if len(values) < len(ACTIONS):
                    values += [0.0] * (len(ACTIONS) - len(values))
                q_table[tuple(state)] = values[:len(ACTIONS)]
            print(f"Загружена Q-таблица: {len(q_table)} состояний из {MODEL_FILE}")
        except Exception as error:
            print(f"Не удалось загрузить таблицу ({error}); создаём новую.")
            q_table = {}
    else:
        q_table = {}
        print("Q-таблица создана с нуля.")


def save_q_table():
    if not SAVE_MODEL:
        return
    try:
        np.save(MODEL_FILE, q_table)
        save_best_score()
        print(f"Сохранено: {MODEL_FILE} ({len(q_table)} состояний), best={best_score_all_time}")
    except Exception as error:
        print(f"Не удалось сохранить таблицу: {error}")


def update_q_table(state, action, reward, next_state, allowed_actions=None):
    row = get_q(state)
    current_q = row[action]

    if next_state is None:
        max_next_q = 0.0
    else:
        next_row = get_q(next_state)
        if allowed_actions:
            max_next_q = max(next_row[a] for a in allowed_actions)
        else:
            max_next_q = max(next_row)

    row[action] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)


def print_q_table():
    print("\nИтоговая Q-таблица (часть):")
    print("Действия: 0=БЕЖАТЬ, 1=ПРЫГАТЬ, 2=ПРИСЕДАТЬ")
    print("Состояние = (тип, размер, ttc, скорость, фаза)")
    items = sorted(q_table.items(), key=lambda item: str(item[0]))[:PRINT_Q_LIMIT]
    for state, values in items:
        best = int(np.argmax(values))
        print(f"{state} -> [{values[0]:7.2f}, {values[1]:7.2f}, {values[2]:7.2f}] = {ACTION_NAMES[best]}")
    if len(q_table) > PRINT_Q_LIMIT:
        print(f"... и ещё {len(q_table) - PRINT_Q_LIMIT} состояний")


# ============================================================
#  ЧТЕНИЕ ИГРЫ
# ============================================================

PAGE_STATE_JS = r"""
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return { status: 'NO_RUNNER' };

    const score = runner.distanceMeter
        ? runner.distanceMeter.getActualDistance(Math.ceil(runner.distanceRan || 0))
        : 0;

    if (runner.crashed) return { status: 'DEAD', score: score };
    if (!runner.playing) return { status: 'NOT_PLAYING', score: score };

    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];

    window.__dinoIds = window.__dinoIds || { counter: 0, map: new WeakMap() };

    const seen = obstacles.map(obstacle => {
        let id = window.__dinoIds.map.get(obstacle);
        if (!id) {
            id = ++window.__dinoIds.counter;
            window.__dinoIds.map.set(obstacle, id);
        }
        return {
            id: id,
            type: obstacle.type || (obstacle.typeConfig && obstacle.typeConfig.type) || '',
            size: obstacle.size || 1,
            xPos: obstacle.xPos,
            yPos: Array.isArray(obstacle.yPos) ? obstacle.yPos[0] : obstacle.yPos,
            width: obstacle.width || (obstacle.typeConfig && obstacle.typeConfig.width) || 0,
            speedOffset: obstacle.speedOffset || 0,
        };
    });

    return {
        status: 'OK',
        score: score,
        speed: runner.currentSpeed,
        dino: {
            xPos: dino.xPos,
            yPos: dino.yPos,
            jumping: !!dino.jumping,
            ducking: !!dino.ducking,
            groundYPos: dino.groundYPos,
        },
        obstacles: seen,
    };
}
"""


def read_game(page):
    try:
        return page.evaluate(PAGE_STATE_JS)
    except Exception as error:
        return {"status": "ERROR", "error": str(error), "obstacles": []}


def update_browser_title(page, score, best, episode):
    try:
        title = f"Dino AI | score: {int(score)} | best: {int(best)} | ep: {episode}"
        page.evaluate("(t) => { document.title = t; }", title)
    except Exception:
        pass


def encode_obstacle(obstacle):
    if obstacle is None:
        return OBSTACLE_NONE

    kind = obstacle.get("type", "")
    y_pos = obstacle.get("yPos", 0)

    if kind == "CACTUS_SMALL":
        return CACTUS_SMALL
    if kind == "CACTUS_LARGE":
        return CACTUS_LARGE
    if kind == "PTERODACTYL":
        if y_pos >= 90:
            return BIRD_LOW
        if y_pos >= 65:
            return BIRD_MID
        return BIRD_HIGH
    return OBSTACLE_NONE


def bucket_ttc(ttc):
    if ttc <= 0:
        return 0
    if ttc <= 2:
        return 1
    if ttc <= 4:
        return 2
    if ttc <= 6:
        return 3
    if ttc <= 8:
        return 4
    if ttc <= 10:
        return 5
    if ttc <= 13:
        return 6
    if ttc <= 18:
        return 7
    return 8


def bucket_speed(speed):
    if speed < 7:
        return 0
    if speed < 9:
        return 1
    if speed < 11:
        return 2
    if speed < 12.5:
        return 3
    return 4


def analyze(data, duck_held):
    """Строит состояние Q-таблицы и список безопасных действий."""
    obstacles = data.get("obstacles") or []
    dino = data["dino"]
    speed = data.get("speed", 6.0)

    dino_width = DINO_WIDTH_DUCK if (duck_held or dino["ducking"]) else DINO_WIDTH

    candidates = []
    for obstacle in obstacles:
        kind = encode_obstacle(obstacle)
        relative_speed = max(speed + obstacle.get("speedOffset", 0), 1.0)
        front = obstacle["xPos"] - (dino["xPos"] + dino_width)
        rear = obstacle["xPos"] + obstacle.get("width", 0) - dino["xPos"]
        if rear <= -DUCK_RELEASE_MARGIN:
            continue
        candidates.append({
            "kind": kind,
            "ttc": front / relative_speed,
            "front": front,
            "rear": rear,
            "obstacle": obstacle,
        })

    candidates.sort(key=lambda item: item["ttc"])
    primary = candidates[0] if candidates else None

    kind = primary["kind"] if primary else OBSTACLE_NONE
    ttc = primary["ttc"] if primary else 999.0
    front = primary["front"] if primary else 999.0
    rear = primary["rear"] if primary else -999.0
    obstacle_id = primary["obstacle"]["id"] if primary else None
    size_bucket = min(int(primary["obstacle"].get("size", 1)), 3) if primary else 0

    keep_duck = False
    if duck_held or dino["ducking"]:
        for candidate in candidates:
            if candidate["kind"] in DUCK_KINDS and candidate["rear"] > -DUCK_RELEASE_MARGIN:
                keep_duck = True
                break

    if dino["jumping"]:
        phase = PHASE_JUMPING
    elif dino["ducking"] or duck_held:
        phase = PHASE_DUCKING
    else:
        phase = PHASE_RUNNING

    state = (kind, size_bucket, bucket_ttc(ttc), bucket_speed(speed), phase)

    # --- Безопасные действия ---
    if phase == PHASE_JUMPING:
        safe = (ACTION_RUN,)
    elif keep_duck:
        safe = (ACTION_DUCK,)
    elif kind in JUMP_KINDS:
        # Если препятствия идут подряд, прыгаем позже, чтобы один прыжок
        # накрыл серию и не приземлиться прямо перед следующим.
        cluster = [
            c for c in candidates
            if c["kind"] in JUMP_KINDS and c["ttc"] <= ttc + 18.0
        ]
        effective_max = JUMP_TTC_MAX
        if len(cluster) > 1:
            effective_max = max(JUMP_TTC_MIN + 1.0, JUMP_TTC_MAX - 2.5 * (len(cluster) - 1))

        if JUMP_TTC_MIN <= ttc <= effective_max:
            safe = (ACTION_JUMP, ACTION_RUN)
        elif JUMP_EMERGENCY_TTC <= ttc < JUMP_TTC_MIN:
            safe = (ACTION_JUMP, ACTION_RUN)   # аварийный поздний прыжок
        else:
            safe = (ACTION_RUN,)
    elif kind in DUCK_KINDS:
        if ttc <= DUCK_TTC_MAX:
            safe = (ACTION_DUCK, ACTION_RUN)
        else:
            safe = (ACTION_RUN,)
    else:
        safe = (ACTION_RUN,)

    # Доп. защита: если очень близко прыжковое препятствие и мы на земле — прыгать.
    if phase == PHASE_RUNNING and kind in JUMP_KINDS and 0 < ttc <= JUMP_EMERGENCY_TTC + 1.5:
        safe = (ACTION_JUMP,)

    if not USE_SAFETY_FILTER:
        safe = tuple(
            a for a in ACTIONS
            if not (phase == PHASE_JUMPING and a in (ACTION_JUMP, ACTION_DUCK))
        )

    info = {
        "state": state,
        "safe": safe,
        "kind": kind,
        "ttc": ttc,
        "front": front,
        "rear": rear,
        "obstacle_id": obstacle_id,
        "all_ids": {item["obstacle"]["id"] for item in candidates},
        "keep_duck": keep_duck,
        "phase": phase,
        "speed": speed,
    }
    return info


# ============================================================
#  ДЕЙСТВИЯ
# ============================================================

def choose_action(info, training):
    global epsilon

    state = info["state"]
    safe = info["safe"]
    row = get_q(state)

    # если есть одно безопасное действие — берём его
    if len(safe) == 1:
        action = safe[0]
    elif training and random.random() < epsilon:
        action = random.choice(safe)
    else:
        # выбираем лучшее среди безопасных
        action = max(safe, key=lambda a: (row[a], -a))

    if training:
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return action


def apply_action(page, action, duck_held):
    """Применяет действие. Возвращает новое состояние удержания ArrowDown."""
    if action == ACTION_DUCK:
        if not duck_held:
            page.keyboard.down("ArrowDown")
        return True

    if duck_held:
        page.keyboard.up("ArrowDown")

    if action == ACTION_JUMP:
        page.keyboard.press("Space")

    return False


# ============================================================
#  УПРАВЛЕНИЕ ИГРОЙ
# ============================================================

def wait_until_playing(page, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = read_game(page)
        if data.get("status") == "OK":
            return data
        time.sleep(0.1)
    return None


def start_game(page):
    page.keyboard.press("Space")
    time.sleep(START_DELAY)
    return wait_until_playing(page)


def restart_game(page, duck_held):
    if duck_held:
        try:
            page.keyboard.up("ArrowDown")
        except Exception:
            pass

    try:
        page.evaluate(
            "() => { const r = window.Runner && window.Runner.instance_; if (r) r.restart(); }"
        )
    except Exception:
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass

    time.sleep(RESTART_DELAY)
    data = wait_until_playing(page, timeout=4.0)
    if data is None:
        try:
            page.keyboard.press("Space")
        except Exception:
            pass
        time.sleep(0.5)
        data = wait_until_playing(page, timeout=3.0)
    return data


# ============================================================
#  ОСНОВНОЙ ЦИКЛ
# ============================================================

def main():
    global epsilon, best_score_all_time

    print("Запускаем Dino AI...")
    print(f"Цель: {TARGET_SCORE} очков | эпизодов: {EPISODES} | модель: {MODEL_FILE}")
    load_q_table()
    load_best_score()
    epsilon = EPSILON_START
    session_best = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel=BROWSER_CHANNEL,
            headless=HEADLESS,
            args=BROWSER_ARGS,
        )
        page = browser.new_page(viewport=VIEWPORT)

        print("Открываем игру...")
        page.goto(GAME_URL, wait_until="domcontentloaded")
        time.sleep(LOAD_DELAY)

        duck_held = False

        try:
            for episode in range(1, EPISODES + 1):
                if episode == 1:
                    data = start_game(page)
                else:
                    data = restart_game(page, duck_held)
                    duck_held = False

                if data is None or data.get("status") != "OK":
                    print(f"Эпизод {episode}: игра не запустилась, перезагружаю страницу.")
                    try:
                        page.goto(GAME_URL, wait_until="domcontentloaded")
                        time.sleep(LOAD_DELAY)
                        data = start_game(page)
                    except Exception:
                        data = None
                    if data is None or data.get("status") != "OK":
                        continue

                print(f"\n--- Игра №{episode} ---")
                total_reward = 0.0
                prev_score = data.get("score", 0)
                handled = {}
                finished_by_goal = False

                for step in range(1, MAX_STEPS + 1):
                    info = analyze(data, duck_held)
                    state = info["state"]
                    action = choose_action(info, TRAIN_MODE)

                    if action in (ACTION_JUMP, ACTION_DUCK) and info["obstacle_id"] is not None:
                        handled.setdefault(info["obstacle_id"], {
                            "state": state,
                            "action": action,
                            "step": step,
                        })

                    duck_held = apply_action(page, action, duck_held)
                    time.sleep(STEP_DELAY)

                    next_data = read_game(page)
                    status = next_data.get("status")

                    if status != "OK":
                        if TRAIN_MODE and status == "DEAD":
                            if handled:
                                last = max(handled.values(), key=lambda item: item["step"])
                                update_q_table(last["state"], last["action"], PENALTY_DEATH, None)
                            else:
                                update_q_table(state, action, PENALTY_DEATH, None)
                        total_reward += PENALTY_DEATH if status == "DEAD" else 0.0
                        prev_score = next_data.get("score", prev_score)
                        best_score_all_time = max(best_score_all_time, int(prev_score))
                        session_best = max(session_best, int(prev_score))
                        update_browser_title(page, prev_score, best_score_all_time, episode)
                        print(f"  смерть: шаг {step}, очки {prev_score}, "
                              f"награда {total_reward:.1f}, epsilon {epsilon:.3f}")
                        break

                    next_info = analyze(next_data, duck_held)
                    next_state = next_info["state"]
                    score = next_data.get("score", prev_score)

                    reward = REWARD_STEP + REWARD_PER_POINT * max(0, score - prev_score)
                    if score // 1000 > prev_score // 1000:
                        reward += REWARD_MILESTONE

                    if (action in (ACTION_JUMP, ACTION_DUCK)
                            and info["kind"] == OBSTACLE_NONE
                            and not info["keep_duck"]):
                        reward += PENALTY_USELESS_ACTION

                    if TRAIN_MODE:
                        update_q_table(state, action, reward, next_state, next_info["safe"])

                    total_reward += reward

                    cleared_ids = [
                        obstacle_id for obstacle_id in list(handled)
                        if obstacle_id not in next_info["all_ids"]
                    ]
                    for obstacle_id in cleared_ids:
                        record = handled.pop(obstacle_id)
                        total_reward += REWARD_CLEARED
                        if TRAIN_MODE:
                            update_q_table(
                                record["state"],
                                record["action"],
                                REWARD_CLEARED,
                                next_state,
                                next_info["safe"],
                            )

                    prev_score = score
                    data = next_data

                    if step % 100 == 0:
                        best_score_all_time = max(best_score_all_time, int(prev_score))
                        session_best = max(session_best, int(prev_score))
                        update_browser_title(page, prev_score, best_score_all_time, episode)

                    if prev_score >= TARGET_SCORE:
                        best_score_all_time = max(best_score_all_time, int(prev_score))
                        session_best = max(session_best, int(prev_score))
                        update_browser_title(page, prev_score, best_score_all_time, episode)
                        finished_by_goal = True
                        print(f"  ЦЕЛЬ ДОСТИГНУТА: {prev_score} очков на шаге {step}!")
                        break
                else:
                    best_score_all_time = max(best_score_all_time, int(prev_score))
                    session_best = max(session_best, int(prev_score))
                    update_browser_title(page, prev_score, best_score_all_time, episode)
                    print(f"  лимит шагов: очки {prev_score}, награда {total_reward:.1f}")

                best_score_all_time = max(best_score_all_time, int(prev_score))
                session_best = max(session_best, int(prev_score))
                print(f"  лучший за сессию: {session_best} | лучший вообще: {best_score_all_time} | "
                      f"epsilon: {epsilon:.3f} | состояний: {len(q_table)}")

                if TRAIN_MODE and episode % SAVE_EVERY_EPISODES == 0:
                    save_q_table()

                if finished_by_goal and STOP_ON_TARGET:
                    break

        finally:
            print_q_table()
            if TRAIN_MODE:
                save_q_table()
            save_best_score()
            if WAIT_FOR_ENTER:
                try:
                    input("\nНажми Enter, чтобы закрыть браузер...")
                except Exception:
                    pass
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
