"""
Immediate-Mode UI Widgets for dens-city Raylib Dashboard.
"""

try:
    import pyray as rl
except ImportError:
    rl = None


def draw_panel(
    x: int, y: int, width: int, height: int, title: str, bg_color=(20, 24, 30, 230), border_color=(50, 60, 75, 255)
):
    if rl is None:
        return
    rl.draw_rectangle(x, y, width, height, bg_color)
    rl.draw_rectangle_lines(x, y, width, height, border_color)
    if title:
        rl.draw_text(title, x + 12, y + 10, 16, (220, 230, 245, 255))
        rl.draw_line(x + 10, y + 32, x + width - 10, y + 32, border_color)


def draw_slider(
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    value: float,
    min_val: float,
    max_val: float,
    format_str: str = "{:.2f}",
) -> float:
    if rl is None:
        return value

    rl.draw_text(label, x, y, 14, (180, 190, 205, 255))
    val_text = format_str.format(value)
    rl.draw_text(val_text, x + width - 60, y, 14, (100, 200, 255, 255))

    bar_y = y + 18
    bar_h = height - 18
    rl.draw_rectangle(x, bar_y, width, bar_h, (35, 42, 52, 255))
    rl.draw_rectangle_lines(x, bar_y, width, bar_h, (60, 70, 85, 255))

    ratio = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    fill_w = int(width * max(0.0, min(1.0, ratio)))
    rl.draw_rectangle(x + 1, bar_y + 1, fill_w, bar_h - 2, (0, 140, 220, 200))

    mouse = rl.get_mouse_position()
    if rl.is_mouse_button_down(0):
        if x <= mouse.x <= x + width and bar_y <= mouse.y <= bar_y + bar_h:
            new_ratio = (mouse.x - x) / float(width)
            value = min_val + new_ratio * (max_val - min_val)
            value = max(min_val, min(max_val, value))

    return value


def draw_button(x: int, y: int, width: int, height: int, text: str, active: bool = False) -> bool:
    if rl is None:
        return False
    mouse = rl.get_mouse_position()
    hover = x <= mouse.x <= x + width and y <= mouse.y <= y + height

    bg = (0, 160, 240, 255) if active else ((55, 68, 85, 255) if hover else (40, 48, 60, 255))
    rl.draw_rectangle(x, y, width, height, bg)
    rl.draw_rectangle_lines(x, y, width, height, (80, 100, 125, 255))

    text_w = rl.measure_text(text, 14)
    rl.draw_text(text, x + (width - text_w) // 2, y + (height - 14) // 2, 14, (240, 245, 255, 255))

    return hover and rl.is_mouse_button_pressed(0)
