# --- Настройки вывода ---
set terminal pngcairo size 1920,1080 enhanced font "Arial,12"
set output "scene_camera.png"

# --- Настройки сцены ---
set title "Сцена и траектория камеры"
set grid
set view equal xyz  # Сохраняет пропорции осей 1:1:1
set xyplane at 0

set xlabel "X"
set ylabel "Y"
set zlabel "Z"

# Угол обзора самого графика (не путать с камерой)
set view 60, 30

# --- Настройка стиля стрелок ---
# 1: ID стиля
# head filled: закрашенные наконечники
# size screen 0.015,15,45: размер наконечника (относительно экрана), угол острия
# fixed: фиксированный размер наконечника, не зависит от длины стрелки
set style arrow 1 head filled size screen 0.015,15,45 fixed lw 2 lc rgb "dark-green"

# --- Коэффициент масштабирования длины стрелки ---
# Если target далеко, dx/dy/dz большие. Умножаем их на 0.2 (или меньше), 
# чтобы стрелка показывала направление, но была короткой.
scale = 0.2
# Шаг отрисовки стрелок (рисуем каждую 5-ю или 10-ю точку, чтобы не было месива)
step = 10

# --- Отрисовка ---
splot \
    "scene_polygons.dat" using 1:2:3 with lines lc rgb "gray50" title "Геометрия", \
    "camera_path.dat" using 1:2:3 with lines lw 3 lc rgb "red" title "Путь камеры", \
    "camera_path.dat" using 1:2:3:($4*scale):($5*scale):($6*scale) \
        every step with vectors arrowstyle 1 title "Взгляд"

unset output