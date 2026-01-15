import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import generador as gen


class CityTrafficLightsEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, mapa_generado):
        super(CityTrafficLightsEnv, self).__init__()

        self.base_map = np.array(mapa_generado)
        self.current_map = self.base_map.copy()
        self.rows, self.cols = self.base_map.shape

        self.targets = []
        self.start_pos = np.array([0, 0])
        self.all_road_coords = []

        # Identificar elementos
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.base_map[r, c]
                if val == gen.START:
                    self.start_pos = np.array([r, c])
                    self.all_road_coords.append((r, c))
                elif val == gen.DOOR:
                    self.targets.append((r, c))
                elif val == gen.ROAD_NORMAL:
                    self.all_road_coords.append((r, c))

        self.n_targets = len(self.targets)
        self.action_space = spaces.Discrete(4)
        self.agent_pos = None
        self.visited_status = None

        # --- SELECCIÓN DE OBSTÁCULOS FIJOS (Solo una vez) ---
        num_jams = int(len(self.all_road_coords) * 0.05)  # 15% de semáforos
        if num_jams < 2:
            num_jams = 2

        # Elegimos coordenadas que NUNCA cambiarán
        self.traffic_spots = random.sample(self.all_road_coords, num_jams)

        # Constante interna para visualizar "Semáforo en Verde"
        # (No está en generador.py, la usamos solo aquí para pintar)
        self.ROAD_LIGHT_GREEN = 6

        self.step_counter = 0
        self.TRAFFIC_CHANGE_FREQ = 20
        self.prev_distance = 0
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.visited_status = [0] * self.n_targets

        # Restaurar mapa base
        self.current_map = self.base_map.copy()
        self.step_counter = 0

        # Pintar los semáforos en sus sitios fijos
        self._update_traffic_lights(force=True)

        self.prev_distance = self._calculate_min_dist()
        return self._get_state(), {}

    def _calculate_min_dist(self):
        min_d = float('inf')
        found = False
        for i, target in enumerate(self.targets):
            if self.visited_status[i] == 0:
                d = abs(self.agent_pos[0]-target[0]) + \
                    abs(self.agent_pos[1]-target[1])
                if d < min_d:
                    min_d = d
                    found = True
        return min_d if found else 0

    def _update_traffic_lights(self, force=False):
        self.step_counter += 1

        # Solo actualizamos colores cada X tiempo
        if force or (self.step_counter % self.TRAFFIC_CHANGE_FREQ == 0):

            # Recorremos SIEMPRE la misma lista self.traffic_spots
            for r, c in self.traffic_spots:

                # Evitar pintar encima del agente
                if r == self.agent_pos[0] and c == self.agent_pos[1]:
                    continue

                # Sorteamos el estado del semáforo
                rng = random.random()

                if rng < 0.4:
                    # VERDE: Usamos el código 6 para que se vea AZUL CIAN
                    # Así el usuario sabe que ahí hay un semáforo aunque esté abierto
                    if self.base_map[r, c] == gen.START:
                        self.current_map[r, c] = gen.START
                    else:
                        self.current_map[r, c] = self.ROAD_LIGHT_GREEN

                elif rng < 0.7:
                    # AMARILLO
                    self.current_map[r, c] = gen.ROAD_SLOW

                else:
                    # ROJO
                    self.current_map[r, c] = gen.ROAD_JAM

    def _get_neighbors_status(self):
        sensors = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in moves:
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                val = self.current_map[nr, nc]
                if val == gen.BUILDING:
                    sensors.append(0)
                elif val == gen.ROAD_JAM:
                    sensors.append(2)  # Rojo
                elif val == gen.ROAD_SLOW:
                    sensors.append(1)  # Amarillo
                else:
                    sensors.append(1)  # Verde/Cian/Blanco (Libre)
            else:
                sensors.append(0)
        return tuple(sensors)

    def _get_state(self):
        return (tuple(self.agent_pos), self._get_neighbors_status(), tuple(self.visited_status))

    def step(self, action):
        self._update_traffic_lights()

        new_pos = self.agent_pos.copy()
        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] -= 1
        elif action == 3:
            new_pos[1] += 1

        reward = 0

        # Colisiones
        if (new_pos[0] < 0 or new_pos[0] >= self.rows or
            new_pos[1] < 0 or new_pos[1] >= self.cols or
                self.current_map[new_pos[0], new_pos[1]] == gen.BUILDING):

            new_pos = self.agent_pos
            reward = -5
        else:
            # Costes
            cell_type = self.current_map[new_pos[0], new_pos[1]]

            if cell_type == gen.ROAD_JAM:
                move_cost = -5
            elif cell_type == gen.ROAD_SLOW:
                move_cost = -2
            else:
                # Normal (1) o Semáforo Verde (6) o Inicio (3)
                move_cost = -1

            self.agent_pos = new_pos
            reward += move_cost

        # GPS
        dist = self._calculate_min_dist()
        if dist < self.prev_distance:
            reward += 2
        elif dist > self.prev_distance:
            reward -= 2
        self.prev_distance = dist

        # Entregas
        pos_tuple = tuple(self.agent_pos)
        if pos_tuple in self.targets:
            idx = self.targets.index(pos_tuple)
            if self.visited_status[idx] == 0:
                self.visited_status[idx] = 1
                reward += 50

        terminated = all(s == 1 for s in self.visited_status)
        if terminated:
            reward += 100

        return self._get_state(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()

        # DEFINICIÓN DE COLORES
        # 0=Negro(Edificio), 1=Blanco(Calle), 2=Verde(Puerta), 3=Gris(Start)
        # 4=Amarillo(Lento), 5=Rojo(Atasco), 6=Cyan(Semáforo Verde)

        colors = ['black', 'white', 'lightgreen',
                  'gray', 'gold', 'red', 'green']
        cmap = mcolors.ListedColormap(colors)

        # Los límites deben coincidir con los números de arriba
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        self.ax.imshow(self.current_map, cmap=cmap, norm=norm, origin='upper')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for i, (tr, tc) in enumerate(self.targets):
            if self.visited_status[i] == 0:
                self.ax.text(tc, tr, "P", color='blue',
                             ha='center', fontweight='bold')

        self.ax.plot(self.agent_pos[1], self.agent_pos[0], 'bs', markersize=12)

        step_mod = self.step_counter % self.TRAFFIC_CHANGE_FREQ
        next_change = self.TRAFFIC_CHANGE_FREQ - step_mod

        entregados = sum(self.visited_status)
        self.ax.set_title(f"Semáforos Fijos | Cambio en: {next_change}")
        plt.pause(0.01)
