import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import generador as gen

# Estados del Agente
STATE_IDLE = 0            # Esperando orden
STATE_DELIVERING = 1      # Llevando paquete normal
STATE_RETURNING = 2       # Volviendo al almacén (urgencia)
STATE_URGENT_DELIVERY = 3 # Llevando paquete urgente

class CityMultiAgentEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, mapa_generado, n_agentes=2):
        super(CityMultiAgentEnv, self).__init__()

        self.base_map = np.array(mapa_generado)
        self.current_map = self.base_map.copy()
        self.rows, self.cols = self.base_map.shape
        self.n_agentes = n_agentes
        
        # --- 1. SETUP MAPA ---
        self.start_pos = np.array([0, 0])
        self.static_targets = []
        self.all_road_coords = []

        for r in range(self.rows):
            for c in range(self.cols):
                val = self.base_map[r, c]
                if val == gen.START:
                    self.start_pos = np.array([r, c])
                    self.all_road_coords.append((r, c))
                elif val == gen.DOOR:
                    self.static_targets.append((r, c))
                elif val == gen.ROAD_NORMAL:
                    self.all_road_coords.append((r, c))

        # --- 2. SETUP TRÁFICO FIJO (Tu lógica) ---
        num_jams = int(len(self.all_road_coords) * 0.10) # 10% de semáforos
        if num_jams < 2: num_jams = 2
        
        # Coordenadas FIJAS de semáforos (se eligen una sola vez)
        self.traffic_spots = random.sample(self.all_road_coords, num_jams)
        self.traffic_timer = 0
        self.TRAFFIC_FREQ = 20
        self.ROAD_LIGHT_GREEN = 6 # Código visual para verde

        # --- 3. GESTIÓN AGENTES Y TAREAS ---
        self.tasks = []
        self.agents = []
        for i in range(n_agentes):
            self.agents.append({
                'id': i,
                'pos': self.start_pos.copy(),
                'state': STATE_IDLE,
                'target': None,     # Destino inmediato
                'final_goal': None, # Cliente final
                'has_package': True
            })

        self.action_space = spaces.Discrete(4)
        self.fig = None; self.ax = None
        
        # Probabilidad de urgencia (baja para no colapsar la logística)
        self.URGENT_PROB = 0.02 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset Agentes
        for agent in self.agents:
            agent['pos'] = self.start_pos.copy()
            agent['state'] = STATE_IDLE
            agent['target'] = None
            agent['final_goal'] = None
            agent['has_package'] = True
            
        # Reset Tareas: Ponemos las puertas como tareas iniciales
        self.tasks = []
        for t in self.static_targets:
            self.tasks.append({'pos': t, 'urgente': False, 'asignado': False})
            
        # Reset Mapa y Tráfico
        self.current_map = self.base_map.copy()
        self.traffic_timer = 0
        self._update_traffic_lights(force=True)
            
        return self._get_states(), {}

    def _update_traffic_lights(self, force=False):
        """Actualiza el color de los semáforos fijos"""
        self.traffic_timer += 1
        if force or (self.traffic_timer % self.TRAFFIC_FREQ == 0):
            for r, c in self.traffic_spots:
                # No cambiar si hay un agente encima
                occupied = any(np.array_equal(a['pos'], [r,c]) for a in self.agents)
                if occupied: continue

                rng = random.random()
                if rng < 0.4:   self.current_map[r, c] = self.ROAD_LIGHT_GREEN # Verde (Cian)
                elif rng < 0.7: self.current_map[r, c] = gen.ROAD_SLOW # Amarillo
                else:           self.current_map[r, c] = gen.ROAD_JAM # Rojo

    def _get_neighbors_status(self, pos):
        """Sensores locales del agente"""
        sensors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                val = self.current_map[nr, nc]
                if val == gen.BUILDING: sensors.append(0)
                elif val == gen.ROAD_JAM: sensors.append(3) # Bloqueado
                elif val == gen.ROAD_SLOW: sensors.append(2) # Lento
                else: sensors.append(1) # Libre
            else: sensors.append(0)
        return tuple(sensors)

    def _get_states(self):
        """Estado: (PosAgente, PosTarget, SensoresLocales)"""
        states = []
        for agent in self.agents:
            if agent['target'] is None:
                target_pos = tuple(agent['pos'])
            else:
                target_pos = tuple(agent['target'])
                
            sensors = self._get_neighbors_status(agent['pos'])
            states.append((tuple(agent['pos']), target_pos, sensors))
        return states

    def _boss_assign_tasks(self):
        """EL JEFE: Asigna tareas a los libres"""
        free_agents = [a for a in self.agents if a['state'] == STATE_IDLE]
        pending_tasks = [t for t in self.tasks if not t['asignado']]
        random.shuffle(free_agents)
        
        for agent in free_agents:
            if not pending_tasks: break
            
            task = pending_tasks.pop(0)
            task['asignado'] = True
            
            agent['final_goal'] = task['pos']
            
            if task['urgente']:
                # Si es urgente, ir a START (Almacén) primero
                agent['state'] = STATE_RETURNING
                agent['target'] = tuple(self.start_pos) 
                agent['has_package'] = False
            else:
                # Si es normal, ir directo
                agent['state'] = STATE_DELIVERING
                agent['target'] = task['pos']
                agent['has_package'] = True

    def _generate_urgent_order(self):
        """Generador aleatorio de pedidos urgentes"""
        if random.random() < self.URGENT_PROB:
            if self.static_targets:
                target = random.choice(self.static_targets)
                self.tasks.append({'pos': target, 'urgente': True, 'asignado': False})

    def step(self, actions):
        self._update_traffic_lights()
        self._generate_urgent_order()
        self._boss_assign_tasks()
        
        rewards = [0] * self.n_agentes
        terminated = False
        
        for i, action in enumerate(actions):
            agent = self.agents[i]
            if agent['target'] is None: continue 
                
            move = [0, 0]
            if action == 0: move[0] = -1
            elif action == 1: move[0] = 1
            elif action == 2: move[1] = -1
            elif action == 3: move[1] = 1
            
            nr, nc = agent['pos'][0] + move[0], agent['pos'][1] + move[1]
            
            # Colisiones
            if (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or 
                self.current_map[nr, nc] == gen.BUILDING):
                rewards[i] -= 5
            else:
                # Costes Tráfico
                cell_type = self.current_map[nr, nc]
                cost = -1
                if cell_type == gen.ROAD_SLOW: cost = -2
                elif cell_type == gen.ROAD_JAM: cost = -5
                
                agent['pos'] = np.array([nr, nc])
                rewards[i] += cost
            
            # Objetivos
            dist = abs(agent['pos'][0]-agent['target'][0]) + abs(agent['pos'][1]-agent['target'][1])
            
            # GPS Shaping (Crucial para mapa 30x30)
            if dist == 0: rewards[i] += 10 # Llegó al sub-objetivo
            else: rewards[i] -= 0.1 # Pequeño coste por tiempo
            
            if dist == 0:
                if agent['state'] == STATE_RETURNING:
                    # Llegó a base, recoge urgente
                    agent['has_package'] = True
                    agent['state'] = STATE_URGENT_DELIVERY
                    agent['target'] = agent['final_goal']
                    rewards[i] += 20
                elif agent['state'] in [STATE_DELIVERING, STATE_URGENT_DELIVERY]:
                    # Entregó paquete
                    rewards[i] += 100
                    # Borrar tarea completada
                    for t in self.tasks:
                        if t['pos'] == agent['final_goal'] and t['asignado']:
                            self.tasks.remove(t)
                            break
                    agent['state'] = STATE_IDLE
                    agent['target'] = None
                    agent['final_goal'] = None

        # Termina si no hay tareas y todos están quietos
        if not self.tasks and all(a['state'] == STATE_IDLE for a in self.agents):
            terminated = True
            
        return self._get_states(), rewards, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.clear()
        
        # Colores
        cmap = mcolors.ListedColormap(['black', 'white', 'lightgreen', 'gray', 'gold', 'red', 'cyan'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        self.ax.imshow(self.current_map, cmap=cmap, norm=norm, origin='upper')

        # Tareas
        for task in self.tasks:
            color = 'purple' if task['urgente'] else 'blue'
            marker = 'P' if task['urgente'] else 'o'
            alpha = 0.3 if task['asignado'] else 1.0
            self.ax.plot(task['pos'][1], task['pos'][0], marker=marker, color=color, markersize=10, alpha=alpha)

        # Agentes
        colors = ['magenta', 'orange', 'lime']
        for agent in self.agents:
            c = colors[agent['id'] % len(colors)]
            fill = 'full' if agent['has_package'] else 'none'
            self.ax.plot(agent['pos'][1], agent['pos'][0], 's', color=c, fillstyle=fill, markersize=8, markeredgewidth=2)

        p = len([t for t in self.tasks if not t['asignado']])
        self.ax.set_title(f"Mapa 30x30 | Pendientes: {p} | Agentes: {self.n_agentes}")
        plt.pause(0.01)
