import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import generador as gen

# Estados
STATE_IDLE = 0            
STATE_DELIVERING = 1      
STATE_RETURNING = 2       
STATE_URGENT_DELIVERY = 3 

class CityMultiAgentEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mapa_generado, n_agentes=2):
        super(CityMultiAgentEnv, self).__init__()
        self.base_map = np.array(mapa_generado)
        self.current_map = self.base_map.copy()
        self.rows, self.cols = self.base_map.shape
        self.n_agentes = n_agentes
        self.start_pos = np.array([0, 0])
        self.static_targets = []
        self.all_road_coords = []
        self.step_logs = []

        # Setup Mapa
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

        # Setup Tráfico
        num_jams = int(len(self.all_road_coords) * 0.10) 
        if num_jams < 2: num_jams = 2
        self.traffic_spots = random.sample(self.all_road_coords, num_jams)
        self.traffic_timer = 0
        self.TRAFFIC_FREQ = 20
        self.ROAD_LIGHT_GREEN = 6 

        # Agentes (Inicialización)
        self.tasks = []
        self.agents = []
        # No inicializamos aquí, lo hacemos en reset()
        
        self.action_space = spaces.Discrete(4)
        self.URGENT_PROB = 0.05 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agents = [] # Reiniciar lista
        for i in range(self.n_agentes):
            self.agents.append({
                'id': i, 
                'pos': self.start_pos.copy(), 
                'state': STATE_IDLE,
                'target': None, 
                'final_goal': None, 
                'has_package': True,
                # --- NUEVAS VARIABLES (KPIs) ---
                'total_distance': 0,      # Pasos dados
                'deliveries_count': 0,    # Paquetes entregados
                'route_history': [tuple(self.start_pos.copy())] # Historial de coordenadas
                # -------------------------------
            })
            
        self.tasks = []
        for t in self.static_targets:
            self.tasks.append({'pos': t, 'urgente': False, 'asignado': False})
            
        self.current_map = self.base_map.copy()
        self.traffic_timer = 0
        self._update_traffic_lights(force=True)
        return self._get_states(), {}

    def _update_traffic_lights(self, force=False):
        self.traffic_timer += 1
        if force or (self.traffic_timer % self.TRAFFIC_FREQ == 0):
            cambios = 0
            for r, c in self.traffic_spots:
                occupied = any(np.array_equal(a['pos'], [r,c]) for a in self.agents)
                if occupied: continue
                rng = random.random()
                prev = self.current_map[r, c]
                if rng < 0.4: self.current_map[r, c] = self.ROAD_LIGHT_GREEN 
                elif rng < 0.7: self.current_map[r, c] = gen.ROAD_SLOW 
                else: self.current_map[r, c] = gen.ROAD_JAM
                
                if prev != self.current_map[r, c] and random.random() < 0.3:
                    cambios += 1
            if cambios > 0:
                self.step_logs.append(f"🚦 TRÁFICO: Se han actualizado {cambios} semáforos.")

    def _get_neighbors_status(self, pos):
        sensors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                val = self.current_map[nr, nc]
                if val == gen.BUILDING: sensors.append(0)
                elif val == gen.ROAD_JAM: sensors.append(3)
                elif val == gen.ROAD_SLOW: sensors.append(2)
                else: sensors.append(1)
            else: sensors.append(0)
        return tuple(sensors)

    def _get_states(self):
        states = []
        for agent in self.agents:
            target_pos = tuple(agent['target']) if agent['target'] is not None else tuple(agent['pos'])
            sensors = self._get_neighbors_status(agent['pos'])
            states.append((tuple(agent['pos']), target_pos, sensors))
        return states

    def _boss_assign_tasks(self):
        free_agents = [a for a in self.agents if a['state'] == STATE_IDLE]
        pending_tasks = [t for t in self.tasks if not t['asignado']]
        random.shuffle(free_agents)
        
        for agent in free_agents:
            if not pending_tasks: break
            task = pending_tasks.pop(0)
            task['asignado'] = True
            agent['final_goal'] = task['pos']
            
            tipo = "URGENTE ⚡" if task['urgente'] else "ESTÁNDAR 📦"
            self.step_logs.append(f"🤖 JEFE: Asignando pedido {tipo} al Agente {agent['id']}")
            
            if task['urgente']:
                agent['state'] = STATE_RETURNING
                agent['target'] = tuple(self.start_pos) 
                agent['has_package'] = False
            else:
                agent['state'] = STATE_DELIVERING
                agent['target'] = task['pos']
                agent['has_package'] = True

    def _generate_urgent_order(self):
        if random.random() < self.URGENT_PROB:
            if self.static_targets:
                target = random.choice(self.static_targets)
                self.tasks.append({'pos': target, 'urgente': True, 'asignado': False})
                self.step_logs.append(f"📞 CALL CENTER: ¡Nuevo pedido URGENTE entrante!")

    def step(self, actions):
        self.step_logs = [] 
        self._update_traffic_lights()
        self._generate_urgent_order()
        self._boss_assign_tasks()
        
        rewards = [0] * self.n_agentes
        terminated = False
        
        for i, action in enumerate(actions):
            agent = self.agents[i]
            if agent['target'] is None: continue 
            
            prev_dist = abs(agent['pos'][0]-agent['target'][0]) + abs(agent['pos'][1]-agent['target'][1])
            prev_pos_tuple = tuple(agent['pos']) # Guardar posición anterior para comparar

            move = [0, 0]
            if action == 0: move[0] = -1
            elif action == 1: move[0] = 1
            elif action == 2: move[1] = -1
            elif action == 3: move[1] = 1
            
            nr, nc = agent['pos'][0] + move[0], agent['pos'][1] + move[1]
            
            if (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or self.current_map[nr, nc] == gen.BUILDING):
                rewards[i] -= 10
            else:
                cell_type = self.current_map[nr, nc]
                cost = -0.1
                if cell_type == gen.ROAD_SLOW: cost = -2.0
                elif cell_type == gen.ROAD_JAM: cost = -5.0
                agent['pos'] = np.array([nr, nc])
                rewards[i] += cost
            
            # --- TELEMETRÍA: Actualizar KPIs ---
            current_pos_tuple = tuple(agent['pos'])
            if current_pos_tuple != prev_pos_tuple:
                agent['total_distance'] += 1
                agent['route_history'].append(current_pos_tuple)
            # -----------------------------------

            curr_dist = abs(agent['pos'][0]-agent['target'][0]) + abs(agent['pos'][1]-agent['target'][1])
            if curr_dist < prev_dist: rewards[i] += 2.0
            elif curr_dist > prev_dist: rewards[i] -= 2.5
            else: rewards[i] -= 0.5

            if curr_dist == 0:
                if agent['state'] == STATE_RETURNING:
                    agent['has_package'] = True
                    agent['state'] = STATE_URGENT_DELIVERY
                    agent['target'] = agent['final_goal']
                    rewards[i] += 50
                    self.step_logs.append(f"🚛 AGENTE {i}: Recogido paquete urgente en base.")
                elif agent['state'] in [STATE_DELIVERING, STATE_URGENT_DELIVERY]:
                    rewards[i] += 200
                    # --- TELEMETRÍA: Sumar entrega ---
                    agent['deliveries_count'] += 1
                    # ---------------------------------
                    for t in self.tasks:
                        if t['pos'] == agent['final_goal'] and t['asignado']:
                            self.tasks.remove(t)
                            break
                    agent['state'] = STATE_IDLE
                    agent['target'] = None
                    agent['final_goal'] = None
                    self.step_logs.append(f"✅ AGENTE {i}: ¡Paquete entregado con éxito!")

        if not self.tasks and all(a['state'] == STATE_IDLE for a in self.agents):
            terminated = True
            self.step_logs.append("🏆 SISTEMA: Todas las entregas finalizadas.")
            
        return self._get_states(), rewards, terminated, False, {"logs": self.step_logs}
            