import numpy as np
import pickle
import os
from entorno import CityMultiAgentEnv
import generador

MODEL_FILE = "q_table_model.pkl"

def get_action(state, q_table, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        return np.argmax(q_table[state])

# --- MODIFICADO: Añadido parámetro callback=None ---
def entrenar_agentes(episodes=5000, callback=None):
    print("🏗️  Generando mapa para entrenamiento...")
    mapa = generador.generate_map()
    env = CityMultiAgentEnv(mapa, n_agentes=2)
    
    q_table = {}
    
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    
    print(f"🚦 Iniciando entrenamiento de {episodes} episodios...")
    
    for i in range(episodes):
        states, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200: # Reduje steps max para acelerar un poco la demo
            actions = []
            for idx in range(env.n_agentes):
                state = states[idx]
                action = get_action(state, q_table, epsilon, env)
                actions.append(action)
            
            next_states, rewards, terminated, _, _ = env.step(actions)
            done = terminated
            
            for idx in range(env.n_agentes):
                state = states[idx]
                action = actions[idx]
                reward = rewards[idx]
                next_state = next_states[idx]
                
                if next_state not in q_table:
                    q_table[next_state] = np.zeros(env.action_space.n)
                if state not in q_table:
                    q_table[state] = np.zeros(env.action_space.n)
                
                old_val = q_table[state][action]
                next_max = np.max(q_table[next_state])
                
                q_table[state][action] = old_val + alpha * (reward + gamma * next_max - old_val)
            
            states = next_states
            steps += 1
            
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            
        # --- MODIFICADO: Llamada al callback para la barra de progreso ---
        if callback and (i+1) % 50 == 0:
            progreso = (i+1) / episodes
            callback(progreso, i+1, epsilon)
            
    data_to_save = {
        "q_table": q_table,
        "mapa": mapa 
    }
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data_to_save, f)
    
    return mapa, q_table

def cargar_modelo():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        return data["mapa"], data["q_table"]
    else:
        return None, None