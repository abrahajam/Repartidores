import numpy as np
import random
import matplotlib.pyplot as plt
import pickle  # <--- NECESARIO PARA GUARDAR
import os      # <--- NECESARIO PARA VERIFICAR SI EXISTE EL ARCHIVO
import generador
from entorno import CityMultiAgentEnv

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = "cerebro_logistico.pkl"
MODO_ENTRENAMIENTO = False


def get_action(state, q_table, epsilon, env):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        return np.argmax(q_table[state])


if __name__ == "__main__":

    mapa = None
    q_table = {}
    env = None

    # ==========================================
    # CASO A: CARGAR MODELO EXISTENTE
    # ==========================================
    if not MODO_ENTRENAMIENTO and os.path.exists(NOMBRE_ARCHIVO):
        print(f"📂 Cargando cerebro y mapa desde '{NOMBRE_ARCHIVO}'...")

        with open(NOMBRE_ARCHIVO, "rb") as f:
            datos_guardados = pickle.load(f)

        mapa = datos_guardados["mapa"]       # Recuperamos el mapa original
        q_table = datos_guardados["q_table"]  # Recuperamos la inteligencia

        # Iniciamos el entorno con el mapa recuperado
        env = CityMultiAgentEnv(mapa, n_agentes=2)
        print("✅ ¡Carga completada! Saltando entrenamiento.")

    # ==========================================
    # CASO B: ENTRENAR DESDE CERO
    # ==========================================
    else:
        print("🏗️  Generando nueva ciudad (30x30) y comenzando entrenamiento...")
        mapa = generador.generate_map()
        env = CityMultiAgentEnv(mapa, n_agentes=2)
        q_table = {}

        episodes = 10000
        alpha = 0.1
        gamma = 0.95
        epsilon = 1.0
        epsilon_decay = 0.9996
        min_epsilon = 0.05

        print(f"🚦 Entrenando flota en mapa gigante (10,000 episodios)...")

        for i in range(episodes):
            states, _ = env.reset()
            done = False
            steps = 0

            while not done and steps < 600:
                actions = []
                # 1. Decidir
                for idx in range(env.n_agentes):
                    state = states[idx]
                    action = get_action(state, q_table, epsilon, env)
                    actions.append(action)

                # 2. Actuar
                next_states, rewards, terminated, _, _ = env.step(actions)
                done = terminated

                # 3. Aprender
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

                    q_table[state][action] = old_val + alpha * \
                        (reward + gamma * next_max - old_val)

                states = next_states
                steps += 1

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            if (i+1) % 1000 == 0:
                print(f"Episodio {i+1} | Epsilon: {epsilon:.2f}")

        print("✅ Entrenamiento completado.")

        # --- GUARDAR TODO AL FINALIZAR ---
        print(f"💾 Guardando modelo en '{NOMBRE_ARCHIVO}'...")
        paquete_datos = {
            "mapa": mapa,      # Guardamos el mapa para que coincida con lo aprendido
            "q_table": q_table  # Guardamos la tabla Q
        }
        with open(NOMBRE_ARCHIVO, "wb") as f:
            pickle.dump(paquete_datos, f)
        print("✅ Guardado exitoso.")

    # ==========================================
    # DEMOSTRACIÓN (Común para ambos casos)
    # ==========================================
    print("\n🚚 MOSTRANDO FLOTA...")
    # Si acabas de cargar, esto será instantáneo.
    input("Presiona Enter para ver la simulación...")

    states, _ = env.reset()
    done = False
    pasos = 0

    # Epsilon 0 para que actúen como expertos (sin explorar)
    epsilon_demo = 0

    while not done and pasos < 800:
        env.render()

        actions = []
        for idx in range(env.n_agentes):
            state = states[idx]
            # Usamos get_action con epsilon 0 (siempre elige la mejor opción)
            # Ojo: si cargaste un modelo, q_table ya tiene toda la info
            if state in q_table:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()
            actions.append(action)

        states, rewards, terminated, _, _ = env.step(actions)
        done = terminated
        pasos += 1

        import time
        time.sleep(0.02)

        if terminated:
            env.render()
            print(f"\n🏆 ¡Trabajo completado en {pasos} pasos!")
            plt.ioff()
            plt.show()
