import numpy as np
import random
import matplotlib.pyplot as plt
import generador
from entorno import CityMultiAgentEnv


def get_action(state, q_table, epsilon, env):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        return np.argmax(q_table[state])


if __name__ == "__main__":

    print("🏗️  Construyendo ciudad (30x30) con tráfico fijo y sistema logístico...")
    mapa = generador.generate_map()

    # 2 Agentes Repartidores
    env = CityMultiAgentEnv(mapa, n_agentes=2)

    q_table = {}

    # EPISODIOS ALTOS (El mapa es muy grande)
    episodes = 10000  # cambiar para

    alpha = 0.1
    gamma = 0.95

    epsilon = 1.0
    epsilon_decay = 0.9996
    min_epsilon = 0.05

    print(f"🚦 Entrenando flota en mapa gigante...")

    for i in range(episodes):
        states, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 600:  # Más pasos permitidos
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

                q_table[state][action] = old_val + alpha * \
                    (reward + gamma * next_max - old_val)

            states = next_states
            steps += 1

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        if (i+1) % 1000 == 0:
            print(f"Episodio {i+1} | Epsilon: {epsilon:.2f}")

    print("✅ Entrenamiento completado.")

    # DEMOSTRACIÓN
    print("\n🚚 MOSTRANDO FLOTA...")
    input("Presiona Enter...")

    states, _ = env.reset()
    done = False
    pasos = 0
    epsilon = 0

    while not done and pasos < 800:
        env.render()

        actions = []
        for idx in range(env.n_agentes):
            state = states[idx]
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
