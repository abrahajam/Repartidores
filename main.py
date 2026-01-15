import numpy as np
import random
import matplotlib.pyplot as plt
import generador
from entorno import CityTrafficLightsEnv

if __name__ == "__main__":

    print("🏗️  Generando ciudad con tráfico aleatorio...")
    mapa_creado = generador.generate_map()

    env = CityTrafficLightsEnv(mapa_generado=mapa_creado)

    q_table = {}

    episodes = 8000
    alpha = 0.2
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.9994
    min_epsilon = 0.05

    print(f"🚦 Entrenando IA (Adaptación a cambios de ruta)...")

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 250:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, _, _ = env.step(action)
            done = terminated

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = old_value + alpha * \
                (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value

            state = next_state
            steps += 1

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        if (i+1) % 1000 == 0:
            print(f"Episodio {i+1} completado. Epsilon: {epsilon:.2f}")

    print("✅ Entrenamiento completado.")

    # DEMOSTRACIÓN
    print("\n🚚 MOSTRANDO RUTA...")
    input("Presiona Enter...")

    state, _ = env.reset()
    done = False
    pasos = 0

    while not done and pasos < 300:
        env.render()

        if state in q_table:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        state, reward, terminated, _, _ = env.step(action)
        done = terminated
        pasos += 1

        if terminated:
            env.render()
            print(f"\n🏆 ¡Ruta completada en {pasos} pasos!")
            plt.ioff()
            plt.show()
