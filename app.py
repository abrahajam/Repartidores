import streamlit as st
import numpy as np
import time
import folium
from streamlit_folium import st_folium
import pandas as pd
import altair as alt
from datetime import datetime

# Importamos tus módulos
import generador
from entorno import CityMultiAgentEnv, STATE_RETURNING, STATE_URGENT_DELIVERY, STATE_IDLE, STATE_DELIVERING
import entrenar

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    layout="wide", 
    page_title="Logística Urbana AI",
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS (Ajustados a Tema Claro) ---
st.markdown("""
<style>
    /* Estilo del Terminal de Logs (Consola oscura para contraste) */
    .terminal-container {
        background-color: #000000;
        color: #00ff00;
        border: 2px solid #333;
        border-radius: 5px;
        padding: 10px;
        height: 350px;
        overflow-y: auto;
        font-family: 'Courier New', Courier, monospace;
        font-size: 13px;
        line-height: 1.4;
    }
    .log-entry { margin-bottom: 2px; }
    .log-time { color: #888; margin-right: 8px; font-size: 0.9em; }
    .log-urgent { color: #ff00ff; font-weight: bold; } /* Morado neón */
    .log-warn { color: #ffff00; } /* Amarillo */
    
    /* Ajustes generales */
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE MAPA Y LEYENDA ---
BASE_LAT = 40.4168
BASE_LON = -3.7038
SCALE = 0.0003 

def grid_to_latlon(row, col, max_rows):
    real_row = max_rows - row 
    lat = BASE_LAT + (real_row * SCALE)
    lon = BASE_LON + (col * SCALE)
    return lat, lon

def add_legend(map_obj):
    legend_html = """
     <div style="position: fixed; bottom: 30px; left: 30px; width: 170px; height: 320px; 
     border:2px solid grey; z-index:9999; font-size:12px; background-color:white; opacity: 0.9; padding: 10px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
     <b>🚚 Estado Agentes</b><br>
     <i class="fa fa-truck" style="color:gray"></i>&nbsp; Libre / Esperando<br>
     <i class="fa fa-truck" style="color:green"></i>&nbsp; Repartiendo<br>
     <i class="fa fa-truck" style="color:orange"></i>&nbsp; Volviendo a Base<br>
     <i class="fa fa-truck" style="color:purple"></i>&nbsp; URGENTE ⚡<br>
     <hr style="margin: 5px 0;">
     <b>📦 Mapa y Pedidos</b><br>
     <i class="fa fa-box" style="color:blue"></i>&nbsp; Paquete Normal<br>
     <i class="fa fa-bolt" style="color:red"></i>&nbsp; Paquete Urgente<br>
     <i class="fa fa-home" style="color:black"></i>&nbsp; Almacén Central<br>
     <hr style="margin: 5px 0;">
     <b>📈 Rutas</b><br>
     <span style="color:darkblue"><b>━</b></span> Ruta Agente 0<br>
     <span style="color:darkgreen"><b>━</b></span> Ruta Agente 1
     </div>
     """
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def crear_mapa_folium(mapa_grid, agentes, tareas):
    rows, cols = mapa_grid.shape
    # Volvemos al mapa claro "Positron" que es muy limpio
    m = folium.Map(location=[BASE_LAT + (rows*SCALE)/2, BASE_LON + (cols*SCALE)/2], 
                   zoom_start=15, tiles="CartoDB positron", zoom_control=False)

    # Base Central
    start_lat, start_lon = grid_to_latlon(0, 0, rows)
    folium.Marker(
        [start_lat, start_lon], 
        tooltip="Central Logística", 
        icon=folium.Icon(color="black", icon="home", prefix="fa")
    ).add_to(m)

    # Tráfico
    for r in range(rows):
        for c in range(cols):
            val = mapa_grid[r, c]
            lat, lon = grid_to_latlon(r, c, rows)
            if val == generador.ROAD_JAM:
                folium.CircleMarker([lat, lon], radius=5, color="red", fill=True, fill_opacity=0.6, tooltip="Atasco").add_to(m)
            elif val == generador.ROAD_SLOW:
                folium.CircleMarker([lat, lon], radius=5, color="orange", fill=True, fill_opacity=0.6, tooltip="Lento").add_to(m)

    # Tareas/Paquetes (MEJORADO: Ahora son Iconos GRANDES)
    for t in tareas:
        if not t['asignado']:
            lat, lon = grid_to_latlon(t['pos'][0], t['pos'][1], rows)
            
            if t['urgente']:
                # Icono Rayo Rojo para urgentes
                folium.Marker(
                    [lat, lon], 
                    tooltip="¡URGENTE!", 
                    icon=folium.Icon(color="red", icon="bolt", prefix="fa")
                ).add_to(m)
            else:
                # Icono Caja Azul para normales (antes era un puntito)
                folium.Marker(
                    [lat, lon], 
                    tooltip="Paquete Pendiente", 
                    icon=folium.Icon(color="blue", icon="box", prefix="fa")
                ).add_to(m)

    # Agentes
    colors_route = ['darkblue', 'darkgreen', 'purple']
    for i, ag in enumerate(agentes):
        lat, lon = grid_to_latlon(ag['pos'][0], ag['pos'][1], rows)
        
        # Ruta histórica
        if len(ag['route_history']) > 1:
            route_points = [grid_to_latlon(r, c, rows) for r, c in ag['route_history'][-20:]]
            folium.PolyLine(route_points, color=colors_route[i % len(colors_route)], weight=3, opacity=0.7).add_to(m)
        
        # Icono Agente
        icon_color = 'gray'
        status_txt = "Libre"
        
        if ag['state'] == STATE_URGENT_DELIVERY: 
            icon_color = 'purple'; status_txt = "URGENTE"
        elif ag['state'] == STATE_RETURNING: 
            icon_color = 'orange'; status_txt = "Volviendo"
        elif ag['has_package']: 
            icon_color = 'green'; status_txt = "Repartiendo"
        
        folium.Marker(
            [lat, lon], 
            tooltip=f"Agente {ag['id']}: {status_txt}", 
            icon=folium.Icon(color=icon_color, icon="truck", prefix="fa")
        ).add_to(m)

    add_legend(m) # Restauramos la leyenda
    return m

# --- SIDEBAR: CONTROLES ---
with st.sidebar:
    st.title("🎛️ Panel de Control")
    
    st.subheader("Configuración")
    modo = st.radio("Modo", ["Monitorización", "Entrenamiento"], index=0)
    
    if modo == "Entrenamiento":
        episodes_slider = st.slider("Episodios", 1000, 10000, 3000, step=500)
        train_btn = st.button("🚀 Entrenar Nuevo Modelo", type="primary")
        
    st.divider()
    
    st.subheader("Simulación")
    velocidad = st.slider("Velocidad (segundos)", 0.1, 1.5, 0.3)
    max_steps = st.number_input("Pasos Máximos", 50, 500, 100)

# --- CARGA DE MODELO ---
mapa_cargado, q_table_cargado = entrenar.cargar_modelo()
if 'mapa' not in st.session_state:
    if mapa_cargado is not None:
        st.session_state['mapa'] = mapa_cargado
        st.session_state['q_table'] = q_table_cargado
    else:
        st.session_state['mapa'] = None

# --- LÓGICA PRINCIPAL ---

st.title("🚚 Centro de Control Logístico")

# RESTAURADO: Explicación del sistema
with st.expander("ℹ️ ¿Cómo funciona esto?", expanded=True):
    st.markdown("""
    **Simulación de Logística Autónoma con IA (Reinforcement Learning)**
    
    Estos camiones no siguen rutas fijas. Tienen un "cerebro" entrenado para:
    1.  🧠 **Navegar:** Aprenden el mapa y recuerdan dónde suele haber atascos (puntos rojos).
    2.  ⚡ **Priorizar:** Si entra un pedido **URGENTE (Rayo Rojo)**, abandonan su ruta actual para volver a la base.
    3.  👀 **Autonomía:** Toman decisiones paso a paso para maximizar el beneficio (Recompensa).
    """)

if modo == "Entrenamiento":
    st.header("Entrenamiento del Agente")
    if train_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, eps, metric):
            progress_bar.progress(progress)
            status_text.text(f"Episodio: {eps} | Epsilon (Exploración): {metric:.4f}")
            
        # Nota: Asegúrate de que entrenar.py acepte el callback, si no, quita el callback=...
        mapa, q_table = entrenar.entrenar_agentes(episodes_slider, callback=update_progress)
        
        st.session_state['mapa'] = mapa
        st.session_state['q_table'] = q_table
        st.success("¡Entrenamiento finalizado!")
        time.sleep(1)
        st.rerun()

elif modo == "Monitorización":
    if st.session_state['mapa'] is None:
        st.warning("⚠️ No hay modelo entrenado. Ve a la pestaña 'Entrenamiento'.")
        st.stop()
        
    # --- INTERFAZ DE MONITOREO ---
    
    # 1. KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    metric_delivered = kpi1.empty()
    metric_profit = kpi2.empty()
    metric_urgent = kpi3.empty()
    
    # Inicialización
    metric_delivered.metric("📦 Entregas", "0")
    metric_profit.metric("💰 Beneficio", "0")
    metric_urgent.metric("⚡ Urgentes Activos", "0")

    # 2. LAYOUT MAPA + DATOS
    col_mapa, col_datos = st.columns([2, 1])
    
    with col_datos:
        st.markdown("### 📊 Rendimiento")
        chart_placeholder = st.empty()
        
        st.markdown("### 📟 Terminal de Operaciones")
        terminal_placeholder = st.empty()
        
        # Espacio para botones
        btn_placeholder = st.empty()

    with col_mapa:
        map_placeholder = st.empty()

    # --- CONTROL DE ESTADO ---
    if 'simulando' not in st.session_state:
        st.session_state['simulando'] = False

    def stop_simulation():
        st.session_state['simulando'] = False

    # Botones
    if not st.session_state['simulando']:
        if btn_placeholder.button("▶️ INICIAR SIMULACIÓN", type="primary", use_container_width=True):
            st.session_state['simulando'] = True
            st.rerun()
    else:
        btn_placeholder.button("⏹️ DETENER", on_click=stop_simulation, type="secondary", use_container_width=True)

    # --- BUCLE DE SIMULACIÓN ---
    if st.session_state['simulando']:
        env = CityMultiAgentEnv(st.session_state['mapa'], n_agentes=2)
        states, _ = env.reset()
        q_table = st.session_state['q_table']
        
        total_reward = 0
        chart_data = pd.DataFrame(columns=["Paso", "Beneficio"])
        log_lines = []
        
        for paso in range(max_steps):
            if not st.session_state['simulando']: break
                
            # Decisiones IA
            actions = []
            for idx in range(env.n_agentes):
                state = states[idx]
                if state in q_table:
                    action = np.argmax(q_table[state])
                else:
                    action = env.action_space.sample()
                actions.append(action)

            # Paso del entorno
            states, rewards, terminated, _, info = env.step(actions)
            total_reward += sum(rewards)
            
            # Actualizar KPIs
            total_deliveries = sum(a['deliveries_count'] for a in env.agents)
            urgentes_activos = sum(1 for t in env.tasks if t['urgente'])
            
            metric_delivered.metric("📦 Entregas", f"{total_deliveries}")
            metric_profit.metric("💰 Beneficio", f"{total_reward:.1f}")
            metric_urgent.metric("⚡ Urgentes Activos", f"{urgentes_activos}", 
                               delta="¡ALERTA!" if urgentes_activos > 0 else "Normal",
                               delta_color="inverse")

            # Gráfico (Altair limpio)
            new_row = pd.DataFrame({"Paso": [paso], "Beneficio": [total_reward]})
            chart_data = pd.concat([chart_data, new_row], ignore_index=True)
            
            c = alt.Chart(chart_data).mark_line(color='#2563eb').encode(
                x='Paso', y='Beneficio'
            ).properties(height=200)
            chart_placeholder.altair_chart(c, use_container_width=True)

            # Terminal
            timestamp = datetime.now().strftime("%H:%M:%S")
            for log in info.get("logs", []):
                css = ""
                if "URGENTE" in log: css = "log-urgent"
                elif "Atasco" in log: css = "log-warn"
                
                log_lines.insert(0, f"<div class='log-entry'><span class='log-time'>[{timestamp}]</span><span class='{css}'>{log}</span></div>")
            
            if len(log_lines) > 15: log_lines = log_lines[:15]
            terminal_html = f"<div class='terminal-container'>{''.join(log_lines)}</div>"
            terminal_placeholder.markdown(terminal_html, unsafe_allow_html=True)

            # Mapa
            with map_placeholder:
                m = crear_mapa_folium(env.current_map, env.agents, env.tasks)
                st_folium(m, height=600, width=None, key=f"map_{paso}", returned_objects=[])

            if terminated:
                st.balloons()
                st.success("✅ Simulación completada con éxito.")
                st.session_state['simulando'] = False
                break
                
            time.sleep(velocidad)