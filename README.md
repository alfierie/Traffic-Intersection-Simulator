# 🚦 Traffic Intersection Stochastic Simulation
**Stochastic Modeling & Simulation Project — Universitas Gadjah Mada**

---

## 📌 Problem Description
A 4-way signalized intersection is modeled stochastically.
Vehicles arrive following a **Poisson process** (exponential inter-arrival times).
When an **ambulance** is detected in any queue, an **Emergency Preemption** system
immediately forces the signal green for that direction until the ambulance clears.

---

## 🧮 Stochastic Model

| Component           | Distribution          | Parameter         |
|---------------------|-----------------------|-------------------|
| Vehicle arrivals    | Poisson / Exponential | λ (vehicles/sec)  |
| Service (crossing)  | Exponential           | μ (mean svc time) |
| Ambulance presence  | Bernoulli             | p (probability)   |
| Phase durations     | Deterministic + override | g_NS, g_EW (s) |

**Queue model:** M/M/1 per direction, with preemptive priority for ambulances.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🎛 Interactive Parameters

| Parameter | Description |
|-----------|-------------|
| Arrival rates (N/S/E/W) | Poisson rate per lane (vehicles/second) |
| Green NS / EW phase | Duration of each signal phase (seconds) |
| Ambulance probability | Fraction of vehicles that are ambulances |
| Preemption duration | How long green is held for emergency direction |
| Simulation duration | Total simulated time (seconds) |
| Mean service time | Average time for one vehicle to cross (seconds) |
| Random seed | Reproducibility control |

---

## 📊 Outputs
- **Intersection snapshot** — live view of queues and signal state
- **Queue length over time** — per-direction time series with emergency shading
- **Signal phase timeline** — visual of NS/EW green/red periods
- **Wait time distribution** — histogram for cars vs. ambulances
- **Sensitivity analysis** — ambulance probability vs. mean wait time sweep
- **CSV export** — queue history and vehicle log

---

## 📝 Notion Article Structure (suggested)

1. **Background** — traffic congestion, emergency vehicle delays in Indonesia
2. **System Modeling** — Poisson arrivals, M/M/c queue, preemption theory
3. **System Design & Assumptions** — 4-way, single intersection, one ambulance preempts at a time
4. **Simulation & Sensitivity Analysis** — vary λ, ambulance probability, green durations
5. **Results & Insights** — average wait reduction for ambulances, impact on car queues
6. **Conclusion** — effectiveness of signal preemption, recommendations

---

## 👥 Group Members
*(Fill in your group members here)*
- 
- 
- 
