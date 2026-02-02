Rocket PID Landing Controller (Educational 2D Simulation)

Educational 2D rocket landing simulation with a cascaded controller:

Lateral guidance: x, vx → θ_des (PD)

Attitude control: θ_des → gimbal (PID)

Vertical control: v_z,des(z) → throttle (PID → accel → thrust)

Includes practical PID details: output saturation, conditional anti-windup, derivative-on-measurement with low-pass filtering, and dt guards. Simulation includes wind gust and sensor noise. Logs are plotted for position, velocity, attitude, control signals, and mass.

How to run:

pip install -r requirements.txt

python rocket_pid.py
