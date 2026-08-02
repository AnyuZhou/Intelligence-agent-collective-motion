% Anyu Zhou, Nanjing University

% Code for paper "Emergence of collective motion of intelligent agents with different intelligence levels: Insights via a reinforcement learning framework"

These codes are developed in Python.

This repository contains the simulation code and analysis tools for our study on intelligence-driven collective motion in active agent systems. We introduce a tunable intelligence parameter (*k*) into a reinforcement learning framework to systematically investigate how individual cognitive capacity—specifically, the ability for value estimation and action selection—governs emergent collective behavior.

Our key finding reveals a sharp disorder-to-order phase transition at a surprisingly moderate intelligence threshold (*k* ≈ 0.5). Below this threshold, agents exhibit random motion and the system remains in a disordered state; above it, agents spontaneously develop aggregation policies and self-organize into ordered hexagonal structures—without explicit attractive interactions or structural constraints.

learn.py handles the learning and optimization of agent policies.

simulation.py loads the trained policy and performs multi-agent dynamics simulations.

plot.py handles visualization and quantitative analysis of simulation results. 
