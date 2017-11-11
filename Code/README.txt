Code for multilayer spiking network learning rule that classifies input patterns using multi-spike target output trains.

Example usage: Record = Main(1, 2, 3, 100, 10, 2, 1000)
Runs a simulation with a random initialization and setup: 
 - 1 input pattern per class
 - 2 classes
 - 3 spikes in each target output spike train
 - 100 input neurons
 - 10 hidden neurons
 - 2 output neurons
 - 1000 learning episodes
 
Optimised code, approximating a sharp firing threshold for a single output neuron, can be run using:
Record = MainSingle(1, 2, 3, 100, 10, 1000)
