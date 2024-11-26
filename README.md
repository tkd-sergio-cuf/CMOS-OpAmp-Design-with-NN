# CMOS-OpAmp-Design-with-NN
Code and data for the automated design of CMOS operational amplifiers using neural networks. Includes Python scripts, training datasets, and results for two topologies: folded cascode and telescopic cascode.

## Description
Designing analog integrated circuits, such as operational amplifiers, often requires significant time and effort due to the complexity of manually adjusting interdependent parameters. This project proposes a novel solution by employing neural networks to automate the design process, reducing development time while maintaining high accuracy.

The neural networks take performance parameters (e.g., gain, CMRR, PSRR) as input and output the corresponding transistor dimensions and bias voltages. The system was trained and evaluated using datasets generated from parametric simulations.

## Requirements
To run the scripts in this repository, ensure the following:
- Python 3.8 or higher
- Libraries: TensorFlow, Keras-Tuner, pandas, NumPy, scikit-learn, Matplotlib


## Authors
- Sergio Andrés Muñoz Cufiño
- Silvana Ferro Durán

