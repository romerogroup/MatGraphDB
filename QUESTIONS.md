# Questions 

1. Models like crabnet, matgl, and automatminer predict a specfic properties. Struture or Compositions embeddings can be extracted from the network for a partiular property. Could a training schedulue be developed to train on all properties at once? Start with training E_bandgap, replace output network for predicting property while keeping the encoder the same and trainable, train on bulk modulus, repeat last two steps for all properties.

2. On the same line as question one is it possible to train on all properties at once. For example in qm9 they have 9 properties for the same dataset. 

2. Mat2vec converts atomic compositions and text to an encoding vector much like LLMs. Is it possible to encode structure information in the same way

4. In what situations would you want to have eqivarince 