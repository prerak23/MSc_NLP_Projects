# This folder
This folder is where all the class files for the pytorch model are stored.

# The model
*no description yet*

# The argument separation
Most elements in the model have a "Args" class defined with them.

The goal is to increase code readability by grouping the parameters needed to build an object and the computations
necessary to use those parameters together.

This leaves only the internal mechanisms of the object in the class definition.

This also allows faster maintenance when the need to change the parameters arises, as what is transmitted through the 
layers of the model is an object instead of separated parameters.

Lastly it allows clear differentiation of which parameter is needed to build which element.
