import numpy as np

# General parameters
NUM_QUBITS = 6
NUM_LAYERS = 3
LEAK_RATE = 0.7

# "linear"   = chain: 0-1-2-3-...
# "ring"     = chain + last-to-first connection
# "all_to_all" = every qubit connected to every other qubit
TOPOLOGY = "linear"

# Number of qubits receiving the input
# "all" = all qubits(8 by default)
# Integer = first N qubits
INPUT_QUBITS = "all"


# "z" = <Z> for every qubit
# "z+zz"
READOUT = "z"

# Readout training
RIDGE_ALPHA = 1e-4



TIME_SERIES = "audio_medieval"
# options
# TIME_SERIES = "mackey"
# TIME_SERIES = "audio_medieval"

RESULT_MODE = "average"
# "average"   = show mean ± standard deviation
# "individual" = show every seed separately

#fixed inputs
NUM_MEASUREMENTS = 128
WASHOUT = 30
MAX_SEQUENCE_LENGTH = 1000
TRAIN_SIZE = 700
TEST_SIZE = 250
# How strongly the input is encoded
INPUT_SCALE = np.pi
#Memory
MEMORY_STRENGTH = 0.5
# Spin specific
EVOLUTION_TIME = 1.0
#Seeds
SEEDS = [1]


#offset {-2,+2} theoretically however I only use 0 and +1
PREDICTION_OFFSET = 0


#BASELINE
# General parameters

#NUM_QUBITS = 8

#WASHOUT = 30
#NUM_LAYERS = 3


#TOPOLOGY = "linear"



#INPUT_QUBITS = All



#LEAK_RATE = 0.5

#READOUT = "z"

#audio not needed anymore
#AUDIO_TIME_SCALE = 1.0


#TIME_SERIES = "audio_medieval"
# options
# TIME_SERIES = "mackey"
# TIME_SERIES = "audio_medieval"

#RESULT_MODE = "individual"
# "average"   = show mean ± standard deviation
# "individual" = show every seed separately

#fixed inputs
#NUM_MEASUREMENTS = 128
#MAX_SEQUENCE_LENGTH = 1000
#TRAIN_SIZE = 700
#TEST_SIZE = 250
# How strongly the input is encoded
#INPUT_SCALE = np.pi
#MEMORY_STRENGTH = 0.5
# Spin specific
#EVOLUTION_TIME = 1.0
#SEEDS = [1, 2, 3,4,5]


#offset {-2,+2} theoretically however I only use 0 and +1
#PREDICTION_OFFSET = 0
