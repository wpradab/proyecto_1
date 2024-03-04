import tensorflow_transform as tft
 
# Define las constantes para los nombres de las características de entrada
FEATURE_NAMES = ['Soil_Type_C2703', 'Soil_Type_C2704','Soil_Type_C2705','Soil_Type_C2717', 'Soil_Type_C4703','Soil_Type_C8771', 'Soil_Type_C8772','Wilderness_Area_Rawah']
 
# Nombre de la característica objetivo
TARGET_FEATURE_NAME = 'Cover_Type'
 
def preprocessing_fn(inputs):
    """
    Función de preprocesamiento que se aplicará a los datos de entrada.
 
    Args:
        inputs: Un diccionario donde las claves son nombres de características y los valores son tensores
                con los valores de esas características.
 
    Returns:
        Un diccionario donde las claves son nombres de características y los valores son tensores
        con los valores transformados de esas características.
    """
    outputs = {}
 
    # Aplica MinMax scaling a las características numéricas
    for feature_name in FEATURE_NAMES:
        # tft.scale_to_0_1 escala los valores numéricos al rango [0, 1]
        outputs[feature_name] = tft.scale_to_0_1(inputs[feature_name])
    # Simplemente copia la característica objetivo al diccionario de salida
    outputs[TARGET_FEATURE_NAME] = inputs[TARGET_FEATURE_NAME]
