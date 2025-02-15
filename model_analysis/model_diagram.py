import tensorflow as tf
from tensorflow.keras.utils import plot_model # type: ignore

def save_model_diagram(model_path, diagram_filename="model_diagram.png"):
    """
    Loads a Keras model and saves a diagram of its architecture.
    
    Args:
        model_path (str): Path to the .keras file.
        diagram_filename (str): Name of the output image file 
                                (can be .png or .pdf).
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create a visual diagram of the model architecture
    # - show_shapes=True: includes tensor shapes in the diagram
    # - show_layer_names=True: includes each layer's name
    # - rankdir="TB": 'TB' (top-bottom) or 'LR' (left-right) layout
    plot_model(
        model, 
        to_file=diagram_filename, 
        show_shapes=True, 
        show_layer_names=True,
        rankdir="LR",
        dpi=300
    )
    print(f"Saved architecture diagram to {diagram_filename}")

if __name__ == "__main__":
    base_model_path = "../trained_models/base_model.keras"
    save_model_diagram(base_model_path, "base_model_diagram.png")
    
    mc_dropout_model_path = "../trained_models/dropout_model.keras"
    save_model_diagram(mc_dropout_model_path, "mc_dropout_diagram.png")
    
    ensemble_model_path = "../trained_models/ensemble_model_1.keras"
    save_model_diagram(ensemble_model_path, "ensemble_model_1_diagram.png")