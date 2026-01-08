import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers

def build_model_from_config(config):
    """
    Builds a Keras model based on the provided configuration dict.
    """
    model_config = config['model']
    input_shape = model_config['input_shape']
    
    model = models.Sequential()
    
    # 1. Explicit Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # 2. Optional: Rescaling (Best practice to normalize inputs to 0-1)
    # If your images are 0-255, this makes training much more stable.
    # We add this check to be safe.
    model.add(layers.Rescaling(1./255)) 

    # 3. Iterate through layers
    for layer_cfg in model_config['layers']:
        layer_type = layer_cfg['type']
        
        # --- PREPROCESSING / AUGMENTATION LAYERS ---
        if layer_type == 'random_flip':
            model.add(layers.RandomFlip(mode=layer_cfg['mode']))
            
        elif layer_type == 'random_rotation':
            model.add(layers.RandomRotation(factor=layer_cfg['factor']))
            
        elif layer_type == 'random_zoom':
            model.add(layers.RandomZoom(
                height_factor=layer_cfg['height_factor'], 
                width_factor=layer_cfg['width_factor']
            ))
            
        elif layer_type == 'random_translation':
            model.add(layers.RandomTranslation(
                height_factor=layer_cfg['height_factor'], 
                width_factor=layer_cfg['width_factor']
            ))
            
        elif layer_type == 'gaussian_noise':
            # Adds noise to inputs (simulates grain/artifacts)
            model.add(layers.GaussianNoise(stddev=layer_cfg['stddev']))
            
        elif layer_type == 'random_contrast':
            model.add(layers.RandomContrast(factor=layer_cfg['factor']))
            
        # --- STANDARD CNN LAYERS ---
        elif layer_type == 'conv':
            model.add(layers.Conv2D(
                filters=layer_cfg['filters'],
                kernel_size=layer_cfg['kernel_size'],
                strides=layer_cfg.get('stride', 1), # Default to 1 if not specified
                padding=layer_cfg.get('padding', 'same'),
                activation=layer_cfg['activation']
            ))
            if layer_cfg.get('batch_norm', False):
                model.add(layers.BatchNormalization())
                
        elif layer_type == 'max_pool':
            model.add(layers.MaxPooling2D(pool_size=layer_cfg['pool_size']))
            
        elif layer_type == 'avg_pool':
            model.add(layers.AveragePooling2D(pool_size=layer_cfg['pool_size']))
            
        elif layer_type == 'global_avg_pool':
            model.add(layers.GlobalAveragePooling2D())
            
        elif layer_type == 'flatten':
            model.add(layers.Flatten())
            
        elif layer_type == 'dense':
            model.add(layers.Dense(
                units=layer_cfg['units'],
                activation=layer_cfg['activation']
            ))
            
        elif layer_type == 'dropout':
            model.add(layers.Dropout(rate=layer_cfg['rate']))
            
        else:
            print(f"⚠️ Warning: Unknown layer type '{layer_type}' skipped.")

    # 4. Compile the model
    # We compile here so the notebook stays clean
    opt_name = config['train']['optimizer']
    learning_rate = config['train']['learning_rate']
    
    if opt_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
    model.compile(
        optimizer=optimizer,
        loss=config['train']['loss'],
        metrics=config['train']['metrics']
    )
    
    return model

if __name__ == "__main__":
    # Quick Test: If run directly, load config and print summary
    import sys
    from pathlib import Path
    import yaml
    
    # Add project root to path to find project_paths
    sys.path.append(str(Path(__file__).parent))
    from project_paths import CONFIG_DIR
    
    cfg_path = CONFIG_DIR / "model_config.yaml"
    
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print(f"Building model for: {cfg['project']['name']}")
        model = build_model_from_config(cfg)
        model.summary()
        print("\n✅ Model built successfully from config!")
    else:
        print(f"❌ Config not found at {cfg_path}")