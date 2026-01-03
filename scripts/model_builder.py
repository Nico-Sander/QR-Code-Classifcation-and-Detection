import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers

def build_model_from_config(config):
    """
    Constructs a compiled Keras model based on the provided configuration dictionary.
    """
    model_cfg = config['model']
    train_cfg = config['train']
    
    # 1. Input Layer
    # We explicitly define input shape to catch dimension errors early
    input_shape = tuple(model_cfg['input_shape'])
    inputs = Input(shape=input_shape)
    x = inputs

    # 2. Iterate through layers defined in YAML
    # This allows for arbitrary depths and architectures
    for i, layer_def in enumerate(model_cfg['layers']):
        l_type = layer_def['type'].lower()

        # --- CONVOLUTIONAL LAYERS ---
        if l_type == 'conv':
            x = layers.Conv2D(
                filters=layer_def['filters'],
                kernel_size=layer_def['kernel_size'],
                strides=layer_def.get('stride', 1),
                padding=layer_def.get('padding', 'same'),
                activation=layer_def.get('activation', 'relu'),
                name=f"conv_{i}"
            )(x)
            
            if layer_def.get('batch_norm', False):
                x = layers.BatchNormalization(name=f"bn_{i}")(x)

        # --- POOLING LAYERS ---
        elif l_type == 'max_pool':
            x = layers.MaxPooling2D(
                pool_size=layer_def.get('pool_size', 2),
                name=f"max_pool_{i}"
            )(x)
            
        elif l_type == 'avg_pool':
            x = layers.AveragePooling2D(
                pool_size=layer_def.get('pool_size', 2),
                name=f"avg_pool_{i}"
            )(x)

        elif l_type == 'global_avg_pool':
            x = layers.GlobalAveragePooling2D(name=f"global_avg_{i}")(x)

        # --- FLATTEN / RESHAPE ---
        elif l_type == 'flatten':
            x = layers.Flatten(name=f"flatten_{i}")(x)

        # --- DENSE LAYERS ---
        elif l_type == 'dense':
            x = layers.Dense(
                units=layer_def['units'],
                activation=layer_def.get('activation', 'relu'),
                name=f"dense_{i}"
            )(x)
            
            if layer_def.get('batch_norm', False):
                x = layers.BatchNormalization(name=f"bn_dense_{i}")(x)

        # --- REGULARIZATION ---
        elif l_type == 'dropout':
            x = layers.Dropout(
                rate=layer_def['rate'],
                name=f"dropout_{i}"
            )(x)
            
        else:
            print(f"⚠️ Warning: Unknown layer type '{l_type}' at index {i}. Skipping.")

    # 3. Create Model Object
    model = models.Model(inputs=inputs, outputs=x, name=config['project']['run_name'])

    # 4. Compile Model
    # We dynamically select the optimizer based on the string name
    opt_name = train_cfg['optimizer'].lower()
    lr = train_cfg['learning_rate']

    if opt_name == 'adam':
        opt = optimizers.Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        opt = optimizers.SGD(learning_rate=lr)
    elif opt_name == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=lr)
    else:
        print(f"⚠️ Warning: Unknown optimizer '{opt_name}'. Defaulting to Adam.")
        opt = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss=train_cfg['loss'],
        metrics=train_cfg['metrics']
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