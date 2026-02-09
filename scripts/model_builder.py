import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_model_from_config(config):
    """
    Builds a Keras model based on config, ensuring correct order:
    Input -> Augmentation -> Adapter -> Base Model -> Head
    """
    model_conf = config['model']
    input_shape = tuple(model_conf['input_shape'])
    arch_type = model_conf.get('type', 'custom') 
    
    # Define what counts as an augmentation layer
    AUGMENTATION_TYPES = [
        'random_flip', 'random_rotation', 'random_zoom', 
        'random_translation', 'gaussian_noise', 'random_contrast'
    ]

    model = models.Sequential()
    
    # 1. Input & Rescaling
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1./255))
    
    # 2. PHASE 1: Add Augmentation Layers FIRST
    # These must operate on the raw image (1 channel)
    for layer_cfg in model_conf['layers']:
        if layer_cfg['type'] in AUGMENTATION_TYPES:
            _add_layer(model, layer_cfg)

    # 3. PHASE 2: Adapter & Base Model (Transfer Learning Only)
    if arch_type == "transfer":
        # A) Adapter: Convert 1 Channel -> 3 Channels
        # Use bias=False because BatchNorm usually follows in modern nets, 
        # but here it feeds into a pre-trained net which expects standardized input. 
        # A simple conv is fine.
        if input_shape[-1] == 1:
            model.add(layers.Conv2D(3, (3, 3), padding='same', use_bias=False, name="grayscale_adapter"))
        
        # B) Base Model
        base_name = model_conf['transfer']['base_model']
        freeze = model_conf['transfer']['freeze_base']
        
        # Input to base model is now effectively 3 channels
        base_input_shape = (input_shape[0], input_shape[1], 3)
        
        if base_name == "MobileNetV3Small":
            base_model = tf.keras.applications.MobileNetV3Small(
                input_shape=base_input_shape, include_top=False, weights='imagenet'
            )
        elif base_name == "MobileNetV2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=base_input_shape, include_top=False, weights='imagenet'
            )
        elif base_name == "EfficientNetB0":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=base_input_shape, include_top=False, weights='imagenet'
            )
        elif base_name == "ResNet50V2":
             base_model = tf.keras.applications.ResNet50V2(
                input_shape=base_input_shape, include_top=False, weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base_model: {base_name}")

        base_model.trainable = not freeze
        model.add(base_model)
        print(f"✅ Loaded {base_name} (Trainable: {not freeze})")

    # 4. PHASE 3: Add Remaining Layers (Head or Custom Body)
    for layer_cfg in model_conf['layers']:
        l_type = layer_cfg['type']
        
        # Skip augmentation (already added)
        if l_type in AUGMENTATION_TYPES:
            continue
            
        # Skip custom conv blocks if we are in transfer mode
        if arch_type == "transfer" and l_type in ['conv', 'max_pool']:
            continue
            
        # Add the layer
        _add_layer(model, layer_cfg)

    # 5. Compile
    optimizer_config = config['train']
    lr = float(optimizer_config['learning_rate'])
    
    if optimizer_config['optimizer'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_config['optimizer'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
        
    model.compile(
        optimizer=opt,
        loss=optimizer_config['loss'],
        metrics=optimizer_config['metrics']
    )
    
    return model

def _add_layer(model, layer_cfg):
    """Helper to add a single layer based on config dictionary."""
    layer_type = layer_cfg['type']
    
    if layer_type == 'random_flip':
        model.add(layers.RandomFlip(mode=layer_cfg['mode']))
    elif layer_type == 'random_rotation':
        model.add(layers.RandomRotation(factor=layer_cfg['factor']))
    elif layer_type == 'gaussian_noise':
        model.add(layers.GaussianNoise(stddev=layer_cfg['stddev']))
    elif layer_type == 'random_contrast':
        model.add(layers.RandomContrast(factor=layer_cfg['factor']))
    elif layer_type == 'random_zoom':
        model.add(layers.RandomZoom(height_factor=layer_cfg['height_factor']))
    elif layer_type == 'random_translation':
        model.add(layers.RandomTranslation(height_factor=layer_cfg['height_factor'], width_factor=layer_cfg['width_factor']))
        
    elif layer_type == 'conv':
        model.add(layers.Conv2D(
            filters=layer_cfg['filters'],
            kernel_size=layer_cfg['kernel_size'],
            padding=layer_cfg.get('padding', 'same'),
            activation=layer_cfg['activation']
        ))
        if layer_cfg.get('batch_norm', False):
            model.add(layers.BatchNormalization())
            
    elif layer_type == 'max_pool':
        model.add(layers.MaxPooling2D(pool_size=layer_cfg['pool_size']))
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

if __name__ == "__main__":
    from pathlib import Path
    import yaml
    cfg_path = Path("/home/nico/workspace/github.com/Nico-Sander/KI-Project-WS2526/runs/custom/run_07_dataset_roboflow_grayscale_20260205_171521/config.yaml")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")
    # Read the config file into a dictionary
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = build_model_from_config(config=config)
    print(model.summary())