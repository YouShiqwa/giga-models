from pathlib import Path


data_path = '/vepfs-cnbje63de6fae220/chengy/code/mobile_pi/datasets/v1.0/target/composite/SetUpCuttingStation/20250817/lerobot'
computed_norm_stats_path = f'{data_path}/meta/giga_pi05_robocasa_norm_stats.json'
norm_stats_path = computed_norm_stats_path if Path(computed_norm_stats_path).is_file() else f'{data_path}/meta/stats.json'

data_or_config = [
    dict(
        _class_name='RoboCasaLeRobotDataset',
        data_path=data_path,
        delta_info=dict(action=50),
        meta_name='meta',
    )
]

config = dict(
    runners=['pi0.Pi0Trainer'],
    project_dir='./experiments/vla/pi05/robocasa_set_up_cutting_station',
    launch=dict(
        gpu_ids=[4,5,6,7],
        distributed_type='FSDP',
        fsdp_config=dict(
            fsdp_version='2',
            fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
            fsdp_transformer_layer_cls_to_wrap='SiglipEncoderLayer,GemmaDecoderLayerWithExpert',
            fsdp_cpu_ram_efficient_loading='false',
            fsdp_state_dict_type='FULL_STATE_DICT',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            # 8 GPUs x 2 samples = OpenPI reference global batch size 16.
            batch_size_per_gpu=16,
            num_workers=16,
            transform=dict(
                type='RoboCasaPi05Transform',
                norm_stats_path=norm_stats_path,
                use_quantiles=False,
                model_dim=32,
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                ),
                prompt_cfg=dict(
                    tokenizer_model_path='/vepfs-cnbje63de6fae220/chengy/code/mobile_pi/checkpoints_pi/paligemma_tokenizer',
                    max_length=200,
                    discrete_state_input=True,
                ),
            ),
            sampler=dict(type='DefaultSampler', shuffle=True),
        ),
    ),
    models=dict(
        pretrained='/vepfs-cnbje63de6fae220/chengy/code/mobile_pi/checkpoints_pi/pi05_torch',
    ),
    optimizers=dict(
        type='AdamW',
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-10,
    ),
    schedulers=dict(
        type='WarmupCosineScheduler',
        warmup_steps=1000,
        decay_steps=30000,
        end_value=0.1,
    ),
    train=dict(
        resume=False,
        max_steps=50000,
        gradient_accumulation_steps=1,
        mixed_precision='no',
        checkpoint_interval=10000,
        checkpoint_total_limit=10,
        checkpoint_keeps=[10000,15000,20000,25000,30000,35000,40000,45000,50000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        dynamo_config=dict(backend='inductor'),
        activation_checkpointing=True,
        activation_class_names=[
            'SiglipEncoderLayer',
            'GemmaDecoderLayerWithExpert__##__1',
        ],
    ),
)
