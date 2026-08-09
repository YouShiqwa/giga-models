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
    runners=['pi0.pi05_dual_query_trainer.Pi05DualQueryTrainer'],
    project_dir='./experiments/vla/pi05/robocasa_set_up_cutting_station_dual_16query',
    launch=dict(
        gpu_ids=[4, 5, 6, 7],
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
            # 4 GPUs x 16 samples = global batch size 64.
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
        # Start from the original Pi0.5 checkpoint, exactly like the dual
        # baseline.  Its Action Expert is cloned into arm/base towers; only the
        # two learned query banks are newly initialized.
        pretrained='/vepfs-cnbje63de6fae220/chengy/code/mobile_pi/checkpoints_pi/pi05_torch',
        dual_action_expert=True,
        manipulation_action_indices=[0, 1, 2, 3, 4, 5, 6],
        mobility_action_indices=[7, 8, 9, 10, 11],
        manipulation_loss_weight=0.5,
        mobility_loss_weight=0.5,
        arm_num_query_tokens=16,
        base_num_query_tokens=16,
        query_init_std=0.02,
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
        checkpoint_keeps=[10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with=['tensorboard', 'wandb'],
        log_interval=1,
        wandb=dict(
            entity=None,
            mode='online',
            project='robocasa_pi',
            name='pi05_dual_16query_from_pi05',
            group='robocasa_set_up_cutting_station',
            job_type='train',
            tags=['robocasa', 'pi05', 'dual-action', '16-query', 'query-bridge'],
            config=dict(
                model_variant='pi05_dual_16query',
                initialization='pi05_torch',
                task='SetUpCuttingStation',
                dataset_split='target',
                seed=6666,
                global_batch_size=64,
                manipulation_action_dim=7,
                mobility_action_dim=5,
                arm_query_token_count=16,
                base_query_token_count=16,
                direct_action_to_vlm_attention=False,
            ),
        ),
        with_ema=True,
        dynamo_config=dict(backend='inductor'),
        activation_checkpointing=True,
        activation_class_names=[
            'SiglipEncoderLayer',
            'GemmaDecoderLayerWithExpert__##__1',
        ],
    ),
)
