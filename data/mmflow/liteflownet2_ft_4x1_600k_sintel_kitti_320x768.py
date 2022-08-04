img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
crop_size = (320, 768)
sintel_global_transform = dict(
    translates=(0.05, 0.05),
    zoom=(1.0, 1.2),
    shear=(0.95, 1.1),
    rotate=(-5.0, 5.0))
sintel_relative_transform = dict(
    translates=(0.00375, 0.00375),
    zoom=(0.985, 1.015),
    shear=(1.0, 1.0),
    rotate=(-1.0, 1.0))
sintel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.2),
            shear=(0.95, 1.1),
            rotate=(-5.0, 5.0)),
        relative_transform=dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0))),
    dict(type='RandomCrop', crop_size=(320, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'])
]
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                   'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'scale_factor', 'pad_shape'])
]
sintel_clean_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.05, 0.05),
                zoom=(1.0, 1.2),
                shear=(0.95, 1.1),
                rotate=(-5.0, 5.0)),
            relative_transform=dict(
                translates=(0.00375, 0.00375),
                zoom=(0.985, 1.015),
                shear=(1.0, 1.0),
                rotate=(-1.0, 1.0))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')
sintel_final_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.05, 0.05),
                zoom=(1.0, 1.2),
                shear=(0.95, 1.1),
                rotate=(-5.0, 5.0)),
            relative_transform=dict(
                translates=(0.00375, 0.00375),
                zoom=(0.985, 1.015),
                shear=(1.0, 1.0),
                rotate=(-1.0, 1.0))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')
sintel_clean_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                       'ori_filename2', 'ori_shape', 'img_shape',
                       'img_norm_cfg', 'scale_factor', 'pad_shape'])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='clean')
sintel_final_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                       'ori_filename2', 'ori_shape', 'img_shape',
                       'img_norm_cfg', 'scale_factor', 'pad_shape'])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')
sparse_global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
sparse_relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
sparse_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.02, 0.02),
            zoom=(0.98, 1.02),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5)),
        relative_transform=dict(
            translates=(0.0025, 0.0025),
            zoom=(0.99, 1.01),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5))),
    dict(type='RandomCrop', crop_size=(320, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'])
]
sparse_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputResize', exponent=6),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]
kitti2015_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.04),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.02, 0.02),
                zoom=(0.98, 1.02),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5)),
            relative_transform=dict(
                translates=(0.0025, 0.0025),
                zoom=(0.99, 1.01),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    test_mode=False)
kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True)
kitti2012_train = ({
    'type':
    'KITTI2012',
    'data_root':
    'data/kitti2012',
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'sparse': True
    }, {
        'type': 'ColorJitter',
        'brightness': 0.5,
        'contrast': 0.5,
        'saturation': 0.5,
        'hue': 0.5
    }, {
        'type': 'RandomGamma',
        'gamma_range': (0.7, 1.5)
    }, {
        'type': 'Normalize',
        'mean': [0.0, 0.0, 0.0],
        'std': [255.0, 255.0, 255.0],
        'to_rgb': False
    }, {
        'type': 'GaussianNoise',
        'sigma_range': (0, 0.04),
        'clamp_range': (0.0, 1.0)
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'horizontal'
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'vertical'
    }, {
        'type': 'RandomAffine',
        'global_transform': {
            'translates': (0.02, 0.02),
            'zoom': (0.98, 1.02),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        },
        'relative_transform': {
            'translates': (0.0025, 0.0025),
            'zoom': (0.99, 1.01),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        }
    }, {
        'type': 'RandomCrop',
        'crop_size': (320, 768)
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type':
        'Collect',
        'keys': ['imgs', 'flow_gt', 'valid'],
        'meta_keys':
        ['img_fields', 'ann_fields', 'filename1', 'filename2', 'ori_filename1',
         'ori_filename2', 'filename_flow', 'ori_filename_flow', 'ori_shape',
         'img_shape', 'img_norm_cfg']
    }],
    'test_mode':
    False
}, )
kitti2012_val_test = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape'
            ])
    ],
    test_mode=True)
data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[{
        'type':
        'Sintel',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations'
        }, {
            'type': 'ColorJitter',
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.05, 0.05),
                'zoom': (1.0, 1.2),
                'shear': (0.95, 1.1),
                'rotate': (-5.0, 5.0)
            },
            'relative_transform': {
                'translates': (0.00375, 0.00375),
                'zoom': (0.985, 1.015),
                'shear': (1.0, 1.0),
                'rotate': (-1.0, 1.0)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'data_root':
        'data/Sintel',
        'test_mode':
        False,
        'pass_style':
        'clean'
    }, {
        'type':
        'Sintel',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations'
        }, {
            'type': 'ColorJitter',
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.05, 0.05),
                'zoom': (1.0, 1.2),
                'shear': (0.95, 1.1),
                'rotate': (-5.0, 5.0)
            },
            'relative_transform': {
                'translates': (0.00375, 0.00375),
                'zoom': (0.985, 1.015),
                'shear': (1.0, 1.0),
                'rotate': (-1.0, 1.0)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'data_root':
        'data/Sintel',
        'test_mode':
        False,
        'pass_style':
        'final'
    }, {
        'type':
        'KITTI2015',
        'data_root':
        'data/kitti2015',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations',
            'sparse': True
        }, {
            'type': 'ColorJitter',
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'GaussianNoise',
            'sigma_range': (0, 0.04),
            'clamp_range': (0.0, 1.0)
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.02, 0.02),
                'zoom': (0.98, 1.02),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            },
            'relative_transform': {
                'translates': (0.0025, 0.0025),
                'zoom': (0.99, 1.01),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt', 'valid'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'test_mode':
        False
    },
           ({
               'type':
               'KITTI2012',
               'data_root':
               'data/kitti2012',
               'pipeline': [{
                   'type': 'LoadImageFromFile'
               }, {
                   'type': 'LoadAnnotations',
                   'sparse': True
               }, {
                   'type': 'ColorJitter',
                   'brightness': 0.5,
                   'contrast': 0.5,
                   'saturation': 0.5,
                   'hue': 0.5
               }, {
                   'type': 'RandomGamma',
                   'gamma_range': (0.7, 1.5)
               }, {
                   'type': 'Normalize',
                   'mean': [0.0, 0.0, 0.0],
                   'std': [255.0, 255.0, 255.0],
                   'to_rgb': False
               }, {
                   'type': 'GaussianNoise',
                   'sigma_range': (0, 0.04),
                   'clamp_range': (0.0, 1.0)
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'horizontal'
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'vertical'
               }, {
                   'type': 'RandomAffine',
                   'global_transform': {
                       'translates': (0.02, 0.02),
                       'zoom': (0.98, 1.02),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   },
                   'relative_transform': {
                       'translates': (0.0025, 0.0025),
                       'zoom': (0.99, 1.01),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   }
               }, {
                   'type': 'RandomCrop',
                   'crop_size': (320, 768)
               }, {
                   'type': 'DefaultFormatBundle'
               }, {
                   'type':
                   'Collect',
                   'keys': ['imgs', 'flow_gt', 'valid'],
                   'meta_keys': ['img_fields', 'ann_fields', 'filename1',
                                 'filename2', 'ori_filename1', 'ori_filename2',
                                 'filename_flow', 'ori_filename_flow',
                                 'ori_shape', 'img_shape', 'img_norm_cfg']
               }],
               'test_mode':
               False
           }, )],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final'),
            dict(
                type='KITTI2015',
                data_root='data/kitti2015',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True),
            dict(
                type='KITTI2012',
                data_root='data/kitti2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True)
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final'),
            dict(
                type='KITTI2015',
                data_root='data/kitti2015',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True),
            dict(
                type='KITTI2012',
                data_root='data/kitti2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', sparse=True),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'valid', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True)
        ],
        separate_eval=True))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl', port=12345)
log_level = 'INFO'
load_from = 'work_dir/lite2t/latest.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='LiteFlowNet',
    encoder=dict(
        type='NetC',
        in_channels=3,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(32, 32, 64, 96, 128, 192),
        strides=(1, 2, 2, 2, 2, 2),
        num_convs=(1, 3, 2, 2, 1, 1),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None),
    decoder=dict(
        type='NetE',
        in_channels=dict(level3=64, level4=96, level5=128, level6=192),
        corr_channels=dict(level3=49, level4=49, level5=49, level6=49),
        sin_channels=dict(level3=130, level4=194, level5=258, level6=386),
        rin_channels=dict(level3=131, level4=131, level5=131, level6=195),
        feat_channels=64,
        mfeat_channels=(128, 128, 96, 64, 32),
        sfeat_channels=(128, 128, 96, 64, 32),
        rfeat_channels=(128, 128, 64, 64, 32, 32),
        patch_size=dict(level3=5, level4=5, level5=3, level6=3),
        corr_cfg=dict(
            level3=dict(
                type='Correlation',
                max_displacement=3,
                stride=2,
                dilation_patch=2),
            level4=dict(type='Correlation', max_displacement=3),
            level5=dict(type='Correlation', max_displacement=3),
            level6=dict(type='Correlation', max_displacement=3)),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        flow_div=20.0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled_corr=False,
        regularized_flow=True,
        extra_training_loss=True,
        flow_loss=dict(
            type='MultiLevelCharbonnierLoss',
            resize_flow='upsample',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level0=0.000625),
            q=0.2,
            eps=0.01,
            reduction='sum'),
        init_cfg=None),
    train_cfg=dict(),
    test_cfg=dict())
optimizer = dict(
    type='Adam', lr=5e-05, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[200000, 300000, 400000, 500000])
runner = dict(type='IterBasedRunner', max_iters=600000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
work_dir = 'work_dir/lite2/lite2fts3'
gpu_ids = range(0, 1)
